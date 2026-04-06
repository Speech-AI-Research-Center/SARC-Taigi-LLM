import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling, 
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainerCallback, TrainerState, TrainerControl
import shutil
import math
import torch.distributed as dist
import datetime
import wandb
from peft import PeftModel

def format_lr(val):
    s = f"{val:e}"
    base, exp = s.split("e")
    print(s, base, exp)
    base = base.rstrip('0').rstrip('.')
    return f"{base}e{int(exp)}"

def main():
    gpu_num = torch.cuda.device_count()
    if gpu_num==0:
        print("No GPU detected!")
        return
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Configuration
    base_model_name = "google/gemma-3-12b-it"
    model_name = "google/gemma-3-12b-it"
    max_seq_length = 2880

    # 4-bit Quantization setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": local_rank},
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # PEFT/LoRA Initialization
    if model_name != base_model_name:
        model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
        model.peft_config["default"].lora_dropout = 0.1
    else: 
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
    model.config.pretraining_tp = 1
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True
    )
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
        
    # Custom Callback: Save best N checkpoints based on eval_loss
    class SaveBestCheckpointsCallback(TrainerCallback):
        def __init__(self, save_total_limit=5, outputDir=f"training_results_{datetime.datetime.now().strftime('%y%m%d')}.txt"):
            self.save_total_limit = save_total_limit
            self.best_loss = float('inf')
            self.checkpoints = []
            self.pending_save = None
            self.outputDir = outputDir
            
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs is None or 'eval_loss' not in logs:
                return control
            should_save_this = False
            if self._is_main_process(args):
                current_loss = logs['eval_loss']
                if len(self.checkpoints) < self.save_total_limit:
                    should_save_this = True
                    reason = f"Limit not reached ({len(self.checkpoints)}/{self.save_total_limit})"
                else:
                    worst_loss = self.checkpoints[-1][0]
                    if current_loss < worst_loss:
                        should_save_this = True
                        reason = f"Better than worst (Current {current_loss:.4f} < Worst {worst_loss:.4f})"
                    else:
                        reason = f"Not better than worst (Current {current_loss:.4f} >= Worst {worst_loss:.4f})"
                
                if should_save_this:
                    print(f"\n✓ Will save {self.outputDir.split('/')[-2]}/checkpoint-{state.global_step}: "
                          f"eval_loss={current_loss:.4f} ({reason})")
                    self.pending_save = {
                        'eval_loss': current_loss,
                        'step': state.global_step
                    }
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        print(f" 🎯 New Best Eval Loss!")
                else:
                    print(f"\n✗ Skip checkpoint-{state.global_step}: "
                          f"eval_loss={current_loss:.4f} ({reason})")
                    self.pending_save = None
            
            should_save_this = self._sync_decision(args, should_save_this)
            control.should_save = should_save_this
            return control
            
        def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if not self._is_main_process(args):
                return control
            if self.pending_save is None:
                return control
            checkpoint_folder = f"checkpoint-{self.pending_save['step']}"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
            if not os.path.exists(checkpoint_path):
                print(f"⚠️ Expected checkpoint path does not exist: {checkpoint_path}")
                self.pending_save = None
                return control
            self.checkpoints.append((
                self.pending_save['eval_loss'],
                self.pending_save['step'],
                checkpoint_path
            ))
            self.checkpoints.sort(key=lambda x: x[0])
            print(f"✓ Saved: {checkpoint_folder} (eval_loss={self.pending_save['eval_loss']:.4f})")
            
            if len(self.checkpoints) > self.save_total_limit:
                worst_loss, worst_step, path_to_remove = self.checkpoints.pop()
                if os.path.exists(path_to_remove):
                    try:
                        shutil.rmtree(path_to_remove)
                        print(f"✗ Removed checkpoint-{worst_step}: eval_loss={worst_loss:.4f} (Worst in list)")
                    except Exception as e:
                        print(f"⚠️ Error removing checkpoint-{worst_step}: {e}")
                        self.checkpoints.append((worst_loss, worst_step, path_to_remove))
                        self.checkpoints.sort(key=lambda x: x[0])
            
            with open(self.outputDir, 'w', encoding='utf-8') as f:
                print(f"\nCurrently keeping {len(self.checkpoints)} checkpoints:")
                f.write(f"Currently keeping {len(self.checkpoints)} checkpoints:\n")
                for i, (loss, step, _) in enumerate(self.checkpoints, 1):
                    marker = f"{i}."
                    print(f"   {marker} checkpoint-{step}: eval_loss={loss:.4f}")
                    f.write(f"   {marker} checkpoint-{step}: eval_loss={loss:.4f}\n")
                print()
                f.write("\n")
            self.pending_save = None
            return control
            
        def _is_main_process(self, args):
            if hasattr(args, 'local_rank'):
                return args.local_rank in [-1, 0]
            if hasattr(args, 'process_index'):
                return args.process_index == 0
            return True
            
        def _sync_decision(self, args, decision):
            if hasattr(args, 'local_rank') and args.local_rank != -1:
                try:
                    if dist.is_initialized():
                        import torch
                        decision_tensor = torch.tensor(
                            [1 if decision else 0],
                            dtype=torch.long,
                            device=args.device
                        )
                        dist.broadcast(decision_tensor, src=0)
                        return bool(decision_tensor.item())
                except Exception as e:
                    if self._is_main_process(args):
                        print(f"⚠️ Sync decision error: {e}, using main process decision")
                    return decision
            return decision

    # CPT Data Preparation
    print("Loading and preprocessing CPT dataset...")
    ots = load_dataset("IMA-Taiwan/taigi-literature-ots", split="train")
    tks = load_dataset("IMA-Taiwan/taigi-literature-tks", split="train")
    abt = load_dataset("IMA-Taiwan/taigi-literature-abt", split="train")
    kkh = load_dataset("IMA-Taiwan/taigi-literature-kkh", split="train")
    olbt = load_dataset("IMA-Taiwan/taigi-literature-olbt", split="train")
    ljk = load_dataset("IMA-Taiwan/taigi-literature-ljk", split="train")
    tsk = load_dataset("IMA-Taiwan/taigi-literature-tsk", split="train")
    manlajo = load_dataset("IMA-Taiwan/taigi-literature-manlajo", split="train")
    asts = load_dataset("IMA-Taiwan/taigi-literature-asts", split="train")
    achiak = load_dataset("IMA-Taiwan/taigi-literature-achiak", split="train")
    ngkh = load_dataset("IMA-Taiwan/taigi-literature-ngkh", split="train")
    ttshs = load_dataset("IMA-Taiwan/taigi-literature-ttshs", split="train")
    pikh = load_dataset("IMA-Taiwan/taigi-literature-pikh", split="train")
    khg = load_dataset("IMA-Taiwan/taigi-literature-khg", split="train")
    sslts = load_dataset("IMA-Taiwan/taigi-literature-sslts", split="train")
    lgs = load_dataset("IMA-Taiwan/taigi-literature-lgs", split="train")
    llb = load_dataset("IMA-Taiwan/taigi-literature-llb", split="train")
    
    dataset1 = concatenate_datasets([ots, tks, sslts, ljk, kkh, tsk, olbt, abt, manlajo, asts, achiak, ngkh, ttshs, pikh, khg, lgs, llb])
    dataset2 = load_dataset("json", data_files="moedict.json", split="train")

    # Formatting functions for literary and dictionary data
    wikipedia_prompt1 = """台語文學
    ### 標題：{}

    ### 文章：
    {}"""
    def formatting_prompts_func1(examples):
        titles = examples["title"]
        texts = examples["text"]
        outputs = []
        for title, text in zip(titles, texts):
            formatted_text = wikipedia_prompt1.format(title, text) + tokenizer.eos_token
            outputs.append(formatted_text)
        return {"text": outputs}
    dataset1 = dataset1.map(formatting_prompts_func1, batched = True, num_proc=4)

    wikipedia_prompt2 = """台語文本：
    {}"""
    def formatting_prompts_func2(examples):
        titles = examples["title"]
        texts = examples["text"]
        outputs = []
        for title, text in zip(titles, texts):
            formatted_text = wikipedia_prompt2.format(text) + tokenizer.eos_token
            outputs.append(formatted_text)
        return {"text": outputs}
    dataset2 = dataset2.map(formatting_prompts_func2, batched = True, num_proc=4)

    dataset= concatenate_datasets([dataset1, dataset2])
    dataset = dataset.shuffle(seed=42)
    print(dataset)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        return tokenized
    
    # Train/Val Split
    train_val_split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,
    )
    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=4,
    )

    # Dynamic parameter adjustment for Single vs Multi-GPU
    now = datetime.datetime.now().strftime('%y%m%d')
    lr = 2.5e-4
    if gpu_num == 1: train_batch_size, accumulation_steps, total_limit, train_epochs = 2, 12, 10, 1
    elif gpu_num == 2: train_batch_size, accumulation_steps, total_limit, train_epochs = 2, 6, 10, 1
    else: train_batch_size, accumulation_steps, total_limit, train_epochs = 1, 4, 10, 1
    outputDir = f"models/output_gemma3_12b_cpt_{now}_{format_lr(lr)}_b{train_batch_size*accumulation_steps}x{gpu_num}_{train_epochs}epo"
    runName = f"gemma3_12b_cpt_{now}_{format_lr(lr)}_b{train_batch_size*accumulation_steps}x{gpu_num}_{train_epochs}epo"
        
    dataloader_num = 4
    split_num = 25
    saveSteps = math.ceil(len(train_dataset)/(train_batch_size*accumulation_steps*gpu_num*split_num*10))
    maxSteps = int(saveSteps*math.ceil(split_num*train_epochs)*10)
    evalSteps = int(5*saveSteps)
    print(f"Dataset stats: {len(train_dataset)} items, saveSteps={saveSteps}, maxSteps={maxSteps}")
    torch.cuda.empty_cache()

    training_config = {
        "output_dir": outputDir,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "max_steps": maxSteps,
        "save_total_limit": None,
        "save_strategy": "steps",
        "learning_rate": lr,
        "bf16": True,
        "fp16": False,
        "eval_strategy": "steps",
        "eval_steps": evalSteps,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": False,
        "report_to": "wandb",
        "run_name": runName,
        "seed": 3407,
        "lr_scheduler_type": "linear",
        "save_steps": evalSteps,
        "logging_steps": evalSteps,
        "warmup_ratio": 0.05,
        "weight_decay": 0.005,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
    }
    
    is_distributed = int(os.environ.get("LOCAL_RANK", -1)) != -1
    if is_distributed:
        training_config.update({
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_num_workers": dataloader_num,
            "dataloader_pin_memory": True,
        })
    else:
        print("Running in Single GPU training mode.")
        
    training_args = TrainingArguments(**training_config)
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )
    save_callback = SaveBestCheckpointsCallback(
        save_total_limit=total_limit,
        outputDir=f"{outputDir}/training_results_{now}.txt"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        callbacks=[save_callback, early_stopping_callback],
    )
    
    print("Starting CPT...")
    trainer_stats = trainer.train()
    # trainer.save_model(f"models/final_gemma3_12b_cpt_{now}")
    # tokenizer.save_pretrained(f"models/final_gemma3_12b_cpt_{now}")
    print("CPT completed!")
    print(f"CPT stats: {trainer_stats}")
    wandb.finish()

    # Memory cleanup for Phase transition
    import gc
    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Broadcast Best CPT Path across nodes
    best_checkpoint_path = [None]
    if local_rank == 0:
        if len(save_callback.checkpoints) > 0:
            best_checkpoint_path[0] = save_callback.checkpoints[0][2]
        else:
            print("No checkpoints saved (CPT process might be too short)")
            return
    if dist.is_initialized():
        dist.broadcast_object_list(best_checkpoint_path, src=0)
        
    best_path = best_checkpoint_path[0]
    print(f"GPU {local_rank} loading Best CPT weights: {best_path}")
    
    # Reload model for SFT using best CPT weights
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, best_path, is_trainable=True)
    model.peft_config["default"].lora_dropout = 0.1
    tokenizer = AutoTokenizer.from_pretrained(
        best_path,
        trust_remote_code=True,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True
    )
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id



    # Custom Callback: Save checkpoints based on Generalization Gap (Train vs Eval Loss)
    class AsyncGapMinimizationCallback(TrainerCallback):
        def __init__(self, save_total_limit=5, outputDir=f"training_results_{datetime.datetime.now().strftime('%y%m%d')}.txt"):
            self.save_total_limit = save_total_limit
            self.train_loss_buffer = []
            self.checkpoints = []
            self.pending_save = None
            self.outputDir = outputDir
            
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs is None: return control
            control.should_save = False # Disable default saving
            if 'loss' in logs and state.global_step % args.save_steps == 0:
                self.train_loss_buffer.append(logs['loss'])
            
            if 'eval_loss' in logs:
                should_save_this = False
                current_eval_loss = logs['eval_loss']
                if self._is_main_process(args):
                    if len(self.train_loss_buffer) > 0:
                        avg_train_loss = sum(self.train_loss_buffer) / len(self.train_loss_buffer)
                        current_gap = abs(current_eval_loss - avg_train_loss)
                        print(f"\n[Step {state.global_step}] Settlement: Gap Stability:")
                        print(f" - Samples: {len(self.train_loss_buffer)}, Avg Train Loss: {avg_train_loss:.4f}")
                        print(f" - Eval Loss: {current_eval_loss:.4f}, Current Gap: {current_gap:.4f}")
                        
                        if len(self.checkpoints) < self.save_total_limit:
                            should_save_this = True
                            reason = "Limit not reached"
                        else:
                            max_gap_in_list = self.checkpoints[-1][0]
                            if current_gap < max_gap_in_list:
                                should_save_this = True
                                reason = f"Better stability than worst in list ({current_gap:.4f} < {max_gap_in_list:.4f})"
                            else:
                                reason = f"Gap is not competitive ({current_gap:.4f} >= {max_gap_in_list:.4f})"
                        
                        if should_save_this:
                            print(f"✓ Passed selection! Preparing to save. Reason: {reason}")
                            print(f"\n✓ Will save {self.outputDir.split('/')[-2]}/checkpoint-{state.global_step}")
                            self.pending_save = {
                                'gap': current_gap,
                                'eval_loss': current_eval_loss,
                                'step': state.global_step
                            }
                        else:
                            print(f"✗ Failed selection. Reason: {reason}")
                            self.pending_save = None
                    self.train_loss_buffer = []
                
                should_save_this = self._sync_decision(args, should_save_this)
                control.should_save = should_save_this
            return control
            
        def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if not self._is_main_process(args) or self.pending_save is None: return control
            checkpoint_folder = f"checkpoint-{self.pending_save['step']}"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
            if os.path.exists(checkpoint_path):
                self.checkpoints.append((
                    self.pending_save['gap'],
                    self.pending_save['eval_loss'],
                    self.pending_save['step'],
                    checkpoint_path
                ))
                self.checkpoints.sort(key=lambda x: x[0])
                if len(self.checkpoints) > self.save_total_limit:
                    worst_gap, _, worst_step, path_to_remove = self.checkpoints.pop()
                    if os.path.exists(path_to_remove):
                        try:
                            shutil.rmtree(path_to_remove)
                            print(f"✗ Removed unstable checkpoint-{worst_step} (Gap={worst_gap:.4f})")
                        except Exception as e:
                            print(f"⚠️ Removal failed: {e}")
                
                with open(self.outputDir, 'w', encoding='utf-8') as f:
                    header = f"Stability Priority Leaderboard (Samples every {args.save_steps} steps, Settlement every {args.eval_steps} steps)\n"
                    f.write(header)
                    print(f"\n{header}")
                    for i, (gap, loss, step, _) in enumerate(self.checkpoints, 1):
                        marker = f"{i}."
                        line = f"   {marker} checkpoint-{step}: Gap={gap:.4f} (Eval_Loss={loss:.4f})"
                        f.write(line + "\n")
                        print(line)
                    print()
            self.pending_save = None
            return control
            
        def _is_main_process(self, args):
            if hasattr(args, 'local_rank'):
                return args.local_rank in [-1, 0]
            if hasattr(args, 'process_index'):
                return args.process_index == 0
            return True
            
        def _sync_decision(self, args, decision):
            if hasattr(args, 'local_rank') and args.local_rank != -1:
                try:
                    if dist.is_initialized():
                        import torch
                        decision_tensor = torch.tensor([1 if decision else 0], dtype=torch.long, device=args.device)
                        dist.broadcast(decision_tensor, src=0)
                        return bool(decision_tensor.item())
                except:
                    return decision
            return decision
        
    # SFT Data Preparation
    print("Loading and preprocessing SFT dataset...")
    converted_list = []

    # 1. General Dialogue Dataset
    alpaca_dataset = load_dataset("json", data_files="alpaca_translated_dataset_taigi_zh_tw.json", split="train")
    for example in alpaca_dataset:
        instruction = example.get("instruction", "").strip()
        input_text  = example.get("input", "").strip()
        output_text = example.get("output", "").strip()
        user_content = (instruction + "\n" + input_text).strip("\n").strip()
        conversation = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output_text}
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        converted_list.append({"text": text})
    
    # 2. Technical/Logical Dataset
    tech_dataset = load_dataset("json", data_files="tech_training_dataset.json", split="train")
    for example in tech_dataset:
        input_text = example.get("Input", "").strip()
        output_text = example.get("Output", "").strip()
        conversation = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        converted_list.append({"text": text})

    dataset = Dataset.from_list(converted_list)
    dataset = dataset.shuffle(seed=42)
    print(dataset)
    
    # Train/Val Split for SFT
    train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        return model_inputs
        
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,
    )
    tokenized_val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=4,
    )

    # Dynamic parameter adjustment for Single vs Multi-GPU
    now = datetime.datetime.now().strftime('%y%m%d')
    lr = 2.5e-4
    if gpu_num == 1: train_batch_size, accumulation_steps, total_limit, train_epochs = 2, 96, 10, 1.5
    elif gpu_num == 2: train_batch_size, accumulation_steps, total_limit, train_epochs = 4, 24, 10, 1.5
    else: train_batch_size, accumulation_steps, total_limit, train_epochs = 2, 12, 2, 1.5
    outputDir = f"models/output_gemma3_12b_sft_{now}_{format_lr(lr)}_b{train_batch_size*accumulation_steps}x{gpu_num}_cosine_{train_epochs}epo_gap"
    runName = f"gemma3_12b_sft_{now}_{format_lr(lr)}_b{train_batch_size*accumulation_steps}x{gpu_num}_cosine_{train_epochs}epo_gap"
    
    dataloader_num = 4
    split_num = 25
    saveSteps = math.ceil(len(train_dataset)/(train_batch_size*accumulation_steps*gpu_num*split_num*10))
    maxSteps=int(saveSteps*math.ceil(split_num*train_epochs)*10)
    evalSteps = int(10*saveSteps)
    print(f"SFT stats: {len(train_dataset)} items, saveSteps={saveSteps}, maxSteps={maxSteps}")
    torch.cuda.empty_cache()
    
    training_config = {
        "output_dir": outputDir,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "max_steps": maxSteps,
        "save_total_limit": None,
        "save_strategy": "steps",
        "learning_rate": lr,
        "bf16": True,
        "fp16": False,
        "eval_strategy": "steps",
        "eval_steps": evalSteps,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": False,
        "report_to": "wandb",
        "run_name": runName,
        "seed": 3407,
        "lr_scheduler_type": "cosine",
        "save_steps": saveSteps,
        "logging_steps": saveSteps,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
    }

    is_distributed = int(os.environ.get("LOCAL_RANK", -1)) != -1
    if is_distributed:
        training_config.update({
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_num_workers": dataloader_num,
            "dataloader_pin_memory": True,
        })
    else:
        print("Running in Single GPU SFT mode.")
        
    training_args = TrainingArguments(**training_config)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )
    save_callback = AsyncGapMinimizationCallback(
        save_total_limit=total_limit,
        outputDir=f"{outputDir}/async_gap_results_{now}.txt"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[save_callback, early_stopping_callback],
    )
    
    print("Starting SFT...")
    trainer_stats = trainer.train()
    # trainer.save_model(f"models/final_gemma3_12b_sft_{now}")
    # tokenizer.save_pretrained(f"models/final_gemma3_12b_sft_{now}")
    print("SFT completed!")
    print(f"SFT stats: {trainer_stats}")

if __name__ == "__main__":
    main()