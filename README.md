# SARC-Taigi-LLM
This repository provides a specialized, high-performance pipeline for fine-tuning **Gemma-3-12b-it** and **Gemma-3-27b-it** to excel in **Taiwanese (Taigi)**. Our method follows a structured progression from linguistic knowledge acquisition to sophisticated reasoning alignment.

---

## The Roadmap: A Three-Phase Evolution

Our project is designed as a three-stage technical stack to build a Taigi model with deep comprehension and self-reasoning capabilities.

### 1. Phase I: Continual Pre-Training (CPT) - *Completed*
* **Goal**: Expand the model's knowledge base with Taigi literature and formal dictionaries.
* **Strategy**: Uses **`SaveBestCheckpointsCallback`** to prioritize **Evaluation Loss**, ensuring a robust linguistic foundation without losing general knowledge.

### 2. Phase II: Supervised Fine-Tuning (SFT) - *Current Status*
* **Goal**: Align the model to follow instructions and perform logical tasks in Taigi.
* **Strategy**: Uses **`AsyncGapMinimizationCallback`** to minimize the **Generalization Gap**. By calculating the moving average of training loss against validation loss, we identify the most stable checkpoints for reliable dialogue.

### 3. Phase III: GRPO (Group Relative Policy Optimization) - *Upcoming*
* **Goal**: Transition from supervised learning to **Reinforcement Learning (RL)**.
* **Mechanism**: We plan to implement **GRPO** to enhance the model's reasoning chains (CoT) for complex Taigi linguistic puzzles and technical queries.
* **Benefit**: Unlike traditional PPO, GRPO will allow us to optimize the model using group-based relative rewards, significantly improving its self-correction and logical consistency in low-resource language contexts.

---

## Key Technical Features

### 1. Integrated Automated Workflow
The script `cpt_sft_final.py` automates the transition between phases, including an "Atomic Handoff" that identifies the **Rank 1** checkpoint from CPT and loads it as the seed for SFT.

### 2. Intelligent Checkpoint Selection
We move beyond simple step-based saving. Our callbacks monitor:
* **Knowledge Density**: Through precise Eval Loss tracking during CPT.
* **Generalization Stability**: Through asynchronous gap sampling during SFT, ensuring the model is "GRPO-ready" by minimizing overfitting.

---

## Technical Implementation Details

### 1. Smart Hardware Adaptation
The script automatically detects GPU counts via `torch.cuda.device_count()` and optimizes parameters for **DDP (Distributed Data Parallel)** mode:
* `use_reentrant=False`: Ensures stability for gradient checkpointing in distributed environments.
* `dataloader_num_workers` & `pin_memory`: Optimized data throughput to eliminate I/O bottlenecks.

### 2. Automated Weight Inheritance
Between CPT and SFT, the script performs an "Atomic Handoff":
1. **Rank 1 Extraction**: Automatically identifies the best checkpoint from the CPT leaderboard.
2. **DDP Synchronization**: Uses `dist.broadcast_object_list` to ensure all GPU nodes load the exact same "Best" path.
3. **Memory Recovery**: Forces `gc.collect()` and `cuda.empty_cache()` to prevent OOM during the 27B model swap.

### 3. Strict Checkpoint Interception
In SFT mode, the callback explicitly sets `control.should_save = False`. This overrides the default `Trainer` behavior, ensuring disk space is only used for "Generalization Champions" verified by the Gap strategy.

---

## Roadmap: Towards GRPO

We are preparing to implement **GRPO (Group Relative Policy Optimization)**:
* **Reasoning Enhancement**: Developing rule-based rewards for Taigi logical puzzles.
* **Chain-of-Thought (CoT)**: Training the model to perform self-reflection and step-by-step reasoning in Taigi.

---

## Training Reports
The pipeline generates real-time ranking reports:
* `training_results_YYMMDD.txt`: Top CPT models by Eval Loss.
* `async_gap_results_YYMMDD.txt`: Top SFT models by Stability (Gap).

---

## Model Download
> **SARC-Taigi-LLM** is now available on Hugging Face!  
> [**Download the Model Weights here**]([https://huggingface.co/Speech-AI-Research-Center/SARC-Taigi-LLM](https://huggingface.co/Speech-AI-Research-Center/SARC-Taigi-LLM))

---

## License

