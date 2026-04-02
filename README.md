# Gemma-3-27B-Taigi: A DeepSeek-R1 Inspired Training Pipeline

This repository contains the complete pipeline for fine-tuning **Gemma-3-27B-it** for **Taiwanese Hokkien (Taigi)** and Traditional Chinese. Our methodology is inspired by the **DeepSeek-R1** training procedure, currently implementing the **CPT** and **SFT** stages, with architecture ready for future **GRPO** (Group Relative Policy Optimization) reinforcement learning.

---

## Project Highlights

* **Base Model**: Google Gemma-3-27B-it.
* **Technique**: QLoRA (4-bit) for efficient large-parameter tuning.
* **Target Language**: Taiwanese Hokkien (Hanzi & Romanization) and Traditional Chinese.
* **Inspiration**: Follows the DeepSeek-R1 philosophy of building a solid linguistic foundation before reasoning alignment.

---

## Two-Stage Training Pipeline

The entire process is integrated into a single script `cpt_sft_final.py`, featuring automated stage transition and memory management.

### Phase 1: Continual Pre-Training (CPT)
* **Goal**: Expand the model's knowledge base with Taigi literature and dictionaries.
* **Key Datasets**: 
    * `moedict.json`: Ministry of Education Dictionary of Frequently-Used Taiwanese Taigi.
    * `taigi-literature`: A collection of classical and modern Taigi literary texts.
* **Strategy: `SaveBestCheckpointsCallback`**
    * **Metric**: Minimized **Evaluation Loss**.
    * **Benefit**: Ensures the model captures linguistic patterns without "catastrophic forgetting" of general knowledge.

### Phase 2: Supervised Fine-Tuning (SFT)
* **Goal**: Align the model to follow instructions and perform logical reasoning in Taigi.
* **Key Datasets**: 
    * `alpaca_gpt4_data_zh_tw.json`: Taigi-version Alpaca instruction set.
    * `tech_training_dataset.json`: Technical multiple-choice questions in Taigi.
* **Strategy: `AsyncGapMinimizationCallback`**
    * **Metric**: Minimized **Generalization Gap** ($|Avg(TrainLoss) - EvalLoss|$).
    * **Benefit**: Uses asynchronous sampling (Moving Average) to filter out overfitted checkpoints, providing a stable foundation for future RL stages.



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

We are preparing to implement **GRPO (Group Relative Policy Optimization)** as seen in DeepSeek-R1:
* **Reasoning Enhancement**: Developing rule-based rewards for Taigi logical puzzles.
* **Chain-of-Thought (CoT)**: Training the model to perform self-reflection and step-by-step reasoning in Taigi.

---

## Training Reports
The pipeline generates real-time ranking reports:
* `training_results_YYMMDD.txt`: Top CPT models by Eval Loss.
* `async_gap_results_YYMMDD.txt`: Top SFT models by Stability (Gap).

---

## License

