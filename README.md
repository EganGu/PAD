# Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models

## Setup
Try the following commands to install the environment:
```sh
mamba env create -f environment.yml
```

## Data Generation
Try the following commands to generate the dataset:
```sh
bash scripts/sampling.sh
bash scripts/pipeline_n4_gemma.sh
```

## Training
Try following commands to train PAD model:
```sh
bash run_ppd.sh
```
You can find the trained model under `outputs/*`.

Please ensure that the file paths in the following file match your configuration:
**File:** `training_configs/gemma-2-2b-it-pd.yaml`

## Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, Arena-Hard, MT-Bench and GSM8K.

* AlpacaEval 2: Please refer to the [AlpacaEval repo](https://github.com/tatsu-lab/alpaca_eval) for evaluation.

* Arena-Hard: Please refer to to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

* MT-Bench: Please refer to the [FastChat repo](https://github.com/lm-sys/FastChat) for evaluation.

* GSM8K: Please refer to the [ZeroEval repo](https://github.com/WildEval/ZeroEval) for evaluation.


## Training Report

### Overview
This repository contains training logs and comparative analysis of three preference alignment methods: SimPO, DPO, and PAD. We document the training process, implementation details, and performance metrics for each approach.

### Implementations
- **DPO**: Based on the implementation from [TRL](https://github.com/huggingface/trl)
- **SimPO**: Based on the implementation from [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)

### Training Configuration

#### Models
- **Student Model**: Gemma-2-2B-It
- **Teacher Model**: Gemma-2-9B-It

#### Hardware
- **GPUs**: 2 × A800 (80G)

#### Training Parameters
- **Training Type**: Full parameter fine-tuning
- **Memory Optimization**: ZeRO Stage 2
- **Epochs**: 1
- **Precision**: BFloat16
- **Dataset Size**:
  - Training samples: 55,321
  - Test samples: 1,130
- **Batch Size**: 128
- **Total Training Steps**: 432
- **Maximum Sequence Length**: 2048
- **Per Device Train Batch Size**: 2
- **Per Device Evaluation Batch Size**: 2
- **Gradient Accumulation Steps**: 32
- **Evaluation Frequency**: Every 100 training steps
- **Gradient Checkpointing**: Enabled

For additional parameters, please refer to the paper or the configuration files.

### Results

| Method | GPU Hours | Alpaca-Eval 2.0 LC (%) |
|--------|-----------|---------------------------------|
| DPO    | 8.7856    | 43.77                           |
| SimPO  | 7.2672    | 44.94                           |
| PAD    | 7.2884    | 45.73                           |

You can find the training log under `gemma-log/*`.

#### Analysis
- **Training Efficiency**: PAD and SimPO require similar computational resources, while DPO demands notably more. This efficiency difference is primarily because DPO requires loading an additional reference model during training, whereas PAD and SimPO do not.
- **Performance**: PAD outperforms both SimPO and DPO in terms of win rate, which aligns with the findings reported in the submission paper.

