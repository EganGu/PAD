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