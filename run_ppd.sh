#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
# export WANDB_API_KEY="xxx"

HOME_PATH=$(echo ~)
source "$HOME_PATH/.bashrc"
micromamba activate pad

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed-zero2-2gpus-p25600.yaml scripts/run_pad.py training_configs/gemma-2-2b-it-pd.yaml
