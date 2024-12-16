#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

HOME_PATH=$(echo ~)
source "$HOME_PATH/.bashrc"
micromamba activate pad

for seed in 0 1 2 3 4; do
    python data_gen/gen/sampling.py --model_name gemma-2-2b-it \
        --dataset data/ultrafeedback-split \
        --dataset_split train \
        --local \
        --max_tokens 4096 \
        --temperature 1.0 \
        --top_p 0.95 \
        --seed $seed \
        --output_dir data/generated/ultrafeedback/gemme-2b-it-step2
done
