#!/bin/bash

module load compilers/gcc-13.1.0

HOME_PATH=$(echo ~)
source "$HOME_PATH/.bashrc"
micromamba activate ygguv2

python data_gen/gen/split.py -k 5 \
    --input_file data/generated/ultrafeedback/gemma-2b-it/agg_outputs_n5.json \
    --output_dir data/generated/ultrafeedback/gemma-2b-it/split_n5

python data_gen/gen/split.py -k 5 \
    --input_file data/generated/ultrafeedback/llama-3b-it/agg_outputs_n5.json \
    --output_dir data/generated/ultrafeedback/llama-3b-it/split_n5
