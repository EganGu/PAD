#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

HOME_PATH=$(echo ~)
source "$HOME_PATH/.bashrc"
micromamba activate pad

# 超参数定义
N=4  # 选项数量
TEMPERATURE=1  # 温度参数
TEACHER_MODEL=../huggingface/google/gemma-2-9b-it
TEACHER_ID=gemma
STUDENT_DIR=data/generated/ultrafeedback/gemma-2b-it

# 聚合数据
python data_gen/gen/agg.py --generation_file_dir $STUDENT_DIR -n $N

python data_gen/gen/prob_sl.py --model_name $TEACHER_MODEL \
    --temperature $TEMPERATURE \
    --input_file $STUDENT_DIR/agg_outputs_n$N.json \
    --output_file $STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.sl.json

python data_gen/gen/prob.py --model_name $TEACHER_MODEL \
    --temperature $TEMPERATURE \
    --num_options $N \
    --input_file $STUDENT_DIR/agg_outputs_n$N.json \
    --output_file $STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.json

python data_gen/gen/generate_dataset.py \
    --prob_file $STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.json \
    --prob_sl_file $STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.sl.json \
    --output_dir $STUDENT_DIR/pkd-dataset-teacher-$TEACHER_ID-n$N
