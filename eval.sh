#!/bin/bash

# 设置工作目录
cd /home/zhipuai/zhangfanjin-15T/pyh/project/crossnd/llm

# 模型和数据参数（通用配置）
BASE_MODEL_PATH="/home/zhipuai/zhangfanjin-15T/pyh/models/models/Qwen/Qwen3-4B"
DATA_DIR="/home/zhipuai/zhangfanjin-15T/pyh/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup"
BATCH_SIZE=8
NUM_TURN=1
CHECKPOINTS=(22 44 66 88 110 132 154 176 198 214)

for CKPT in "${CHECKPOINTS[@]}"
do
    MODEL_PATH="output/qwen3b-cls-lora-train-set-wo-truedata/checkpoint-${CKPT}"
    SAVE_PATH="output/double_eval/checkpoint-${CKPT}.json"
    # 第一个模型的评估（对应main.sh的配置）
    accelerate launch --num_processes=4 --gpu_ids=0,1,2,3 eval.py \
        --base_model_path $BASE_MODEL_PATH \
        --model_path $MODEL_PATH \
        --test_data_path $DATA_DIR \
        --output_file "evaluation_results_with_outer.json" \
        --predictions_file "model_predictions_with_outer.json" \
        --batch_size $BATCH_SIZE \
        --num_turn $NUM_TURN \
        --apply_chat_template true \
        --use_outer true \
        --save_path $SAVE_PATH

    MODEL_PATH="output/qwen4b-cls-single-source/checkpoint-${CKPT}"
    SAVE_PATH="output/single_eval/checkpoint-${CKPT}.json"
    accelerate launch --num_processes=4 --gpu_ids=0,1,2,3 eval.py \
        --base_model_path $BASE_MODEL_PATH \
        --model_path $MODEL_PATH \
        --test_data_path $DATA_DIR \
        --output_file "evaluation_results_without_outer.json" \
        --predictions_file "model_predictions_without_outer.json" \
        --batch_size $BATCH_SIZE \
        --num_turn $NUM_TURN \
        --apply_chat_template true \
        --use_outer false \
        --save_path $SAVE_PATH
done
