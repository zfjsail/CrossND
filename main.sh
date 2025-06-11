#!/bin/bash

# 设置工作目录
cd /home/zhipuai/zhangfanjin-15T/pyh/project/crossnd/llm

# 配置wandb
# wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
# wandb online
# wandb enabled
export WANDB_PROJECT=crossnd

# 设置训练设备
DEEPSPEED_GPUS="localhost:0,1,2,3"

# 模型和数据参数
MODEL_PATH="/home/zhipuai/zhangfanjin-15T/pyh/models/models/Qwen/Qwen3-4B"
DATA_DIR="/home/zhipuai/zhangfanjin-15T/pyh/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup"
OUTPUT_DIR="output/qwen3b-cls-lora-train-set-wo-truedata"
NUM_TURN=1

# LoRA配置
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 训练参数
NUM_EPOCHS=2
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION=16
EVAL_STEPS=0.1
SAVE_STEPS=0.1
# 运行训练命令
deepspeed --include $DEEPSPEED_GPUS \
    pipeline.py \
    --data_dir $DATA_DIR \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --output_dir $OUTPUT_DIR \
    --do_eval \
    --num_turn $NUM_TURN \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --run_name "crossnd" \
    --deepspeed "config/ds_zero_2.json" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --logging_steps 1 \
    --save_only_model \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --metric_for_best_model AUC \
    --greater_is_better true \
    --gradient_checkpointing \
    --include_inputs_for_metrics true \
    --bf16 > output-double-source.log 2>&1