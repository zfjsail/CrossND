#!/bin/bash

# 使用 IterableDataset + 动态 num_turn 调度的训练脚本
# 特点：
# - 流式生成数据，无需重构
# - 训练集和验证集 num_turn 同步
# - 指数增长：1 -> 2 -> 4 -> 8 -> 16 ...

cd /workspace/pangyunhe/project/crossnd/llm
pip install -r requirements.txt > /dev/null 2>&1
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b > /dev/null 2>&1
wandb online > /dev/null 2>&1
wandb enabled > /dev/null 2>&1

export WANDB_PROJECT=crossnd_kddcup_iterable

# 设置训练设备
# DEEPSPEED_GPUS="localhost:0,1,2,3,4,5,6,7"
DEEPSPEED_GPUS="localhost:3"

# 模型和数据参数
MODEL_PATH="/workspace/pangyunhe/models/Qwen/Qwen3-8B"
DATA_SRC="/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09_inout_sim.json"
DATA_DIR="/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup"
OUTPUT_DIR="output/kddcup/train_iterable_dataset"
RUN_NAME="train_iterable_dataset"
LOSS_TYPE="psl_v2"
NUM_TURN=32
LABEL_THR=0.9

# LoRA 配置
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

# 训练参数
NUM_EPOCHS=10
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
EVAL_STEPS=0.1
SAVE_STEPS=0.1

# 运行训练命令
deepspeed --master_port 29555 --include $DEEPSPEED_GPUS \
    train.py \
    --upsample true \
    --max_seq_length 10000 \
    --label_thr $LABEL_THR \
    --hybrid_train false \
    --paper_slct_num 80 \
    --loss_type $LOSS_TYPE \
    --use_binary_head false \
    --use_outer true \
    --src $DATA_SRC \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --do_eval \
    --num_turn $NUM_TURN \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --eval_accumulation_steps 1 \
    --run_name $RUN_NAME \
    --deepspeed "config/ds_zero_2.json" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --logging_steps 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --metric_for_best_model AUC_MAP \
    --greater_is_better true \
    --gradient_checkpointing \
    --include_inputs_for_metrics true \
    --remove_unused_columns false \
    --eval_use_gather_object true \
    --save_total_limit 2 \
    --save_only_model true \
    --dataloader_num_workers 20 \
    --use_num_turn_scheduler true \
    --use_iterable_dataset true \
    --num_turn_schedule_type exponential \
    --bf16

