#!/bin/bash

# 设置工作目录
cd /workspace/pangyunhe/project/crossnd/llm
pip install -r requirements.txt
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled
export WANDB_PROJECT=crossnd

# 设置训练设备
# DEEPSPEED_GPUS="localhost:0,1,2,3,4,5,6,7"
DEEPSPEED_GPUS="localhost:1,2,3"
# 模型和数据参数
MODEL_PATH="/workspace/pangyunhe/models/Qwen/Qwen3-8B"
DATA_SRC="/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09.json"

DATA_DIR="/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup"
OUTPUT_DIR="output/Qwen8B/cls-hard-outer-hybrid-v4"
RUN_NAME="cls_hard_outer_hybrid-v4"
LOSS_TYPE="ce"
NUM_TURN=10
# LoRA配置
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 训练参数
NUM_EPOCHS=4
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
EVAL_STEPS=0.1
SAVE_STEPS=0.1
# 运行训练命令
deepspeed --master_port 29500  --include $DEEPSPEED_GPUS \
    train.py \
    --hybrid_train true \
    --paper_slct_num 100 \
    --loss_type $LOSS_TYPE \
    --use_binary_head true \
    --use_outer true \
    --src $DATA_SRC \
    --model_path $MODEL_PATH \
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
    --eval_accumulation_steps 1 \
    --run_name $RUN_NAME \
    --deepspeed "config/ds_zero_1.json" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --logging_steps 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --save_strategy steps \
    --metric_for_best_model AUC_MAP \
    --greater_is_better true \
    --gradient_checkpointing \
    --include_inputs_for_metrics true \
    --remove_unused_columns false \
    --load_best_model_at_end true \
    --eval_use_gather_object true \
    --save_total_limit 2 \
    --bf16 