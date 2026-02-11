#!/bin/bash

# 设置训练设备
DEEPSPEED_GPUS="localhost:0,1,2,3,4,5,6,7"
# DEEPSPEED_GPUS="localhost:4"
# 模型和数据参数
MODEL_PATH="/workspace/pangyunhe/models/Qwen/your_model_path"
DATA_SRC="whoiswho_data/self_clean_float_halfsubset_pinout_sim.json"

DATA_DIR="whoiswho_data"


OUTPUT_DIR="output/whoiswho/psl"
RUN_NAME="whoiswho_psl"
LOSS_TYPE="psl_v2"
NUM_TURN=10
# LoRA配置
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 训练参数
NUM_EPOCHS=3
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
EVAL_STEPS=0.05
SAVE_STEPS=0.05
# 运行训练命令
deepspeed --master_port 29500  --include $DEEPSPEED_GPUS \
    inference.py \
    --lora_path "output/whoiswho/psl_lambda0.8_psi1.0/checkpoint-888" \
    --dataset whoiswho \
    --label_thr 0.9 \
    --hybrid_train false \
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
    --seed 84 \
    --save_total_limit 2 \
    --report_to none \
    --bf16 