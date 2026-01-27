#!/bin/bash

# 设置工作目录

pip install -r requirements.txt
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled
export WANDB_PROJECT=test

# 设置训练设备
DEEPSPEED_GPUS="localhost:0,1,2,3,4,5,6,7"
# DEEPSPEED_GPUS="localhost:0"
# 模型和数据参数
MODEL_PATH="/workspace/pangyunhe/models/Qwen/Qwen3-8B"
DATA_SRC="whoiswho_data/self_clean_float_halfsubset_pinout_sim.json"

DATA_DIR="whoiswho_data"
OUTPUT_DIR="output/whoiswho/psl_lambda0.5_psi1.4"
RUN_NAME="whoiswho_psl_lambda0.5_psi1.4"
LOSS_TYPE="psl_v2"
NUM_TURN=10
PSL_LAMBDA=0.5
PSL_PSI=1.4
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
    train.py \
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
    --psl_lambda $PSL_LAMBDA \
    --psl_psi $PSL_PSI \
    --bf16 

echo "训练完成，开始使用多个seed进行推理..."

# 设置推理的seed列表
SEEDS=(42 84 123 456 789)

# 循环遍历每个seed进行推理
for SEED in "${SEEDS[@]}"
do    
    # 运行推理命令
    deepspeed --master_port 29500  --include $DEEPSPEED_GPUS \
        inference.py \
        --lora_path "$OUTPUT_DIR/best_checkpoint" \
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
        --run_name "${RUN_NAME}_inf_seed${SEED}" \
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
        --seed $SEED \
        --save_total_limit 2 \
        --psl_lambda $PSL_LAMBDA \
        --psl_psi $PSL_PSI \
        --bf16
done

echo "所有推理任务完成！"
