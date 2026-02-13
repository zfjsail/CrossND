#!/bin/bash

# 设置工作目录
cd /workspace/pangyunhe/project/crossnd/llm/crossnd

# 设置推理设备
DEEPSPEED_GPUS="localhost:0,1,2,3,4,5,6,7"

# 模型和数据参数
MODEL_PATH="Qwen/Qwen3-8B"  # 使用 Hugging Face 模型路径
DATA_SRC="kddcup_data/alldata_nd_thr09_inout_sim.json"
DATA_DIR="kddcup_data"
LORA_PATH="output/Qwen8B/cls_outer_v4_psl/checkpoint-200"
OUTPUT_DIR="output/inference"

# LoRA配置
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 推理参数
LOSS_TYPE="psl_v2"
NUM_TURN=10
PAPER_SLCT_NUM=100
EVAL_BATCH_SIZE=1
RUN_NAME="crossnd_inference"

# 运行推理命令
deepspeed --master_port 29501 --include $DEEPSPEED_GPUS \
    ../inference.py \
    --lora_path $LORA_PATH \
    --paper_slct_num $PAPER_SLCT_NUM \
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
    --eval_steps 1.0 \
    --save_steps 1.0 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 1 \
    --run_name $RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --logging_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --save_strategy steps \
    --metric_for_best_model AUC_MAP \
    --greater_is_better true \
    --gradient_checkpointing \
    --include_inputs_for_metrics true \
    --remove_unused_columns false \
    --load_best_model_at_end true \
    --eval_use_gather_object true \
    --save_total_limit 1 \
    --bf16

echo ""
echo "========================================"
echo "推理完成！"
echo "========================================"
echo "结果保存在: $OUTPUT_DIR"
echo "========================================"
