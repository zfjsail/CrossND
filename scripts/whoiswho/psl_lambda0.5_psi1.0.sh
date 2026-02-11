#!/bin/bash

# 设置工作目录

pip install -r requirements.txt


# 设置训练设备
DEEPSPEED_GPUS="localhost:0,1,2,3,4,5,6,7"
# DEEPSPEED_GPUS="localhost:0"
# 模型和数据参数
MODEL_PATH="/workspace/pangyunhe/models/Qwen/your_model_path"
DATA_SRC="whoiswho_data/self_clean_float_halfsubset_pinout_sim.json"

DATA_DIR="whoiswho_data"
OUTPUT_DIR="output/whoiswho/psl_lambda0.5_psi1.0"
RUN_NAME="whoiswho_psl_lambda0.5_psi1.0"
LOSS_TYPE="psl_v2"
NUM_TURN=10
PSL_LAMBDA=0.5
PSL_PSI=1.0
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
# deepspeed --master_port 29500  --include $DEEPSPEED_GPUS \
#     train.py \
#     --dataset whoiswho \
#     --label_thr 0.9 \
#     --hybrid_train false \
#     --paper_slct_num 100 \
#     --loss_type $LOSS_TYPE \
#     --use_binary_head true \
#     --use_outer true \
#     --src $DATA_SRC \
#     --model_path $MODEL_PATH \
#     --data_dir $DATA_DIR \
#     --lora_r $LORA_R \
#     --lora_alpha $LORA_ALPHA \
#     --lora_dropout $LORA_DROPOUT \
#     --output_dir $OUTPUT_DIR \
#     --do_eval \
#     --num_turn $NUM_TURN \
#     --eval_strategy steps \
#     --eval_steps $EVAL_STEPS \
#     --save_steps $SAVE_STEPS \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
#     --eval_accumulation_steps 1 \
#     --run_name $RUN_NAME \
#     --deepspeed "config/ds_zero_1.json" \
#     --num_train_epochs $NUM_EPOCHS \
#     --per_device_train_batch_size $TRAIN_BATCH_SIZE \
#     --per_device_eval_batch_size $EVAL_BATCH_SIZE \
#     --logging_steps 1 \
#     --learning_rate $LEARNING_RATE \
#     --weight_decay $WEIGHT_DECAY \
#     --warmup_ratio $WARMUP_RATIO \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.1 \
#     --save_strategy steps \
#     --metric_for_best_model AUC_MAP \
#     --greater_is_better true \
#     --gradient_checkpointing \
#     --include_inputs_for_metrics true \
#     --remove_unused_columns false \
#     --load_best_model_at_end true \
#     --eval_use_gather_object true \
#     --save_total_limit 2 \
#     --psl_lambda $PSL_LAMBDA \
#     --psl_psi $PSL_PSI \
#     --report_to none \
#     --bf16 

# echo "训练完成，开始使用多个seed进行推理..."

# # 设置推理的seed列表
# SEEDS=(42 84 123 456 789)

# # 循环遍历每个seed进行推理
# for SEED in "${SEEDS[@]}"
# do    
#     # 运行推理命令
#     deepspeed --master_port 29500  --include $DEEPSPEED_GPUS \
#         inference.py \
#         --lora_path "/workspace/pangyunhe/project/crossnd/crossnd2/crossnd/clean_checkpoints/psl_lambda0.5_psi1.0" \
#         --dataset whoiswho \
#         --label_thr 0.9 \
#         --hybrid_train false \
#         --paper_slct_num 100 \
#         --loss_type $LOSS_TYPE \
#         --use_binary_head true \
#         --use_outer true \
#         --src $DATA_SRC \
#         --model_path $MODEL_PATH \
#         --data_dir $DATA_DIR \
#         --lora_r $LORA_R \
#         --lora_alpha $LORA_ALPHA \
#         --lora_dropout $LORA_DROPOUT \
#         --output_dir $OUTPUT_DIR \
#         --do_eval \
#         --num_turn $NUM_TURN \
#         --eval_strategy steps \
#         --eval_steps $EVAL_STEPS \
#         --save_steps $SAVE_STEPS \
#         --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
#         --eval_accumulation_steps 1 \
#         --run_name "${RUN_NAME}_inf_seed${SEED}" \
#         --num_train_epochs $NUM_EPOCHS \
#         --per_device_train_batch_size $TRAIN_BATCH_SIZE \
#         --per_device_eval_batch_size $EVAL_BATCH_SIZE \
#         --logging_steps 1 \
#         --learning_rate $LEARNING_RATE \
#         --weight_decay $WEIGHT_DECAY \
#         --warmup_ratio $WARMUP_RATIO \
#         --lr_scheduler_type cosine \
#         --warmup_ratio 0.1 \
#         --save_strategy steps \
#         --metric_for_best_model AUC_MAP \
#         --greater_is_better true \
#         --gradient_checkpointing \
#         --include_inputs_for_metrics true \
#         --remove_unused_columns false \
#         --load_best_model_at_end true \
#         --eval_use_gather_object true \
#         --seed $SEED \
#         --save_total_limit 2 \
#         --psl_lambda $PSL_LAMBDA \
#         --psl_psi $PSL_PSI \
#         --report_to none \
#         --bf16
# done

echo "多seed推理完成！"

# ============================================================
# TTS (Test-Time Selection) 策略推理
# 使用第一次推理的预测分数，对测试集论文顺序进行优化后重新推理
# ============================================================

echo "开始TTS策略推理..."

# 第一次推理的预测分数文件（作为TTS排序依据）
MULTITURN_PATH="${OUTPUT_DIR}/predictions_with_labels_test.json"

# TTS输出根目录
TTS_OUTPUT_ROOT="output/whoiswho/tts"

# 类希尔排序算法变体测试
# 每个策略跑5次，使用不同的seed
TTS_STRATEGIES=(
    # "turn_interleave_confidence_first"   # 类希尔排序+置信度：高低置信度交替
    # "turn_interleave_v2"                 # 类希尔排序v2：高-低-中三段交替
    # "turn_interleave_reverse"            # 类希尔排序反向：低高分交替
    # "turn_interleave_confidence_v2"      # 类希尔排序+置信度v2：高-低-中置信度三段交替
    # "turn_shell_sort"                    # 真·希尔排序：使用希尔增量序列
    "turn_confidence_desc"
)

# 5个不同的seed
TTS_SEEDS=(42 84 123 456 789)

for STRATEGY in "${TTS_STRATEGIES[@]}"
do
    echo "=========================================="
    echo "开始处理策略: ${STRATEGY}"
    echo "=========================================="
    
    TTS_OUTPUT_DIR="${TTS_OUTPUT_ROOT}/${STRATEGY}"
    echo "----------------------------------------"
    echo "TTS策略: ${STRATEGY}, Seed: ${SEED}"
    echo "输出目录: ${TTS_OUTPUT_DIR}"
    echo "----------------------------------------"
    for SEED in "${TTS_SEEDS[@]}"
    do
        deepspeed --master_port 29500  --include $DEEPSPEED_GPUS \
            inference.py \
            --lora_path "/workspace/pangyunhe/project/crossnd/crossnd2/crossnd/clean_checkpoints/psl_lambda0.5_psi1.0" \
            --dataset whoiswho \
            --seed $SEED \
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
            --output_dir $TTS_OUTPUT_DIR \
            --do_eval \
            --num_turn $NUM_TURN \
            --eval_strategy steps \
            --eval_steps $EVAL_STEPS \
            --save_steps $SAVE_STEPS \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
            --eval_accumulation_steps 1 \
            --run_name "${RUN_NAME}_tts_${STRATEGY}" \
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
            --multiturn_path $MULTITURN_PATH \
            --tts_strategy $STRATEGY \
            --report_to none \
            --bf16
    done
    echo "策略 ${STRATEGY} (Seed ${SEED}) 推理完成！"

    
    echo "策略 ${STRATEGY} 所有seed推理完成！"
done

echo "=========================================="
echo "所有类希尔排序策略多seed推理完成！"
echo "=========================================="
echo "结果保存在: ${TTS_OUTPUT_ROOT}/"
echo ""
echo "策略目录（每个策略5个seed，输出到同一目录）:"
for STRATEGY in "${TTS_STRATEGIES[@]}"; do
    echo "  - ${TTS_OUTPUT_ROOT}/${STRATEGY}/ (seeds: ${TTS_SEEDS[@]})"
done
echo ""
echo "总共推理: ${#TTS_STRATEGIES[@]} 个策略 × ${#TTS_SEEDS[@]} 个seed = $((${#TTS_STRATEGIES[@]} * ${#TTS_SEEDS[@]})) 次"
echo ""
echo "策略说明:"
echo "  1. turn_interleave_confidence_first: 类希尔排序+置信度，高低置信度交替"
echo "  2. turn_interleave_v2: 类希尔排序v2，高-低-中三段交替"
echo "  3. turn_interleave_reverse: 类希尔排序反向，低高分交替"
echo "  4. turn_interleave_confidence_v2: 类希尔排序+置信度v2，高-低-中置信度三段交替"
echo "  5. turn_shell_sort: 真·希尔排序，使用希尔增量序列"
