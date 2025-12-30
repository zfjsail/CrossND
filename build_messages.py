#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""

# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# Adapted from

import logging
import os
from trainer import ModelArguments, CrossNDTrainer_v2, compute_metrics
from transformers import AutoTokenizer
from utils_new import *
from dataclasses import dataclass, field


# #!/bin/bash

# # 设置工作目录
# cd /workspace/pangyunhe/project/crossnd/llm
# pip install -r requirements.txt
# wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
# wandb offline
# wandb disabled
# wandb login 14a5316013f658f8ff2f0771a42ee134919be51b

# export WANDB_PROJECT=test
# # 设置训练设备
# # DEEPSPEED_GPUS="localhost:0,1,2,3,4,5,6,7"
# DEEPSPEED_GPUS="localhost:1"
# # 模型和数据参数
# # MODEL_PATH="/workspace/pangyunhe/models/Qwen/Qwen3-4B-Instruct-2507"
# MODEL_PATH="/workspace/pangyunhe/models/Qwen/Qwen3-8B"
# DATA_SRC="/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09_inout_sim.json"

# DATA_DIR="/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup"
# OUTPUT_DIR="output/kddcup/test"
# RUN_NAME="test"
# LOSS_TYPE="psl"
# NUM_TURN=10
# LABEL_THR=0.9

# # LoRA配置
# LORA_R=32
# LORA_ALPHA=64
# LORA_DROPOUT=0.05

# # 训练参数
# NUM_EPOCHS=4
# LEARNING_RATE=2e-5
# WEIGHT_DECAY=0.01
# WARMUP_RATIO=0.1
# TRAIN_BATCH_SIZE=1
# EVAL_BATCH_SIZE=1
# GRADIENT_ACCUMULATION=8
# EVAL_STEPS=0.1
# SAVE_STEPS=0.1
# # 运行训练命令
# deepspeed --master_port 29505  --include $DEEPSPEED_GPUS \
#     train.py \
#     --upsample true \
#     --max_seq_length 20000 \
#     --label_thr $LABEL_THR \
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
#     --eval_use_gather_object true \
#     --save_total_limit 2 \
#     --save_only_model true \
#     --dataloader_num_workers 20 \
#     --bf16 




@dataclass
class ModelArguments:
    """
    LoRA相关的参数
    """
    dataset:str = field(
        default = "kddcup"
    )
    data_dir:str = field(
        default = "/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup"
    )
    src: str = field(
        default= "/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09_inout_sim.json",
        metadata={"help":"数据集"}
    )
    model_path: str = field(
        default="/workspace/pangyunhe/models/Qwen/Qwen3-4B",
        metadata={"help": "模型路径"}
    )

    use_outer: bool = field(
        default=True,
        metadata={"help": "是否使用外部数据"}
    )
    num_turn: int = field(
        default=10,
        metadata={"help": "对话轮数"}
    )
    paper_slct_num: int = field(
        default=80
    )
    label_thr: float = field(
        default=0.7
    )
    author_sim: float = field(
        default=1.0
    )
    upsample: bool = field(
        default=True
    )
    use_hybrid_head: bool = field(  
        default=False
    )
    max_seq_length: int = field(
        default=18000,
        metadata={"help": "最大序列长度"}
    )




def main():
    # 使用HfArgumentParser处理命令行参数
    model_args = ModelArguments(num_turn=10,max_seq_length=22000)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)


    # 创建原始数据集实例
    train_dataset = CrossNDDataset(
        data_dir="/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup",
        tokenizer=tokenizer,
        model_args = model_args,
        mode="train",
        # noise_ratio=0.1, # for sft messages
        # drop=True,
    )
    
    # 创建验证数据集实例
    eval_dataset = CrossNDDataset(
        data_dir=model_args.data_dir,
        tokenizer=tokenizer,
        model_args = model_args,
        mode="eval",  # 使用评估模式
    )
    test_dataset = CrossNDDataset(
        data_dir=model_args.data_dir,
        tokenizer=tokenizer,
        model_args = model_args,
        mode="test",
    )
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    def build_dataset_parallel(dataset, num_workers=None, desc="Building Dataset"):
        """
        使用多线程并行构建数据集列表。
        
        Args:
            dataset: 实现了 __len__ 和 __getitem__ 的数据集对象
            num_workers: 线程数，默认为 CPU 核心数 * 2 (适合 I/O 密集型) 或 CPU 核心数 (适合计算密集型)
            desc: 进度条显示的描述文字
        
        Returns:
            list: 包含所有数据项的列表，顺序与原 dataset 一致
        """
        # 如果未指定线程数，默认使用 min(32, cpu_count + 4) 以避免过度开销
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
        
        data_length = len(dataset)
        results = [None] * data_length
        
        # 定义获取单个数据的辅助函数
        def get_item(idx):
            return idx, dataset[idx]

        print(f"Start processing {desc} with {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            # 注意：这里我们提交任务，future 会在后台执行
            futures = {executor.submit(get_item, i): i for i in range(data_length)}
            
            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(futures), total=data_length, desc=desc):
                try:
                    idx, item = future.result()
                    results[idx] = item
                except Exception as e:
                    print(f"Error processing index {futures[future]}: {e}")
                    
        return results

    # ==========================================
    # 调用部分 (替换你原来的最后三行代码)
    # ==========================================

    # train_data = [train_dataset[i] for i in range(len(train_dataset))]
    # breakpoint()
    # 1. 构建训练集
    train_data = build_dataset_parallel(
        train_dataset, 
        num_workers=32,  # 你可以根据服务器性能调整这个数字
        desc="Train Data",
    )

    # 2. 构建验证集
    val_data = build_dataset_parallel(
        eval_dataset, 
        num_workers=32, 
        desc="Valid Data"
    )

    # 3. 构建测试集
    test_data = build_dataset_parallel(
        test_dataset, 
        num_workers=32, 
        desc="Test Data"
    )
        #build_with multi processor
    
    train_data = [item for index,item in enumerate(train_dataset)]
    val_data = [item for index,item in enumerate(eval_dataset)]
    test_data = [item for index,item in enumerate(test_dataset)]

    import pandas as pd
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    valid_data= pd.DataFrame(val_data)
    #to parquet data
    train_data.to_parquet("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/noise_train_turn20.parquet")
    test_data.to_parquet("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/test_turn20.parquet")
    valid_data.to_parquet("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/valid_turn20.parquet")
    # save_to

    # # Save to parquet files
    # train_df = pd.DataFrame(train_data)
    # test_df = pd.DataFrame(test_data)

    # train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    # test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    
    # with open("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/train.json",'w') as f:
    #     json.dump(train_data,f)
    # with open("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/val.json",'w') as f:
    #     json.dump(val_data,f)
    # with open("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/test.json",'w') as f:
    #     json.dump(test_data,f)

if __name__ == "__main__":
    main()
    