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
import pandas as pd
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




def main(num_turn=20,src=None):
    # 使用HfArgumentParser处理命令行参数
    model_args = ModelArguments(num_turn=num_turn,max_seq_length=30000,src=f"/workspace/pangyunhe/project/crossnd/llm/data/hard_data/train_data_{src}.json")
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
    
    train_path = os.path.join("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/hard_data", f"train_data_{src}_{num_turn}.parquet")
    test_path = os.path.join("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/hard_data", f"test_data_{num_turn}.parquet")
    val_path = os.path.join("/workspace/pangyunhe/project/crossnd/verl/data/crossnd/hard_data", f"val_data_{num_turn}.parquet")
    if not os.path.exists(train_path):
        train_data = build_dataset_parallel(
            train_dataset, 
            num_workers=32,  # 你可以根据服务器性能调整这个数字
            desc="Train Data",
        )
        train_data = [item for index,item in enumerate(train_dataset)]
        train_data = pd.DataFrame(train_data)
        train_data.to_parquet(train_path)

    if not os.path.exists(val_path):
        val_data = build_dataset_parallel(
            eval_dataset, 
            num_workers=32, 
            desc="Valid Data"
        )
        val_data = [item for index,item in enumerate(eval_dataset)]
        val_data = pd.DataFrame(val_data)
        val_data.to_parquet(val_path)

    if not os.path.exists(test_path):
        test_data = build_dataset_parallel(
            test_dataset, 
            num_workers=32, 
            desc="Test Data"
        )
        test_data = [item for index,item in enumerate(test_dataset)]
        test_data = pd.DataFrame(test_data)
        test_data.to_parquet(test_path)
    

if __name__ == "__main__":
    for src in [1000,2000,5000]:
    # for src in [2000]:
        for num_turn in [10,20]:
            main(num_turn=num_turn,src=src)
    