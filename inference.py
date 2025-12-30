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
import sys
import transformers
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers import TrainingArguments, Trainer,EarlyStoppingCallback
from trainer import DataArguments, ModelArguments, CrossNDTrainer_v2, compute_metrics

from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import *
from model import Qwen3ForCrossND, LlamaForCrossND

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    From https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main():
    # 使用HfArgumentParser处理命令行参数
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.include_inputs_for_metrics =[]
    training_args.include_for_metrics =["inputs"]
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

    config.use_cache = False
    # config._attn_implementation = "flash_attention_2" #use flash attention

    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
            
    # 根据模型路径选择合适的模型类
    model_path_lower = model_args.model_path.lower()
    if "qwen" in model_path_lower:
        logger.info(f"Initializing Qwen3ForCrossND model from {model_args.model_path}")
        model = Qwen3ForCrossND.from_pretrained(
            model_args.model_path, 
            torch_dtype=dtype,
            config=config, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).cuda()
    elif "llama" in model_path_lower:
        logger.info(f"Initializing LlamaForCrossND model from {model_args.model_path}")
        model = LlamaForCrossND.from_pretrained(
            model_args.model_path, 
            torch_dtype=dtype,
            config=config, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).cuda()
    else:
        raise ValueError(f"Unsupported model type in path: {model_args.model_path}. Path should contain 'qwen' or 'llama'.")

    if tokenizer.pad_token is None:
        special_token_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_token_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_token_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_token_dict["unk_token"] = DEFAULT_UNK_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_token_dict,
        tokenizer=tokenizer,
        model=model,
    )
    model.add_special_tokens(tokenizer)
    if model_args.use_binary_head:
        if model_args.loss_type == 'ls':
            model_args.label_type = 'soft'
        else:
            model_args.label_type = 'hard'
        model.monkey_patch_cls_head(use_hybrid_head=model_args.use_hybrid_head)
        # model_args.modules_to_save = ["lm_head", "embed_tokens"]
    else:
        model_args.label_type = 'hard'
        # model_args.modules_to_save = ["lm_head", "embed_tokens"]
    if not model_args.freeze_header:
        model_args.modules_to_save = ["lm_head", "embed_tokens"]
    model.loss_type = model_args.loss_type

    # 创建原始数据集实例
    train_dataset = CrossNDDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        model_args = model_args,
        data_args = data_args,
        num_turn=model_args.num_turn,
        mode="train",
    )
    
    # 创建验证数据集实例
    eval_dataset = CrossNDDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        model_args = model_args,
        data_args = data_args,
        num_turn=model_args.num_turn,
        mode="eval",  # 使用评估模式
    )
    test_dataset = CrossNDDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        model_args = model_args,
        data_args = data_args,
        num_turn=model_args.num_turn,
        mode="test",
    )

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.target_modules,
        lora_dropout=model_args.lora_dropout,
        task_type=model_args.task_type,
        modules_to_save=model_args.modules_to_save
    )

    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()  # 打印可训练参数信息
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    os.makedirs(training_args.output_dir, exist_ok=True)
    data_collator = CrossNDCollator(tokenizer=tokenizer)
    training_args.remove_unused_columns = False
    # 创建训练器
    trainer = CrossNDTrainer_v2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.tokenizer = tokenizer

    # 开始训练
    # trainer.train()

    # best_checkpoint = trainer.state.best_model_checkpoint
    # logger.info(f"Best checkpoint: {best_checkpoint}")
    trainer._load_from_checkpoint(model_args.lora_path)
    
    trainer.predict(test_dataset=test_dataset)
    
if __name__ == "__main__":
    main()