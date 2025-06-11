from datasets import load_dataset
from transformers import Qwen3ForTokenClassification, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model
import logging
import os
import random
from transformers import TrainingArguments, Trainer, HfArgumentParser
from utils import *
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
# from trainer import ResampleCallback

@dataclass
class ModelArguments:
    """
    模型相关的参数
    """
    model_path: str = field(
        default="/home/zhipuai/zhangfanjin-15T/pyh/models/models/Qwen/Qwen3-4B",
        metadata={"help": "预训练模型的路径"}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "分类任务标签数量"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )

@dataclass
class DataArguments:
    """
    数据集相关的参数
    """
    data_dir: str = field(
        default="/home/zhipuai/zhangfanjin-15T/pyh/pangyunhe1/git/crossnd-202211/data/kddcup",
        metadata={"help": "数据目录"}
    )
    num_turn: int = field(
        default=1,
        metadata={"help": "对话轮数"}
    )
    apply_chat_template: bool = field(
        default=True,
        metadata={"help": "是否应用聊天模板"}
    )
    use_outer: bool = field(
        default=True,
        metadata={"help": "是否使用外部数据"}
    )

@dataclass
class LoraArguments:
    """
    LoRA相关的参数
    """
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout率"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "需要应用LoRA的模块"}
    )
    task_type: str = field(
        default="SEQUENCE_CLASSIFICATION",
        metadata={"help": "任务类型"}
    )
    modules_to_save: List[str] = field(
        default_factory=lambda: ["score", "embed_tokens"],
        metadata={"help": "需要保存的模块"}
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.
    From https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        # for qwen3fortokenclassification model , got no output_embeddings
        # output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # output_embeddings[-num_new_tokens:] = output_embeddings_avg

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    """
    计算评估指标

    Args:
        eval_pred: 包含predictions和label_ids的元组
        
    Returns:
        包含评估指标的字典
    """
    logits, labels = eval_pred
    
    # 对于token classification，logits形状为 [batch_size, seq_length, num_labels]
    # 我们需要找到每个token位置的预测
    predictions = np.argmax(logits, axis=-1)  # 二分类，使用argmax
    
    # 由于标签中有-100（被忽略的标签），我们只计算有效标签的准确率
    valid_mask = labels != -100
    
    # 计算有效标签的准确率
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 计算基本的准确率
    accuracy = (valid_predictions == valid_labels).mean()
    
    # 计算AUC分数
    # 获取有效位置的logits以计算概率
    valid_logits = logits[valid_mask]
    
    # 二分类情况（使用softmax）
    probs = np.exp(valid_logits[:, 1]) / np.exp(valid_logits).sum(axis=-1)
    auc_score = roc_auc_score(valid_labels, probs)
    
    # 收集结果
    result = {
        "accuracy": accuracy,
        "auc": auc_score
    }

    return result

def main():
    # 使用HfArgumentParser处理命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    logger.info("正在加载数据集...")
    
    # 加载模型和分词器
    logger.info(f"正在加载模型和分词器: {model_args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=model_args.trust_remote_code)
    
    # 设置特殊token
    if tokenizer.pad_token is None:
        special_token_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_token_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_token_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_token_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    # 创建原始数据集实例
    train_dataset = CrossNDDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        num_turn=data_args.num_turn,
        mode="train",
        apply_chat_template=data_args.apply_chat_template,
        use_outer=data_args.use_outer
    )
    
    # 创建验证数据集实例
    logger.info("创建验证数据集...")
    eval_dataset = CrossNDDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        num_turn=data_args.num_turn,
        mode="eval",  # 使用评估模式
        apply_chat_template=data_args.apply_chat_template,
        use_outer=data_args.use_outer
    )

    # 获取标签数量
    logger.info(f"分类任务标签数量: {model_args.num_labels}")
    
    # 加载基础模型
    logger.info("加载基础模型...")
    base_model = Qwen3ForTokenClassification.from_pretrained(
        model_args.model_path,
        num_labels=model_args.num_labels,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # 调整tokenizer和模型的embedding大小以添加特殊token
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_token_dict,
        tokenizer=tokenizer,
        model=base_model,
    )
    
    # 确保模型的pad_token_id与tokenizer一致
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # 配置LoRA
    logger.info("配置LoRA参数...")
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        task_type=lora_args.task_type,
        modules_to_save=lora_args.modules_to_save
    )
    
    # 将模型转换为PEFT模型
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数信息
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 使用自定义的数据整理器
    data_collator = CrossNDCollator(tokenizer=tokenizer)
    
    # 创建训练器
    logger.info("创建Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # 添加评估指标计算函数
    )
    
    # 将处理函数和分词器添加到trainer，以便回调使用
    trainer.tokenizer = tokenizer

    # 开始训练
    logger.info("开始训练...")

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main() 