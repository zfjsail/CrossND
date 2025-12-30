#!/usr/bin/env python
# coding=utf-8
"""
基于GRPO (Group Relative Policy Optimization) 的强化学习训练脚本

参考: https://github.com/SiliangZeng/Multi-Turn-RL-Agent
针对多轮论文异常分配检测任务进行策略优化
"""

import os
import json
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    # 基础超参数
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # GRPO特定超参数
    num_generations: int = 4  # 每个prompt生成多少个samples
    group_size: int = 4  # Group内的样本数
    turn_advantage_coef: float = 0.5  # Turn-level advantage的系数
    use_turn_level_advantage: bool = True  # 是否使用turn-level advantage

    # Reward相关
    outcome_reward_weight: float = 1.0
    turn_level_reward_weight: float = 0.5
    reward_normalization: bool = True

    # 训练相关
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8

    # 保存和日志
    output_dir: str = "./output/rl_training"
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 100

    # 设备设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) 训练器

    核心思想:
    - 对同一个prompt生成多个response
    - 计算每个response的reward
    - 使用group内的reward作为baseline进行relative优化
    - 支持turn-level advantage estimation
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        reward_fn,
        config: GRPOConfig,
    ):
        """
        初始化GRPO训练器

        Args:
            model: 要训练的语言模型
            tokenizer: 分词器
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            reward_fn: Reward函数
            config: GRPO配置
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_fn = reward_fn
        self.config = config

        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        # 学习率调度
        self.lr_scheduler = None

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float('-inf')

        # 用于保存训练过程中的metrics
        self.training_history = {
            'steps': [],
            'losses': [],
            'outcome_rewards': [],
            'turn_rewards': [],
        }

    def train(self):
        """执行完整的训练流程"""
        logger.info("开始GRPO训练...")
        logger.info(f"配置: {self.config.to_dict()}")

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()

            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {epoch_loss:.4f}")

            # 评估
            if (epoch + 1) % max(1, self.config.num_epochs // 5) == 0:
                eval_reward = self._evaluate()
                logger.info(f"Epoch {epoch + 1}, Evaluation Reward: {eval_reward:.4f}")

                # 保存最好的模型
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self._save_checkpoint(is_best=True)

            # 定期保存模型
            if (epoch + 1) % max(1, self.config.num_epochs // 2) == 0:
                self._save_checkpoint(suffix=f"epoch_{epoch + 1}")

        logger.info("训练完成!")
        logger.info(f"最佳评估Reward: {self.best_reward:.4f}")

    def _train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 创建数据加载器
        from torch.utils.data import DataLoader
        from utils import CrossNDCollator

        collator = CrossNDCollator(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        progress_bar = tqdm(dataloader, desc="Training", disable=False)

        for batch_idx, batch in enumerate(progress_bar):
            # 移到device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播获取预测
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )
                logits = outputs.logits

            # 获取label_token的位置和预测
            predictions, confidences = self._extract_predictions(
                logits, batch.get('metadata', [])
            )

            # 计算rewards
            batch_rewards = []
            batch_labels = batch.get('labels', torch.zeros(len(predictions))).cpu().numpy()

            for pred, label in zip(predictions, batch_labels):
                reward_comp = self.reward_fn.compute_combined_reward(
                    pred, label.tolist() if isinstance(label, np.ndarray) else label
                )
                batch_rewards.append(reward_comp)

            # 计算loss (策略梯度)
            loss = self._compute_grpo_loss(
                logits, batch, predictions, batch_rewards
            )

            # 反向传播
            loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # 记录loss
            total_loss += loss.item()
            num_batches += 1

            # 记录metrics
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_outcome_reward = np.mean([r.outcome_reward for r in batch_rewards])

                self.training_history['steps'].append(self.global_step)
                self.training_history['losses'].append(avg_loss)
                self.training_history['outcome_rewards'].append(avg_outcome_reward)

                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'reward': f'{avg_outcome_reward:.4f}',
                })

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _compute_grpo_loss(
        self,
        logits: torch.Tensor,
        batch: Dict[str, Any],
        predictions: List[List[int]],
        batch_rewards: List,
    ) -> torch.Tensor:
        """
        计算GRPO loss

        GRPO的核心思想:
        - 将batch分成多个group (每个group内的samples来自相同的prompt)
        - 在每个group内计算relative rewards (相对于group内的baseline)
        - 使用这些relative rewards来进行策略优化

        Args:
            logits: 模型输出的logits [batch_size, seq_len, vocab_size]
            batch: 批次数据
            predictions: 预测结果列表
            batch_rewards: 计算好的rewards列表

        Returns:
            损失值
        """
        # 这是一个简化的实现，完整的GRPO需要更复杂的逻辑
        # 包括生成多个samples、计算group-relative rewards等

        # 提取label_token的logits
        label_logits, label_positions = self._extract_label_logits(
            logits, batch
        )

        if label_logits is None or len(label_logits) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 计算policy loss (使用outcome rewards)
        loss = 0.0
        for i, (logit, reward) in enumerate(zip(label_logits, batch_rewards)):
            # 计算log probability of predicted action
            log_probs = F.log_softmax(logit, dim=-1)

            # 使用reward作为权重进行策略梯度
            # 注意: 这里需要实现正确的policy gradient计算
            # 简化版本: negative cross entropy weighted by reward
            predicted_label = predictions[i][0] if predictions[i] else 0
            loss += -log_probs[predicted_label] * reward.outcome_reward

        # 平均loss
        loss = loss / max(len(label_logits), 1)

        return loss

    def _extract_predictions(
        self,
        logits: torch.Tensor,
        metadata: List[Dict],
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        从logits中提取预测结果

        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            metadata: 元数据列表

        Returns:
            (predictions, confidences) - 预测结果和置信度
        """
        from utils import LABEL_TOKEN

        predictions = []
        confidences = []

        # 获取label_token的ID
        label_token_id = self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN)

        for batch_idx, meta_list in enumerate(metadata):
            sample_predictions = []
            sample_confidences = []

            # 这是一个简化的实现
            # 实际需要找到label_token的位置并获取其对应的logits
            for meta in (meta_list if isinstance(meta_list, list) else [meta_list]):
                # 二分类: 0 (异常) 或 1 (正常)
                # 简化: 使用logits的最大值位置
                pred = int(torch.argmax(logits[batch_idx, -1, :2]).item()) % 2
                conf = float(torch.max(F.softmax(logits[batch_idx, -1, :2], dim=-1)).item())

                sample_predictions.append(pred)
                sample_confidences.append(conf)

            predictions.append(sample_predictions)
            confidences.append(sample_confidences)

        return predictions, confidences

    def _extract_label_logits(
        self,
        logits: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        """
        提取label_token对应的logits

        Args:
            logits: 模型输出logits [batch_size, seq_len, vocab_size]
            batch: 批次数据

        Returns:
            (label_logits, positions)
        """
        from utils import LABEL_TOKEN

        label_token_id = self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN)

        label_logits = []
        label_positions = []

        input_ids = batch['input_ids']
        for batch_idx, input_id_seq in enumerate(input_ids):
            # 找到label_token的位置
            positions = (input_id_seq == label_token_id).nonzero(as_tuple=True)[0]

            if len(positions) > 0:
                # 取第一个label_token的logits
                pos = positions[0].item()
                label_logits.append(logits[batch_idx, pos, :2])  # 只取前2个token的logits (0和1)
                label_positions.append(pos)

        if not label_logits:
            return None, None

        return torch.stack(label_logits), label_positions

    def _evaluate(self) -> float:
        """评估模型"""
        self.model.eval()

        from torch.utils.data import DataLoader
        from utils import CrossNDCollator

        collator = CrossNDCollator(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        total_reward = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )

                predictions, _ = self._extract_predictions(
                    outputs.logits, batch.get('metadata', [])
                )
                batch_labels = batch.get('labels', torch.zeros(len(predictions)))

                for pred, label in zip(predictions, batch_labels):
                    label_list = label.cpu().numpy().tolist() if isinstance(label, torch.Tensor) else label
                    reward_comp = self.reward_fn.compute_combined_reward(
                        pred, label_list
                    )
                    total_reward += reward_comp.outcome_reward
                    num_samples += 1

        avg_reward = total_reward / max(num_samples, 1)
        return avg_reward

    def _save_checkpoint(self, is_best: bool = False, suffix: str = ""):
        """保存模型checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint{'-' + suffix if suffix else ''}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        if is_best:
            best_model_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_model_path)

        logger.info(f"模型保存到 {checkpoint_dir}")

    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"训练历史保存到 {history_path}")


def main():
    """主函数 - RL训练入口"""
    from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
    from train import DataArguments, ModelArguments
    from utils import CrossNDDataset, special_token_dict
    from model import Qwen3ForCrossND, LlamaForCrossND
    from reward import CrossNDRewardFunction
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    # 解析参数 (可以复用train.py中的参数)
    parser = HfArgumentParser((DataArguments, ModelArguments, GRPOConfig))
    data_args, model_args, grpo_config = parser.parse_args_into_dataclasses()

    # 加载模型和tokenizer
    config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

    # 添加特殊token
    tokenizer.add_special_tokens({'additional_special_tokens': special_token_dict['additional_special_tokens']})

    # 加载模型
    if "qwen" in model_args.model_path.lower():
        model = Qwen3ForCrossND.from_pretrained(
            model_args.model_path,
            config=config,
            trust_remote_code=True,
        ).to(grpo_config.device)
    else:
        model = LlamaForCrossND.from_pretrained(
            model_args.model_path,
            config=config,
            trust_remote_code=True,
        ).to(grpo_config.device)

    # 加载数据集
    train_dataset = CrossNDDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        model_args=model_args,
        data_args=data_args,
        mode="train",
    )

    eval_dataset = CrossNDDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        model_args=model_args,
        data_args=data_args,
        mode="eval",
    )

    # 创建Reward函数
    reward_fn = CrossNDRewardFunction(
        reward_type="accuracy",
        outcome_weight=grpo_config.outcome_reward_weight,
        turn_weight=grpo_config.turn_level_reward_weight,
    )

    # 创建训练器
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_fn=reward_fn,
        config=grpo_config,
    )

    # 开始训练
    trainer.train()
    trainer.save_training_history()


if __name__ == "__main__":
    main()
