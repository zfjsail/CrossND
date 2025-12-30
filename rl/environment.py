#!/usr/bin/env python
# coding=utf-8
"""
多轮论文异常分配检测的RL环境实现
基于CrossND数据集，实现多轮分类任务的RL环境接口
"""

from typing import List, Dict, Tuple, Any, Optional
import torch
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiTurnClassificationStep:
    """多轮分类任务的单步环节"""
    paper_idx: int  # 当前处理的论文索引 (在num_turn中的位置)
    prediction: int  # 模型预测结果 (0/1)
    ground_truth: int  # 真实标签
    confidence: float  # 预测置信度
    turn_number: int  # 当前轮次 (1-based)
    total_turns: int  # 总轮数


@dataclass
class MultiTurnClassificationEpisode:
    """完整的多轮分类任务episode"""
    episode_id: int
    author_id: str
    paper_ids: List[str]
    predictions: List[int]  # 模型的预测结果列表
    ground_truths: List[int]  # 真实标签列表
    confidences: List[float]  # 模型置信度列表
    steps: List[MultiTurnClassificationStep]  # 详细的每步信息


class CrossNDRLEnvironment:
    """
    多轮论文异常分配检测的RL环境

    环境特性:
    - 每个episode包含num_turn个分类任务
    - 每个任务都有一个二分类标签 (0: 异常论文, 1: 正常论文)
    - 我们关心的是整个episode的分类准确率

    Attributes:
        num_turn: 单个episode中的轮数
        vocab_size: 词汇大小
        model: 用于推理的模型
        tokenizer: 分词器
        dataset: 数据集
    """

    def __init__(
        self,
        num_turn: int = 10,
        vocab_size: int = 151936,
        model = None,
        tokenizer = None,
        dataset = None,
    ):
        """
        初始化RL环境

        Args:
            num_turn: 单个episode中的轮数
            vocab_size: 词汇大小
            model: 用于推理的模型
            tokenizer: 分词器
            dataset: 数据集
        """
        self.num_turn = num_turn
        self.vocab_size = vocab_size
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

        # 环境状态
        self.current_episode_id = 0
        self.current_step = 0
        self.episode_done = False

    def reset(self, episode_id: int = None) -> Dict[str, Any]:
        """
        重置环境，开始新的episode

        Args:
            episode_id: 要加载的episode ID

        Returns:
            初始观察信息字典
        """
        if episode_id is None:
            episode_id = self.current_episode_id

        self.current_episode_id = episode_id
        self.current_step = 0
        self.episode_done = False

        # 从数据集中获取episode数据
        if self.dataset is not None and episode_id < len(self.dataset):
            sample = self.dataset[episode_id]

            observation = {
                'input_ids': sample['input_ids'],
                'attention_mask': sample['attention_mask'],
                'metadata': sample.get('metadata', []),
                'labels': sample.get('labels', None),
                'episode_id': episode_id,
                'num_turns': min(self.num_turn, len(sample.get('metadata', [])))
            }
        else:
            observation = {
                'episode_id': episode_id,
                'num_turns': self.num_turn,
                'input_ids': None,
                'attention_mask': None,
                'metadata': [],
                'labels': None
            }

        return observation

    def step(
        self,
        predictions: torch.Tensor,
        logits: torch.Tensor = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        执行一步环境交互

        Args:
            predictions: 模型预测结果 [batch_size, num_turns]
            logits: 模型logits输出，用于计算置信度

        Returns:
            (observation, done) 观察和episode是否结束
        """
        self.current_step += 1

        # 判断episode是否完成
        if self.current_step >= self.num_turn:
            self.episode_done = True

        observation = {
            'episode_id': self.current_episode_id,
            'step': self.current_step,
            'predictions': predictions,
            'logits': logits,
        }

        return observation, self.episode_done

    def get_episode_data(self, episode_id: int) -> Optional[MultiTurnClassificationEpisode]:
        """
        获取特定episode的完整数据

        Args:
            episode_id: episode ID

        Returns:
            MultiTurnClassificationEpisode 对象或 None
        """
        if self.dataset is None or episode_id >= len(self.dataset):
            return None

        sample = self.dataset[episode_id]
        metadata = sample.get('metadata', [])
        labels = sample.get('labels', None)

        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            if labels.dim() == 2:
                labels = labels.squeeze(0)
            labels = labels.tolist()
        else:
            labels = [0] * len(metadata)

        # 提取author_id 和 paper_ids
        author_id = str(metadata[0].get('aid1', 'unknown')) if metadata else 'unknown'
        paper_ids = [meta.get('pid', f'paper_{i}') for i, meta in enumerate(metadata)]

        return MultiTurnClassificationEpisode(
            episode_id=episode_id,
            author_id=author_id,
            paper_ids=paper_ids,
            predictions=[0] * len(metadata),  # 初始化，在推理后更新
            ground_truths=labels,
            confidences=[0.0] * len(metadata),
            steps=[]
        )

    def compute_episode_accuracy(
        self,
        predictions: List[int],
        ground_truths: List[int]
    ) -> float:
        """
        计算单个episode的准确率

        Args:
            predictions: 模型预测结果列表
            ground_truths: 真实标签列表

        Returns:
            准确率 (0.0 - 1.0)
        """
        if len(predictions) != len(ground_truths):
            logger.warning(
                f"Predictions and ground truths length mismatch: "
                f"{len(predictions)} vs {len(ground_truths)}"
            )
            min_len = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_len]
            ground_truths = ground_truths[:min_len]

        if len(predictions) == 0:
            return 0.0

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        return float(correct) / len(predictions)

    def compute_per_turn_accuracy(
        self,
        predictions: List[int],
        ground_truths: List[int]
    ) -> List[float]:
        """
        计算每一轮的准确率 (累积)

        Args:
            predictions: 模型预测结果列表
            ground_truths: 真实标签列表

        Returns:
            每一轮的累积准确率列表
        """
        per_turn_acc = []
        for i in range(1, len(predictions) + 1):
            acc = self.compute_episode_accuracy(
                predictions[:i],
                ground_truths[:i]
            )
            per_turn_acc.append(acc)
        return per_turn_acc


class BatchedRLEnvironment:
    """
    批量处理多个episodes的RL环境包装器

    支持批量操作，提高效率
    """

    def __init__(
        self,
        base_env: CrossNDRLEnvironment,
        batch_size: int = 1,
    ):
        """
        初始化批量环境

        Args:
            base_env: 基础RL环境
            batch_size: 批大小
        """
        self.base_env = base_env
        self.batch_size = batch_size
        self.current_batch_episodes = []

    def reset_batch(self, episode_ids: List[int]) -> Dict[str, Any]:
        """
        重置一个批次的episodes

        Args:
            episode_ids: 要加载的episode ID列表

        Returns:
            批量观察信息字典
        """
        self.current_batch_episodes = episode_ids

        batch_observations = {
            'input_ids_list': [],
            'attention_mask_list': [],
            'metadata_list': [],
            'labels_list': [],
            'episode_ids': episode_ids,
        }

        for episode_id in episode_ids:
            obs = self.base_env.reset(episode_id)
            batch_observations['input_ids_list'].append(obs['input_ids'])
            batch_observations['attention_mask_list'].append(obs['attention_mask'])
            batch_observations['metadata_list'].append(obs.get('metadata', []))
            batch_observations['labels_list'].append(obs.get('labels', None))

        return batch_observations

    def step_batch(
        self,
        batch_predictions: List[torch.Tensor],
        batch_logits: List[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], List[bool]]:
        """
        批量执行一步

        Args:
            batch_predictions: 预测结果列表
            batch_logits: logits列表

        Returns:
            (batch_observations, dones)
        """
        batch_observations = {
            'predictions_list': batch_predictions,
            'logits_list': batch_logits if batch_logits is not None else [None] * len(batch_predictions),
            'episode_ids': self.current_batch_episodes,
        }

        dones = [self.base_env.episode_done for _ in range(len(self.current_batch_episodes))]

        return batch_observations, dones


if __name__ == '__main__':
    # 简单的环境测试
    env = CrossNDRLEnvironment(num_turn=10)

    # 测试准确率计算
    predictions = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    ground_truths = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

    acc = env.compute_episode_accuracy(predictions, ground_truths)
    per_turn_acc = env.compute_per_turn_accuracy(predictions, ground_truths)

    print(f"Episode Accuracy: {acc:.4f}")
    print(f"Per-turn Accuracy: {[f'{a:.4f}' for a in per_turn_acc]}")
