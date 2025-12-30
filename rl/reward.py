#!/usr/bin/env python
# coding=utf-8
"""
RL Reward函数实现
支持多种reward计算方式：
- 基于准确率的reward (turn-level 和 outcome-level)
- 组合reward
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardComputation:
    """Reward计算结果"""
    outcome_reward: float  # 最终结果级别的reward
    turn_level_rewards: List[float]  # 每一轮的turn-level reward
    turn_level_advantages: List[float]  # 每一轮相对于baseline的advantage
    combined_reward: float  # 组合后的总reward


class CrossNDRewardFunction:
    """
    论文异常分配检测任务的Reward函数

    设计原理：
    - Outcome Reward: 基于整个episode的最终分类准确率
    - Turn-level Reward: 基于每一轮累积准确率的改进
    - Advantage: 相对于baseline的优势估计

    这种设计鼓励模型在多轮任务中逐步改进其预测准确率。
    """

    def __init__(
        self,
        reward_type: str = "accuracy",  # "accuracy", "f1", "auc" 等
        outcome_weight: float = 1.0,  # outcome reward的权重
        turn_weight: float = 0.5,  # turn-level reward的权重
        use_baseline: bool = True,  # 是否使用baseline来计算advantage
        baseline_strategy: str = "moving_average",  # "moving_average" 或 "previous_turn"
    ):
        """
        初始化Reward函数

        Args:
            reward_type: 奖励类型
            outcome_weight: outcome reward权重
            turn_weight: turn-level reward权重
            use_baseline: 是否使用baseline
            baseline_strategy: baseline策略
        """
        self.reward_type = reward_type
        self.outcome_weight = outcome_weight
        self.turn_weight = turn_weight
        self.use_baseline = use_baseline
        self.baseline_strategy = baseline_strategy

        # 用于计算moving average baseline
        self.baseline_history = []
        self.baseline_ema = 0.0
        self.baseline_ema_decay = 0.99

    def compute_outcome_reward(
        self,
        predictions: List[int],
        ground_truths: List[int],
        reward_type: str = None,
    ) -> float:
        """
        计算outcome-level reward (基于最终准确率)

        Args:
            predictions: 模型预测结果列表
            ground_truths: 真实标签列表
            reward_type: 奖励类型

        Returns:
            outcome reward值 (通常在 -1.0 到 1.0 之间)
        """
        if reward_type is None:
            reward_type = self.reward_type

        if len(predictions) != len(ground_truths) or len(predictions) == 0:
            return 0.0

        # 计算准确率
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        if reward_type == "accuracy":
            # 基于准确率的reward: acc * 2 - 1 (映射到 [-1, 1])
            acc = np.mean(predictions == ground_truths)
            reward = acc * 2 - 1  # [0, 1] -> [-1, 1]

        elif reward_type == "f1":
            # 基于F1分数的reward
            from sklearn.metrics import f1_score
            try:
                f1 = f1_score(ground_truths, predictions, average='binary')
                reward = f1 * 2 - 1  # [0, 1] -> [-1, 1]
            except:
                reward = 0.0

        elif reward_type == "balanced_acc":
            # 平衡准确率
            from sklearn.metrics import balanced_accuracy_score
            try:
                bacc = balanced_accuracy_score(ground_truths, predictions)
                reward = bacc * 2 - 1
            except:
                reward = 0.0

        else:
            # 默认使用准确率
            acc = np.mean(predictions == ground_truths)
            reward = acc * 2 - 1

        return float(reward)

    def compute_turn_level_rewards(
        self,
        predictions: List[int],
        ground_truths: List[int],
        reward_type: str = None,
    ) -> List[float]:
        """
        计算turn-level reward (基于每一轮的累积准确率改进)

        Args:
            predictions: 模型预测结果列表
            ground_truths: 真实标签列表
            reward_type: 奖励类型

        Returns:
            每一轮的turn-level reward列表
        """
        if reward_type is None:
            reward_type = self.reward_type

        if len(predictions) == 0:
            return []

        turn_rewards = []
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        # 计算每一轮的累积准确率
        prev_acc = 0.0
        for i in range(1, len(predictions) + 1):
            current_preds = predictions[:i]
            current_gts = ground_truths[:i]

            if reward_type == "accuracy":
                current_acc = np.mean(current_preds == current_gts)
            elif reward_type == "f1":
                try:
                    from sklearn.metrics import f1_score
                    current_acc = f1_score(current_gts, current_preds, average='binary')
                except:
                    current_acc = 0.0
            else:
                current_acc = np.mean(current_preds == current_gts)

            # Turn reward: 累积改进
            turn_reward = current_acc - prev_acc  # 这一轮的改进

            turn_rewards.append(turn_reward)
            prev_acc = current_acc

        return turn_rewards

    def compute_advantages(
        self,
        turn_rewards: List[float],
        baseline_values: List[float] = None,
    ) -> List[float]:
        """
        计算advantage (优势估计)

        Args:
            turn_rewards: turn-level reward列表
            baseline_values: baseline值列表

        Returns:
            advantage列表
        """
        if not turn_rewards:
            return []

        if baseline_values is None:
            if self.baseline_strategy == "moving_average":
                # 使用moving average作为baseline
                baseline_values = self._compute_moving_average_baseline(turn_rewards)
            elif self.baseline_strategy == "previous_turn":
                # 使用前一轮的值作为baseline
                baseline_values = [0.0] + turn_rewards[:-1]
            else:
                baseline_values = [0.0] * len(turn_rewards)

        advantages = [
            reward - baseline
            for reward, baseline in zip(turn_rewards, baseline_values)
        ]

        return advantages

    def _compute_moving_average_baseline(self, rewards: List[float]) -> List[float]:
        """
        计算moving average baseline

        Args:
            rewards: reward列表

        Returns:
            baseline值列表
        """
        baseline_values = []

        for reward in rewards:
            # 更新EMA baseline
            if len(self.baseline_history) == 0:
                self.baseline_ema = reward
            else:
                self.baseline_ema = (
                    self.baseline_ema_decay * self.baseline_ema +
                    (1 - self.baseline_ema_decay) * reward
                )

            baseline_values.append(self.baseline_ema)
            self.baseline_history.append(reward)

        return baseline_values

    def compute_combined_reward(
        self,
        predictions: List[int],
        ground_truths: List[int],
    ) -> RewardComputation:
        """
        计算综合reward (组合outcome和turn-level)

        Args:
            predictions: 模型预测结果列表
            ground_truths: 真实标签列表

        Returns:
            RewardComputation对象，包含各种reward值
        """
        # 计算outcome reward
        outcome_reward = self.compute_outcome_reward(predictions, ground_truths)

        # 计算turn-level rewards
        turn_rewards = self.compute_turn_level_rewards(predictions, ground_truths)

        # 计算advantages (如果使用)
        if self.use_baseline:
            advantages = self.compute_advantages(turn_rewards)
        else:
            advantages = turn_rewards

        # 组合reward: outcome * outcome_weight + avg(turn_rewards) * turn_weight
        if turn_rewards:
            avg_turn_reward = float(np.mean(turn_rewards))
            combined_reward = (
                self.outcome_weight * outcome_reward +
                self.turn_weight * avg_turn_reward
            )
        else:
            combined_reward = self.outcome_weight * outcome_reward

        return RewardComputation(
            outcome_reward=outcome_reward,
            turn_level_rewards=turn_rewards,
            turn_level_advantages=advantages,
            combined_reward=combined_reward,
        )

    def compute_batch_rewards(
        self,
        batch_predictions: List[List[int]],
        batch_ground_truths: List[List[int]],
    ) -> List[RewardComputation]:
        """
        批量计算rewards

        Args:
            batch_predictions: 预测结果列表 [batch_size, num_turns]
            batch_ground_truths: 真实标签列表 [batch_size, num_turns]

        Returns:
            RewardComputation列表
        """
        batch_rewards = []
        for predictions, ground_truths in zip(batch_predictions, batch_ground_truths):
            reward_comp = self.compute_combined_reward(predictions, ground_truths)
            batch_rewards.append(reward_comp)

        return batch_rewards

    def reset_baseline(self):
        """重置baseline历史"""
        self.baseline_history = []
        self.baseline_ema = 0.0


class TurnLevelRewardShaper:
    """
    Turn-level reward shaping，用于改进RL训练的stability和convergence

    支持多种shaping策略：
    - discount: 对未来reward进行discount
    - normalization: 对reward进行normalization
    - clipping: 对reward进行clipping
    """

    def __init__(
        self,
        gamma: float = 0.99,  # discount factor
        normalize: bool = True,  # 是否进行normalization
        clip_range: Tuple[float, float] = (-1.0, 1.0),  # clipping范围
    ):
        """
        初始化reward shaper

        Args:
            gamma: discount factor
            normalize: 是否进行normalization
            clip_range: clipping范围
        """
        self.gamma = gamma
        self.normalize = normalize
        self.clip_range = clip_range

        # 用于normalization的统计量
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []

    def shape_rewards(
        self,
        rewards: List[float],
        apply_discount: bool = True,
        apply_normalize: bool = True,
        apply_clip: bool = True,
    ) -> List[float]:
        """
        对rewards进行shaping

        Args:
            rewards: 原始reward列表
            apply_discount: 是否应用discount
            apply_normalize: 是否应用normalization
            apply_clip: 是否应用clipping

        Returns:
            处理后的reward列表
        """
        shaped_rewards = list(rewards)

        # 应用discount
        if apply_discount:
            shaped_rewards = self._apply_discount(shaped_rewards)

        # 记录用于normalization
        self.reward_history.extend(shaped_rewards)

        # 应用normalization
        if apply_normalize and self.normalize:
            shaped_rewards = self._apply_normalize(shaped_rewards)

        # 应用clipping
        if apply_clip:
            shaped_rewards = self._apply_clip(shaped_rewards)

        return shaped_rewards

    def _apply_discount(self, rewards: List[float]) -> List[float]:
        """应用discount"""
        discounted = []
        cumulative = 0.0
        for reward in reversed(rewards):
            cumulative = reward + self.gamma * cumulative
            discounted.insert(0, cumulative)
        return discounted

    def _apply_normalize(self, rewards: List[float]) -> List[float]:
        """应用normalization"""
        if len(self.reward_history) > 0:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-8

        normalized = [
            (r - self.reward_mean) / self.reward_std
            for r in rewards
        ]
        return normalized

    def _apply_clip(self, rewards: List[float]) -> List[float]:
        """应用clipping"""
        clipped = [
            np.clip(r, self.clip_range[0], self.clip_range[1])
            for r in rewards
        ]
        return clipped


if __name__ == '__main__':
    # 测试reward函数
    reward_fn = CrossNDRewardFunction(
        reward_type="accuracy",
        outcome_weight=1.0,
        turn_weight=0.5,
    )

    # 示例预测和标签
    predictions = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    ground_truths = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

    # 计算rewards
    reward_comp = reward_fn.compute_combined_reward(predictions, ground_truths)

    print(f"Outcome Reward: {reward_comp.outcome_reward:.4f}")
    print(f"Turn-level Rewards: {[f'{r:.4f}' for r in reward_comp.turn_level_rewards]}")
    print(f"Turn-level Advantages: {[f'{a:.4f}' for a in reward_comp.turn_level_advantages]}")
    print(f"Combined Reward: {reward_comp.combined_reward:.4f}")

    # 测试reward shaping
    shaper = TurnLevelRewardShaper(gamma=0.99, normalize=True)
    shaped_rewards = shaper.shape_rewards(reward_comp.turn_level_rewards)
    print(f"Shaped Rewards: {[f'{r:.4f}' for r in shaped_rewards]}")
