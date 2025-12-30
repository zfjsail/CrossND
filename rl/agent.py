#!/usr/bin/env python
# coding=utf-8
"""
多轮论文异常分配检测的RL Agent实现

Agent负责:
1. 接收环境观察
2. 生成多轮预测
3. 与环境交互
4. 基于reward进行学习
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Agent的单个动作"""
    label: int  # 预测标签 (0或1)
    confidence: float  # 预测置信度
    logits: torch.Tensor  # 原始logits
    log_prob: torch.Tensor  # log probability


@dataclass
class AgentStep:
    """Agent执行的单步"""
    turn: int  # 当前轮次
    observation: Dict[str, Any]  # 环境观察
    action: AgentAction  # 采取的行动
    reward: float = 0.0  # 获得的reward (后续设置)


@dataclass
class AgentTrajectory:
    """Agent的完整轨迹 (episode)"""
    episode_id: int
    steps: List[AgentStep]  # 所有步骤
    total_reward: float = 0.0  # 总reward
    episode_reward: float = 0.0  # episode reward
    turn_rewards: List[float] = None  # 每一轮的reward


class CrossNDAgent:
    """
    多轮论文异常分配检测的RL Agent

    Agent架构:
    - Policy Head: 生成动作的概率分布
    - Value Head: 估计状态价值 (可选)
    - 支持多轮交互和轨迹记录
    """

    def __init__(
        self,
        model,
        tokenizer,
        num_turns: int = 10,
        temperature: float = 1.0,
        use_sampling: bool = False,
    ):
        """
        初始化Agent

        Args:
            model: 语言模型
            tokenizer: 分词器
            num_turns: 最大轮数
            temperature: 采样温度 (仅当use_sampling=True时使用)
            use_sampling: 是否使用采样而不是贪心选择
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_turns = num_turns
        self.temperature = temperature
        self.use_sampling = use_sampling

        # Agent状态
        self.current_episode_id = 0
        self.current_turn = 0
        self.trajectory = None

        # 用于记录内部状态
        self.hidden_states_cache = {}
        self.attention_weights_cache = {}

    def initialize_episode(self, episode_id: int) -> Dict[str, Any]:
        """
        初始化一个新的episode

        Args:
            episode_id: episode ID

        Returns:
            初始观察
        """
        self.current_episode_id = episode_id
        self.current_turn = 0
        self.trajectory = AgentTrajectory(
            episode_id=episode_id,
            steps=[],
            turn_rewards=[],
        )

        observation = {
            'episode_id': episode_id,
            'turn': 0,
            'num_turns': self.num_turns,
        }

        return observation

    def predict_action(
        self,
        observation: Dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_token_positions: List[int] = None,
    ) -> AgentAction:
        """
        根据观察预测动作

        Args:
            observation: 环境观察
            input_ids: 输入token ID [seq_len]
            attention_mask: 注意力掩码 [seq_len]
            label_token_positions: label_token的位置列表

        Returns:
            AgentAction
        """
        with torch.no_grad():
            # 前向传播
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                output_hidden_states=True,
                output_attentions=False,
            )

            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # 提取label位置的logits
        if label_token_positions is not None and len(label_token_positions) > self.current_turn:
            pos = label_token_positions[self.current_turn]
        else:
            pos = -1  # 使用最后一个token

        label_logits = logits[pos, :2]  # 只考虑token 0和1 (对应异常/正常)

        # 计算概率
        probs = F.softmax(label_logits, dim=-1)

        # 采取动作
        if self.use_sampling:
            # 采样动作
            scaled_probs = probs ** (1.0 / self.temperature)
            scaled_probs = scaled_probs / scaled_probs.sum()
            label = torch.multinomial(scaled_probs, num_samples=1).item()
        else:
            # 贪心选择最可能的动作
            label = int(torch.argmax(probs).item())

        confidence = float(probs[label].item())
        log_prob = torch.log(probs[label])

        action = AgentAction(
            label=label,
            confidence=confidence,
            logits=label_logits,
            log_prob=log_prob,
        )

        return action

    def step(
        self,
        observation: Dict[str, Any],
        action: AgentAction,
        reward: float = 0.0,
    ) -> AgentStep:
        """
        执行一步，记录轨迹

        Args:
            observation: 环境观察
            action: 采取的动作
            reward: 获得的reward

        Returns:
            AgentStep
        """
        self.current_turn += 1

        agent_step = AgentStep(
            turn=self.current_turn,
            observation=observation,
            action=action,
            reward=reward,
        )

        self.trajectory.steps.append(agent_step)
        if self.trajectory.turn_rewards is not None:
            self.trajectory.turn_rewards.append(reward)

        return agent_step

    def generate_trajectory(
        self,
        episode_id: int,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        label_token_positions_list: List[List[int]] = None,
    ) -> AgentTrajectory:
        """
        生成完整的episode轨迹

        Args:
            episode_id: episode ID
            input_ids_list: 输入token ID列表
            attention_mask_list: 注意力掩码列表
            label_token_positions_list: label_token位置列表

        Returns:
            AgentTrajectory
        """
        self.initialize_episode(episode_id)

        predictions = []
        confidences = []

        for turn, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            # 生成观察
            observation = {
                'episode_id': episode_id,
                'turn': turn,
                'num_turns': len(input_ids_list),
            }

            # 获取label_token位置
            label_positions = None
            if label_token_positions_list is not None:
                label_positions = label_token_positions_list[turn]

            # 预测动作
            action = self.predict_action(
                observation,
                input_ids,
                attention_mask,
                label_positions,
            )

            predictions.append(action.label)
            confidences.append(action.confidence)

            # 记录步骤
            self.step(observation, action)

        self.trajectory.predictions = predictions
        self.trajectory.confidences = confidences

        return self.trajectory

    def compute_trajectory_logprobs(self) -> torch.Tensor:
        """
        计算轨迹中所有动作的log probabilities

        Returns:
            log prob的总和 (用于策略梯度)
        """
        if self.trajectory is None or len(self.trajectory.steps) == 0:
            return torch.tensor(0.0)

        log_probs = torch.stack([step.action.log_prob for step in self.trajectory.steps])
        return log_probs.sum()

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        获取轨迹的摘要信息

        Returns:
            摘要字典
        """
        if self.trajectory is None:
            return {}

        return {
            'episode_id': self.trajectory.episode_id,
            'num_steps': len(self.trajectory.steps),
            'predictions': self.trajectory.predictions,
            'confidences': self.trajectory.confidences,
            'total_reward': self.trajectory.total_reward,
            'episode_reward': self.trajectory.episode_reward,
            'turn_rewards': self.trajectory.turn_rewards,
        }


class MultiAgentCollector:
    """
    多Agent轨迹收集器

    用于并行收集多个agents的轨迹，支持批处理
    """

    def __init__(self, num_agents: int = 4):
        """
        初始化收集器

        Args:
            num_agents: agent数量
        """
        self.num_agents = num_agents
        self.agents: List[CrossNDAgent] = []
        self.trajectories: List[AgentTrajectory] = []

    def add_agent(self, agent: CrossNDAgent):
        """添加一个agent"""
        self.agents.append(agent)

    def collect_trajectories_batch(
        self,
        episode_ids: List[int],
        batch_input_ids: List[List[torch.Tensor]],
        batch_attention_masks: List[List[torch.Tensor]],
    ) -> List[AgentTrajectory]:
        """
        批量收集轨迹

        Args:
            episode_ids: episode ID列表
            batch_input_ids: 输入token列表
            batch_attention_masks: 注意力掩码列表

        Returns:
            轨迹列表
        """
        trajectories = []

        for agent, episode_id, input_ids_list, attention_mask_list in zip(
            self.agents,
            episode_ids,
            batch_input_ids,
            batch_attention_masks,
        ):
            trajectory = agent.generate_trajectory(
                episode_id,
                input_ids_list,
                attention_mask_list,
            )
            trajectories.append(trajectory)

        self.trajectories = trajectories
        return trajectories

    def get_group_rewards(self) -> Dict[int, Dict[str, float]]:
        """
        获取group内的reward信息 (用于GRPO)

        Returns:
            {episode_id: {'rewards': [...], 'mean': float, 'std': float}}
        """
        group_rewards = {}

        for trajectory in self.trajectories:
            rewards = []
            for step in trajectory.steps:
                rewards.append(step.reward)

            if trajectory.episode_id not in group_rewards:
                group_rewards[trajectory.episode_id] = {
                    'rewards': rewards,
                    'mean': np.mean(rewards) if rewards else 0.0,
                    'std': np.std(rewards) if rewards else 0.0,
                }

        return group_rewards


if __name__ == '__main__':
    # 简单的agent测试
    logger.info("CrossNDAgent 测试模块")

    # Agent基本功能测试
    print("Agent轨迹数据结构定义完成")
    print(f"- AgentAction: 记录单个动作信息")
    print(f"- AgentStep: 记录单步执行信息")
    print(f"- AgentTrajectory: 记录完整episode轨迹")
