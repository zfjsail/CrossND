#!/usr/bin/env python
# coding=utf-8
"""
RL模块初始化文件

导出主要的类和函数
"""

from .environment import (
    CrossNDRLEnvironment,
    BatchedRLEnvironment,
    MultiTurnClassificationStep,
    MultiTurnClassificationEpisode,
)

from .reward import (
    CrossNDRewardFunction,
    TurnLevelRewardShaper,
    RewardComputation,
)

from .agent import (
    CrossNDAgent,
    MultiAgentCollector,
    AgentAction,
    AgentStep,
    AgentTrajectory,
)

from .train_rl import (
    GRPOTrainer,
    GRPOConfig,
)

__all__ = [
    # Environment
    'CrossNDRLEnvironment',
    'BatchedRLEnvironment',
    'MultiTurnClassificationStep',
    'MultiTurnClassificationEpisode',
    # Reward
    'CrossNDRewardFunction',
    'TurnLevelRewardShaper',
    'RewardComputation',
    # Agent
    'CrossNDAgent',
    'MultiAgentCollector',
    'AgentAction',
    'AgentStep',
    'AgentTrajectory',
    # Training
    'GRPOTrainer',
    'GRPOConfig',
]
