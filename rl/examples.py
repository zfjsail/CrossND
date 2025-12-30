#!/usr/bin/env python
# coding=utf-8
"""
强化学习模块使用说明和快速开始指南

本模块实现了基于GRPO算法的强化学习框架，用于多轮论文异常分配检测任务。

主要特性：
1. 多轮分类任务环境 (Environment)
2. 基于多轮准确率的Reward函数
3. RL Agent的实现
4. GRPO训练算法
"""

import torch
import json
import sys
import os
from typing import Dict, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment import CrossNDRLEnvironment
from rl.reward import CrossNDRewardFunction, TurnLevelRewardShaper
from rl.agent import CrossNDAgent, MultiAgentCollector
from rl.train_rl import GRPOConfig, GRPOTrainer


def example_1_environment():
    """
    示例1: RL环境的基本使用

    演示如何：
    1. 创建RL环境
    2. 重置环境
    3. 执行环境步骤
    4. 计算准确率
    """
    print("\n" + "="*60)
    print("示例1: RL环境的基本使用")
    print("="*60)

    # 创建环境
    env = CrossNDRLEnvironment(num_turn=10)

    # 重置环境并获取初始观察
    obs = env.reset(episode_id=0)
    print(f"\n环境重置成功")
    print(f"Episode ID: {obs['episode_id']}")
    print(f"Num Turns: {obs['num_turns']}")

    # 模拟多轮预测
    predictions = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    ground_truths = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

    # 计算准确率
    episode_acc = env.compute_episode_accuracy(predictions, ground_truths)
    per_turn_acc = env.compute_per_turn_accuracy(predictions, ground_truths)

    print(f"\n预测结果: {predictions}")
    print(f"真实标签: {ground_truths}")
    print(f"\nEpisode准确率: {episode_acc:.4f}")
    print(f"逐轮准确率: {[f'{acc:.4f}' for acc in per_turn_acc]}")

    return env, episode_acc


def example_2_reward_function():
    """
    示例2: Reward函数的使用

    演示如何：
    1. 创建Reward函数
    2. 计算outcome reward
    3. 计算turn-level reward
    4. 计算综合reward
    """
    print("\n" + "="*60)
    print("示例2: Reward函数的使用")
    print("="*60)

    # 创建Reward函数
    reward_fn = CrossNDRewardFunction(
        reward_type="accuracy",
        outcome_weight=1.0,
        turn_weight=0.5,
        use_baseline=True,
    )

    # 示例数据
    predictions = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    ground_truths = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

    # 计算综合reward
    reward_comp = reward_fn.compute_combined_reward(predictions, ground_truths)

    print(f"\nOutcome Reward: {reward_comp.outcome_reward:.4f}")
    print(f"Turn-level Rewards: {[f'{r:.4f}' for r in reward_comp.turn_level_rewards]}")
    print(f"Turn-level Advantages: {[f'{a:.4f}' for a in reward_comp.turn_level_advantages]}")
    print(f"Combined Reward: {reward_comp.combined_reward:.4f}")

    # Reward Shaping
    print(f"\nApplying Reward Shaping...")
    shaper = TurnLevelRewardShaper(gamma=0.99, normalize=True, clip_range=(-1.0, 1.0))
    shaped_rewards = shaper.shape_rewards(reward_comp.turn_level_rewards)
    print(f"Shaped Rewards: {[f'{r:.4f}' for r in shaped_rewards]}")

    return reward_fn


def example_3_agent_trajectory():
    """
    示例3: Agent轨迹收集

    演示如何：
    1. 创建Agent
    2. 初始化episode
    3. 生成轨迹
    4. 获取轨迹摘要
    """
    print("\n" + "="*60)
    print("示例3: Agent轨迹生成")
    print("="*60)

    print(f"\nAgent类提供以下功能:")
    print(f"1. initialize_episode(): 初始化新的episode")
    print(f"2. predict_action(): 根据观察预测动作")
    print(f"3. step(): 执行一步并记录轨迹")
    print(f"4. generate_trajectory(): 生成完整episode轨迹")
    print(f"5. compute_trajectory_logprobs(): 计算轨迹的log概率")

    print(f"\n轨迹数据结构:")
    print(f"- AgentAction: 记录单个动作(标签、置信度、logits)")
    print(f"- AgentStep: 记录单步执行(轮次、观察、动作、reward)")
    print(f"- AgentTrajectory: 记录完整episode(所有步骤、预测、reward)")

    # 创建MultiAgentCollector
    collector = MultiAgentCollector(num_agents=4)
    print(f"\nMultiAgentCollector用于:")
    print(f"1. 并行管理多个agents")
    print(f"2. 批量收集轨迹")
    print(f"3. 计算group rewards (用于GRPO)")

    return collector


def example_4_training_config():
    """
    示例4: GRPO训练配置

    演示如何：
    1. 配置GRPO参数
    2. 创建训练器
    3. 执行训练
    """
    print("\n" + "="*60)
    print("示例4: GRPO训练配置")
    print("="*60)

    # 创建GRPO配置
    config = GRPOConfig(
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,  # 每个prompt生成4个samples
        turn_advantage_coef=0.5,  # Turn-level advantage系数
        outcome_reward_weight=1.0,
        turn_level_reward_weight=0.5,
        output_dir="./output/rl_training",
    )

    print(f"\nGRPO关键超参数:")
    print(f"- num_generations: {config.num_generations} (每个prompt的样本数)")
    print(f"- turn_advantage_coef: {config.turn_advantage_coef} (turn-level优势系数)")
    print(f"- outcome_reward_weight: {config.outcome_reward_weight}")
    print(f"- turn_level_reward_weight: {config.turn_level_reward_weight}")

    print(f"\nGRPO核心思想:")
    print(f"1. 对每个prompt生成多个response")
    print(f"2. 在group内计算relative rewards (相对于group baseline)")
    print(f"3. 使用相对reward进行策略优化")
    print(f"4. 支持turn-level advantage estimation用于更细粒度的优化")

    return config


def example_5_complete_workflow():
    """
    示例5: 完整工作流

    展示从环境、奖励、Agent到训练的完整流程
    """
    print("\n" + "="*60)
    print("示例5: 完整RL工作流")
    print("="*60)

    print(f"\n完整工作流步骤:")
    print(f"\n1. 环境交互阶段")
    print(f"   - 初始化环境: env = CrossNDRLEnvironment()")
    print(f"   - 重置环境: obs = env.reset(episode_id)")
    print(f"   - 环境步骤: obs, done = env.step(predictions, logits)")

    print(f"\n2. Agent推理阶段")
    print(f"   - 创建Agent: agent = CrossNDAgent(model, tokenizer)")
    print(f"   - 初始化episode: agent.initialize_episode(episode_id)")
    print(f"   - 预测动作: action = agent.predict_action(obs, input_ids, ...)")
    print(f"   - 生成轨迹: trajectory = agent.generate_trajectory(...)")

    print(f"\n3. Reward计算阶段")
    print(f"   - 创建Reward函数: reward_fn = CrossNDRewardFunction()")
    print(f"   - 计算Outcome Reward: outcome = reward_fn.compute_outcome_reward(...)")
    print(f"   - 计算Turn-level Reward: turns = reward_fn.compute_turn_level_rewards(...)")
    print(f"   - 计算综合Reward: reward_comp = reward_fn.compute_combined_reward(...)")

    print(f"\n4. 策略学习阶段 (GRPO)")
    print(f"   - 配置GRPO: config = GRPOConfig(...)")
    print(f"   - 创建训练器: trainer = GRPOTrainer(model, ..., config)")
    print(f"   - 执行训练: trainer.train()")

    print(f"\n5. 评估和保存")
    print(f"   - 评估模型: eval_reward = trainer._evaluate()")
    print(f"   - 保存检查点: trainer._save_checkpoint(...)")
    print(f"   - 保存历史: trainer.save_training_history()")


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "="*70)
    print(" 强化学习模块使用指南和示例")
    print("="*70)

    # 运行所有示例
    example_1_environment()
    example_2_reward_function()
    example_3_agent_trajectory()
    example_4_training_config()
    example_5_complete_workflow()

    # 使用总结
    print("\n" + "="*70)
    print("使用总结")
    print("="*70)

    print(f"""
RL模块提供的核心功能：

1. 环境 (environment.py)
   - CrossNDRLEnvironment: 多轮分类任务环境
   - 支持准确率计算和逐轮准确率
   - BatchedRLEnvironment: 批量环境处理

2. Reward函数 (reward.py)
   - 基于准确率/F1/平衡准确率的reward
   - Turn-level reward和outcome reward
   - Advantage estimation和reward shaping
   - 支持reward normalization和clipping

3. Agent (agent.py)
   - CrossNDAgent: 与环境交互的智能体
   - 支持贪心选择和采样探索
   - 轨迹记录和log概率计算
   - MultiAgentCollector: 批量管理agents

4. RL训练 (train_rl.py)
   - GRPOTrainer: GRPO算法实现
   - Group relative policy optimization
   - Turn-level advantage estimation
   - 完整的训练循环和评估

使用步骤：

1. 创建数据集和环境
   ```python
   from rl.environment import CrossNDRLEnvironment
   env = CrossNDRLEnvironment(num_turn=10)
   ```

2. 创建Reward函数
   ```python
   from rl.reward import CrossNDRewardFunction
   reward_fn = CrossNDRewardFunction(reward_type="accuracy")
   ```

3. 创建Agent
   ```python
   from rl.agent import CrossNDAgent
   agent = CrossNDAgent(model, tokenizer)
   ```

4. 配置和执行GRPO训练
   ```python
   from rl.train_rl import GRPOConfig, GRPOTrainer
   config = GRPOConfig(...)
   trainer = GRPOTrainer(model, tokenizer, dataset, reward_fn, config)
   trainer.train()
   ```

5. 推理时使用训练好的模型
   ```python
   trajectory = agent.generate_trajectory(episode_id, inputs, ...)
   predictions = trajectory.predictions
   ```

详见各模块的docstring和示例代码。
    """)

    print("="*70)
    print("示例执行完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
