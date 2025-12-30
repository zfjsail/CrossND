# 强化学习模块 (RL Module)

## 概述

本模块为论文异常分配检测任务实现了一套完整的强化学习框架，基于多轮分类设计。通过GRPO (Group Relative Policy Optimization) 算法，对多轮预测的累积准确率进行优化。

## 核心组件

### 1. 环境 (environment.py)

**CrossNDRLEnvironment**: 多轮论文异常分配检测的RL环境

```python
from rl.environment import CrossNDRLEnvironment

# 创建环境
env = CrossNDRLEnvironment(num_turn=10, dataset=dataset)

# 重置环境，开始新的episode
obs = env.reset(episode_id=0)

# 执行环境步骤
obs, done = env.step(predictions, logits)

# 计算准确率
acc = env.compute_episode_accuracy(predictions, ground_truths)
per_turn_acc = env.compute_per_turn_accuracy(predictions, ground_truths)
```

**主要特性**:
- 支持 `num_turn` 轮数的多轮分类
- 计算整体和逐轮准确率
- 获取episode的完整数据
- BatchedRLEnvironment 用于批量处理

### 2. Reward函数 (reward.py)

**CrossNDRewardFunction**: 多种reward计算方式

```python
from rl.reward import CrossNDRewardFunction, TurnLevelRewardShaper

# 创建Reward函数
reward_fn = CrossNDRewardFunction(
    reward_type="accuracy",  # "accuracy", "f1", "balanced_acc"
    outcome_weight=1.0,      # outcome reward权重
    turn_weight=0.5,         # turn-level reward权重
    use_baseline=True,       # 使用baseline计算advantage
)

# 计算综合reward
reward_comp = reward_fn.compute_combined_reward(predictions, ground_truths)
print(f"Outcome Reward: {reward_comp.outcome_reward}")
print(f"Turn-level Rewards: {reward_comp.turn_level_rewards}")
print(f"Advantages: {reward_comp.turn_level_advantages}")
print(f"Combined Reward: {reward_comp.combined_reward}")
```

**Reward类型**:
- **Outcome Reward**: 基于最终准确率，范围[-1, 1]，公式: `acc * 2 - 1`
- **Turn-level Reward**: 基于每一轮的累积准确率改进
- **Advantage**: 相对于baseline的优势估计，用于variance reduction

**Reward Shaping**:
```python
shaper = TurnLevelRewardShaper(gamma=0.99, normalize=True)
shaped_rewards = shaper.shape_rewards(rewards)
```

### 3. Agent (agent.py)

**CrossNDAgent**: 与环境交互的智能体

```python
from rl.agent import CrossNDAgent, MultiAgentCollector

# 创建单个Agent
agent = CrossNDAgent(
    model=model,
    tokenizer=tokenizer,
    num_turns=10,
    temperature=1.0,
    use_sampling=False,  # 使用贪心还是采样
)

# 初始化episode
obs = agent.initialize_episode(episode_id=0)

# 预测单个动作
action = agent.predict_action(obs, input_ids, attention_mask)

# 生成完整episode轨迹
trajectory = agent.generate_trajectory(
    episode_id=0,
    input_ids_list=input_ids_list,
    attention_mask_list=attention_mask_list,
)

# 获取轨迹摘要
summary = agent.get_trajectory_summary()
```

**多Agent管理**:
```python
# 创建多个Agent进行并行推理
collector = MultiAgentCollector(num_agents=4)
for _ in range(4):
    agent = CrossNDAgent(model, tokenizer)
    collector.add_agent(agent)

# 批量收集轨迹
trajectories = collector.collect_trajectories_batch(
    episode_ids=[0, 1, 2, 3],
    batch_input_ids=batch_input_ids,
    batch_attention_masks=batch_attention_masks,
)

# 获取group rewards (用于GRPO)
group_rewards = collector.get_group_rewards()
```

### 4. GRPO训练 (train_rl.py)

**GRPOTrainer**: Group Relative Policy Optimization 训练器

```python
from rl.train_rl import GRPOConfig, GRPOTrainer

# 配置GRPO
config = GRPOConfig(
    learning_rate=5e-5,
    num_epochs=3,
    batch_size=4,
    num_generations=4,           # 每个prompt的样本数
    turn_advantage_coef=0.5,      # Turn-level优势系数
    outcome_reward_weight=1.0,
    turn_level_reward_weight=0.5,
    output_dir="./output/rl_training",
)

# 创建训练器
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_fn=reward_fn,
    config=config,
)

# 执行训练
trainer.train()

# 保存训练历史
trainer.save_training_history()
```

**GRPO核心思想**:
1. 对同一个prompt生成多个response
2. 计算每个response的outcome reward和turn-level reward
3. 在group内计算relative reward: `reward - group_baseline`
4. 使用relative reward进行策略梯度优化
5. 支持turn-level advantage estimation用于更细粒度的优化

## 数据流

```
Input Data (论文信息)
    ↓
Environment (多轮分类任务)
    ↓
Agent (生成预测轨迹)
    ↓
Model Forward (获取logits)
    ↓
Reward Function (计算reward)
    ├─ Outcome Reward (最终准确率)
    ├─ Turn-level Reward (逐轮改进)
    └─ Advantage (相对优势)
    ↓
GRPO Loss (策略梯度)
    ↓
Model Update (梯度优化)
```

## 使用示例

### 示例1: 基本的Reward计算

```python
from rl.reward import CrossNDRewardFunction

# 创建Reward函数
reward_fn = CrossNDRewardFunction(reward_type="accuracy")

# 模拟预测和标签
predictions = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
ground_truths = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

# 计算rewards
reward_comp = reward_fn.compute_combined_reward(predictions, ground_truths)

print(f"Outcome Reward: {reward_comp.outcome_reward:.4f}")      # 0.8000
print(f"Turn Rewards: {reward_comp.turn_level_rewards}")
print(f"Combined Reward: {reward_comp.combined_reward:.4f}")
```

### 示例2: 环境和Agent的集成

```python
from rl.environment import CrossNDRLEnvironment
from rl.agent import CrossNDAgent

# 创建环境
env = CrossNDRLEnvironment(num_turn=10)

# 创建Agent
agent = CrossNDAgent(model, tokenizer)

# 生成轨迹
trajectory = agent.generate_trajectory(
    episode_id=0,
    input_ids_list=input_ids_list,
    attention_mask_list=attention_mask_list,
)

# 从环境获取ground truth
episode_data = env.get_episode_data(episode_id=0)

# 计算准确率
acc = env.compute_episode_accuracy(
    trajectory.predictions,
    episode_data.ground_truths
)
print(f"Episode Accuracy: {acc:.4f}")
```

### 示例3: 完整的GRPO训练

```python
from rl.train_rl import GRPOConfig, GRPOTrainer
from rl.reward import CrossNDRewardFunction

# 准备数据、模型等
# ... (省略初始化代码)

# 配置
config = GRPOConfig(
    num_epochs=5,
    batch_size=8,
    learning_rate=5e-5,
)

# 创建Reward函数
reward_fn = CrossNDRewardFunction(reward_type="accuracy")

# 创建训练器
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_fn=reward_fn,
    config=config,
)

# 训练
trainer.train()
```

## 配置参数说明

### GRPOConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 5e-5 | 学习率 |
| num_epochs | 3 | 训练epoch数 |
| batch_size | 4 | 批大小 |
| num_generations | 4 | 每个prompt的生成样本数 |
| turn_advantage_coef | 0.5 | Turn-level advantage系数 |
| outcome_reward_weight | 1.0 | Outcome reward的权重 |
| turn_level_reward_weight | 0.5 | Turn-level reward的权重 |
| max_grad_norm | 1.0 | 梯度裁剪上界 |
| output_dir | "./output/rl_training" | 模型输出目录 |

## 与现有代码的集成

该RL模块设计用于与现有的CrossND系统集成：

1. **模型**: 使用已有的 `Qwen3ForCrossND` 或 `LlamaForCrossND` 模型
2. **数据集**: 使用 `CrossNDDataset` 数据集 (支持 `num_turn` 参数)
3. **Tokenizer**: 使用 `Qwen3Tokenizer` 或其他兼容的tokenizer
4. **特殊Token**: 支持已有的 `LABEL_TOKEN`, `EMBED_TOKEN` 等

## 关键设计决策

### 1. Reward设计

采用两层次reward结构：
- **Outcome Reward**: 衡量最终结果的准确率
- **Turn-level Reward**: 衡量每一轮的改进速度

这种设计鼓励模型在多轮任务中逐步改进，而不是只关注最终结果。

### 2. Advantage Estimation

使用baseline来计算advantage:
- **Moving Average Baseline**: 跟踪历史reward的指数移动平均
- **Previous Turn Baseline**: 使用前一轮的reward

这有助于减少variance，加速训练收敛。

### 3. GRPO算法

Group Relative Policy Optimization的核心思想：
- 在相同prompt的samples之间计算相对reward
- 使用group内的统计量作为baseline
- 支持turn-level granularity的优化

## 扩展和改进

### 可能的改进方向

1. **更复杂的Reward设计**
   - 基于论文相似度的reward
   - 基于作者特征的reward
   - 组合多个reward函数

2. **更高级的RL算法**
   - PPO (Proximal Policy Optimization)
   - A3C (Asynchronous Advantage Actor-Critic)
   - TRPO (Trust Region Policy Optimization)

3. **更好的Value函数**
   - 独立的value network
   - GAE (Generalized Advantage Estimation)

4. **并行化和分布式训练**
   - 多GPU/多机支持
   - Experience replay buffer

## 常见问题

### Q: 如何调整reward的权重？

A: 通过 `GRPOConfig` 中的参数：
```python
config = GRPOConfig(
    outcome_reward_weight=1.5,    # 增加outcome reward权重
    turn_level_reward_weight=0.3,  # 减少turn-level reward权重
)
```

### Q: 多少轮数比较合适？

A: 根据数据集的特点：
- 较小作者库: 5-10轮
- 较大作者库: 10-20轮
- 默认推荐: 10轮

### Q: 如何处理不平衡的标签？

A: 可以使用不同的reward类型：
```python
reward_fn = CrossNDRewardFunction(reward_type="f1")  # F1分数
# 或
reward_fn = CrossNDRewardFunction(reward_type="balanced_acc")  # 平衡准确率
```

### Q: 训练不收敛怎么办？

A: 尝试以下方法：
1. 降低学习率
2. 增加gradient accumulation steps
3. 启用reward normalization
4. 调整reward权重

## 文件结构

```
rl/
├── __init__.py              # 模块初始化
├── environment.py           # RL环境定义
├── reward.py               # Reward函数实现
├── agent.py                # Agent实现
├── train_rl.py             # GRPO训练脚本
├── examples.py             # 使用示例
└── README.md               # 本文档
```

## 参考文献

- 论文异常分配检测: CrossND 数据集
- GRPO算法: https://github.com/SiliangZeng/Multi-Turn-RL-Agent
- 强化学习基础: Sutton & Barto, "Reinforcement Learning: An Introduction"

## 许可证

与主项目保持一致

## 作者

基于CrossND项目和Multi-Turn-RL-Agent项目构建
