# RL模块快速入门

## 5分钟快速开始

### 基础功能测试

```python
from rl.environment import CrossNDRLEnvironment
from rl.reward import CrossNDRewardFunction

# 1. 创建环境
env = CrossNDRLEnvironment(num_turn=10)

# 2. 模拟多轮预测
predictions = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
ground_truths = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

# 3. 计算准确率
acc = env.compute_episode_accuracy(predictions, ground_truths)
print(f"Accuracy: {acc:.4f}")  # 0.8000

# 4. 创建Reward函数
reward_fn = CrossNDRewardFunction(reward_type="accuracy")

# 5. 计算Reward
reward_comp = reward_fn.compute_combined_reward(predictions, ground_truths)
print(f"Outcome Reward: {reward_comp.outcome_reward:.4f}")     # 0.6000
print(f"Combined Reward: {reward_comp.combined_reward:.4f}")
```

### 完整训练流程

```python
from rl.train_rl import GRPOConfig, GRPOTrainer
from rl.reward import CrossNDRewardFunction
from utils import CrossNDDataset

# 1. 准备数据
train_dataset = CrossNDDataset(...)
eval_dataset = CrossNDDataset(...)

# 2. 配置GRPO
config = GRPOConfig(
    learning_rate=5e-5,
    num_epochs=3,
    batch_size=4,
    output_dir="./output/rl_training"
)

# 3. 创建Reward函数
reward_fn = CrossNDRewardFunction(reward_type="accuracy")

# 4. 创建训练器
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_fn=reward_fn,
    config=config
)

# 5. 训练
trainer.train()
```

## 核心概念

### Reward设计

- **Outcome Reward**: 基于最终准确率，范围[-1, 1]
- **Turn-level Reward**: 基于每一轮的累积准确率改进
- **Advantage**: 相对于baseline的优势，用于variance reduction

### GRPO算法

1. 生成多个samples (num_generations)
2. 计算每个sample的reward
3. 在group内计算相对reward
4. 使用策略梯度更新模型

### 多轮任务

- 每个episode包含 `num_turn` 个分类任务
- 模型依次预测每个任务的标签
- 最终reward基于所有预测的准确率

## 常见用法

### 用法1: 评估模型

```python
from rl.environment import CrossNDRLEnvironment
from rl.agent import CrossNDAgent

env = CrossNDRLEnvironment(num_turn=10, dataset=test_dataset)
agent = CrossNDAgent(model, tokenizer)

total_acc = 0
for i in range(len(test_dataset)):
    trajectory = agent.generate_trajectory(i, ...)
    episode_data = env.get_episode_data(i)
    acc = env.compute_episode_accuracy(
        trajectory.predictions,
        episode_data.ground_truths
    )
    total_acc += acc

avg_acc = total_acc / len(test_dataset)
print(f"Average Accuracy: {avg_acc:.4f}")
```

### 用法2: 比较不同reward函数

```python
from rl.reward import CrossNDRewardFunction

# 使用不同的reward类型
reward_fns = {
    'accuracy': CrossNDRewardFunction(reward_type="accuracy"),
    'f1': CrossNDRewardFunction(reward_type="f1"),
    'balanced': CrossNDRewardFunction(reward_type="balanced_acc"),
}

for name, reward_fn in reward_fns.items():
    reward = reward_fn.compute_combined_reward(predictions, labels)
    print(f"{name}: {reward.combined_reward:.4f}")
```

### 用法3: 调整训练参数

```python
from rl.train_rl import GRPOConfig

# 方案1: 快速迭代 (少量epoch, 小batch)
config_fast = GRPOConfig(
    num_epochs=1,
    batch_size=2,
    num_generations=2,  # 少生成samples
    learning_rate=1e-4,  # 较大学习率
)

# 方案2: 精细调优 (多epoch, 大batch)
config_fine = GRPOConfig(
    num_epochs=10,
    batch_size=32,
    num_generations=8,  # 多生成samples
    learning_rate=1e-5,  # 较小学习率
)

# 方案3: 长期训练 (持续优化)
config_long = GRPOConfig(
    num_epochs=100,
    batch_size=16,
    num_generations=4,
    learning_rate=5e-5,
    warmup_steps=1000,
    weight_decay=0.01,
)
```

## 文件结构说明

```
rl/
├── environment.py       # 任务环境定义
├── reward.py           # Reward计算
├── agent.py            # Agent实现
├── train_rl.py         # GRPO训练脚本
├── examples.py         # 使用示例
├── __init__.py         # 模块初始化
├── README.md           # 详细文档
├── INTEGRATION_GUIDE.md # 集成指南
└── QUICK_START.md      # 本文件
```

## 常见问题

**Q: 我应该用方案A (直接GRPO) 还是方案B (SL+GRPO)?**

A: 推荐方案B:
- 先用现有train.py进行SL预训练 (1-2天)
- 再用GRPO进行RL微调 (6-12小时)
- 效果通常会更好

**Q: 多少个num_turn比较合适?**

A: 取决于数据集:
- 小数据集 (论文数<100): 5-10轮
- 中等数据集 (论文数100-1000): 10-20轮
- 大数据集 (论文数>1000): 20-50轮
- 默认推荐: 10轮

**Q: Outcome reward和Turn reward的权重应该怎么设置?**

A: 从默认值开始:
```python
outcome_weight=1.0,      # 最终结果
turn_weight=0.5,         # 过程改进
```

- 若关注最终结果: outcome_weight=2.0, turn_weight=0.2
- 若关注过程改进: outcome_weight=1.0, turn_weight=1.0

**Q: 训练为什么这么慢?**

A: 可能的原因和解决方案:
1. num_generations太大 → 减小到2-4
2. batch_size太小 → 增大到8-16
3. 数据集很大 → 使用sampler采样
4. 模型很大 → 使用量化或剪枝

**Q: 如何监控训练进度?**

A: 查看输出目录中的文件:
```python
import json

# 查看训练历史
history = json.load(open("output/rl_training/training_history.json"))
print(f"Final Loss: {history['losses'][-1]}")
print(f"Final Reward: {history['outcome_rewards'][-1]}")

# 绘制曲线
import matplotlib.pyplot as plt
plt.plot(history['steps'], history['losses'])
plt.title('Training Loss')
plt.savefig('loss_curve.png')
```

## 性能指标

### 推荐的超参数范围

| 参数 | 推荐范围 | 默认值 |
|------|---------|--------|
| learning_rate | 1e-5 ~ 5e-4 | 5e-5 |
| num_epochs | 1 ~ 10 | 3 |
| batch_size | 2 ~ 32 | 4 |
| num_generations | 2 ~ 8 | 4 |
| turn_advantage_coef | 0.1 ~ 1.0 | 0.5 |

### 典型训练时间

| 配置 | 数据集规模 | GPU | 时间 |
|------|-----------|-----|------|
| 快速 | 1K | V100 | ~1小时 |
| 标准 | 10K | V100 | ~6小时 |
| 仔细 | 100K | 8×V100 | ~24小时 |

## 下一步

1. 运行示例代码验证基本功能
   ```bash
   python rl/examples.py
   ```

2. 阅读详细文档
   - 基础: `rl/README.md`
   - 集成: `rl/INTEGRATION_GUIDE.md`

3. 在小规模数据上测试
   - 准备小样本数据 (~1K samples)
   - 运行1-2个epoch的GRPO
   - 对比SL和GRPO的性能

4. 扩展到完整数据集
   - 在完整数据上训练
   - 调优超参数
   - 分析结果

## 联系和反馈

有任何问题，请参考:
- `RL_IMPLEMENTATION_SUMMARY.md` - 完整实现总结
- `rl/README.md` - 详细API文档
- `rl/INTEGRATION_GUIDE.md` - 集成指南
