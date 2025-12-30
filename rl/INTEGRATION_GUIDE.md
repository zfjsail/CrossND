#!/usr/bin/env python
# coding=utf-8
"""
RL模块与现有CrossND系统的集成指南

本文档说明如何将RL模块集成到现有的训练管道中
"""

# ============================================================================
# 集成指南: 如何在train.py中使用RL模块
# ============================================================================

"""
步骤1: 导入RL模块

在 train.py 中添加以下导入:

```python
from rl.environment import CrossNDRLEnvironment
from rl.reward import CrossNDRewardFunction
from rl.agent import CrossNDAgent, MultiAgentCollector
from rl.train_rl import GRPOConfig, GRPOTrainer
```
"""

# ============================================================================
# 方案A: 直接使用GRPO训练替换原有训练
# ============================================================================

"""
如果希望完全替换原有的标准监督学习训练，使用GRPO：

1. 准备阶段:
   ```python
   # 加载模型、数据集、tokenizer等（同train.py）
   model = Qwen3ForCrossND.from_pretrained(...)
   train_dataset = CrossNDDataset(...)

   # 创建Reward函数
   reward_fn = CrossNDRewardFunction(
       reward_type="accuracy",
       outcome_weight=1.0,
       turn_weight=0.5,
   )

   # 配置GRPO
   grpo_config = GRPOConfig(
       learning_rate=5e-5,
       num_epochs=3,
       batch_size=4,
       num_generations=4,  # 核心: 为每个样本生成多个predictions
       turn_advantage_coef=0.5,
   )
   ```

2. 训练阶段:
   ```python
   trainer = GRPOTrainer(
       model=model,
       tokenizer=tokenizer,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       reward_fn=reward_fn,
       config=grpo_config,
   )
   trainer.train()
   ```

优点:
- 利用多轮结构的特性
- 显式优化最终准确率
- 支持turn-level的细粒度优化

缺点:
- 需要完全重新训练
- 计算量较大 (num_generations倍)
"""

# ============================================================================
# 方案B: 混合训练 (推荐)
# ============================================================================

"""
先使用标准监督学习预训练，再用GRPO微调:

1. 第一阶段: 标准监督学习
   ```python
   # 使用现有的train.py进行预训练
   # python train.py --model_path <model> --data_dir <data> --num_turn 10

   # 保存预训练模型
   model = trainer.model
   torch.save(model.state_dict(), "pretrained_model.pt")
   ```

2. 第二阶段: GRPO微调
   ```python
   # 加载预训练模型
   model = Qwen3ForCrossND.from_pretrained(...)
   model.load_state_dict(torch.load("pretrained_model.pt"))

   # 创建RL components
   reward_fn = CrossNDRewardFunction(...)
   grpo_config = GRPOConfig(
       learning_rate=1e-5,  # 更小的学习率用于微调
       num_epochs=2,
       batch_size=4,
   )

   # 使用GRPO微调
   trainer = GRPOTrainer(...)
   trainer.train()
   ```

优点:
- 既保留预训练的好处，又能优化多轮准确率
- 训练更稳定
- 计算量相对可控

推荐使用这个方案！
"""

# ============================================================================
# 方案C: 在推理阶段使用RL Agent
# ============================================================================

"""
保留标准监督学习训练，但在推理时使用RL Agent:

1. 训练阶段: 使用现有的train.py

2. 推理阶段: 使用RL Agent
   ```python
   from rl.agent import CrossNDAgent

   # 加载训练好的模型
   model = Qwen3ForCrossND.from_pretrained(...)

   # 创建Agent
   agent = CrossNDAgent(
       model=model,
       tokenizer=tokenizer,
       num_turns=10,
       use_sampling=False,  # 推理时使用贪心
   )

   # 推理
   for sample in test_data:
       trajectory = agent.generate_trajectory(
           episode_id=sample['id'],
           input_ids_list=sample['input_ids_list'],
           attention_mask_list=sample['attention_mask_list'],
       )
       predictions = trajectory.predictions
   ```

优点:
- 最小改动现有代码
- 快速验证RL思想的有效性
- 推理效率高

缺点:
- 不能从RL中获得完整训练收益
"""

# ============================================================================
# 方案D: 评估和对比
# ============================================================================

"""
评估RL训练的效果:

1. 创建评估脚本:
   ```python
   from rl.environment import CrossNDRLEnvironment
   from sklearn.metrics import roc_auc_score, average_precision_score

   env = CrossNDRLEnvironment()

   # 标准监督学习的预测
   sl_predictions = model(test_data)

   # RL微调后的预测
   rl_predictions = rl_model(test_data)

   # 计算author-level AUC/MAP
   def compute_author_level_metrics(predictions, labels, metadata):
       author_preds = {}
       author_labels = {}
       for pred, label, meta in zip(predictions, labels, metadata):
           author_id = meta['aid1']
           if author_id not in author_preds:
               author_preds[author_id] = []
               author_labels[author_id] = []
           author_preds[author_id].append(pred)
           author_labels[author_id].append(label)

       auc_scores = []
       for author_id in author_preds:
           auc = roc_auc_score(
               author_labels[author_id],
               author_preds[author_id]
           )
           auc_scores.append(auc)
       return np.mean(auc_scores)

   sl_auc = compute_author_level_metrics(sl_predictions, ...)
   rl_auc = compute_author_level_metrics(rl_predictions, ...)

   print(f"SL AUC: {sl_auc:.4f}")
   print(f"RL AUC: {rl_auc:.4f}")
   print(f"Improvement: {(rl_auc - sl_auc) / sl_auc * 100:.2f}%")
   ```

2. 分析reward曲线:
   ```python
   import json
   history = json.load(open("output/rl_training/training_history.json"))

   import matplotlib.pyplot as plt
   plt.plot(history['steps'], history['outcome_rewards'])
   plt.xlabel('Global Step')
   plt.ylabel('Outcome Reward')
   plt.title('GRPO Training Reward Curve')
   plt.savefig('reward_curve.png')
   ```
"""

# ============================================================================
# 完整集成示例脚本
# ============================================================================

"""
如果创建一个新的 train_rl.py 脚本:

#!/usr/bin/env python
# coding=utf-8

import sys
import os
import torch
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from train import DataArguments, ModelArguments
from utils import CrossNDDataset, CrossNDCollator, special_token_dict
from model import Qwen3ForCrossND, LlamaForCrossND

from rl.reward import CrossNDRewardFunction
from rl.train_rl import GRPOConfig, GRPOTrainer

def main():
    # 解析参数
    parser = HfArgumentParser((DataArguments, ModelArguments, GRPOConfig))
    data_args, model_args, grpo_config = parser.parse_args_into_dataclasses()

    # 加载模型和数据集 (复用train.py中的代码)
    config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

    # 添加特殊tokens
    tokenizer.add_special_tokens(
        {'additional_special_tokens': special_token_dict['additional_special_tokens']}
    )

    # 加载模型
    if "qwen" in model_args.model_path.lower():
        model = Qwen3ForCrossND.from_pretrained(...)
    else:
        model = LlamaForCrossND.from_pretrained(...)

    # 加载数据集
    train_dataset = CrossNDDataset(...)
    eval_dataset = CrossNDDataset(...)

    # 创建Reward函数
    reward_fn = CrossNDRewardFunction(reward_type="accuracy")

    # 创建训练器并训练
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_fn=reward_fn,
        config=grpo_config,
    )

    trainer.train()
    trainer.save_training_history()

if __name__ == "__main__":
    main()
"""

# ============================================================================
# 参数对应关系
# ============================================================================

"""
CrossND原有参数 <-> RL模块参数的对应:

原有参数                          RL对应参数
─────────────────────────────────────────────────────────
model_path                       GRPOTrainer.model
data_dir                         train_dataset
num_turn                         GRPOConfig (env会自动检测)
learning_rate                    GRPOConfig.learning_rate
num_epochs                       GRPOConfig.num_epochs
batch_size                       GRPOConfig.batch_size
output_dir                       GRPOConfig.output_dir

新增RL特定参数:
─────────────────────────────────────────────────────────
num_generations                  为每个样本生成的数量
turn_advantage_coef              Turn-level advantage系数
outcome_reward_weight            Outcome reward权重
turn_level_reward_weight         Turn-level reward权重
use_turn_level_advantage         是否使用turn-level advantage
"""

# ============================================================================
# 命令行使用示例
# ============================================================================

"""
# 方案A: 直接GRPO训练 (如果实现了完整的train_rl.py)
python rl/train_rl.py \\
    --model_path qwen/qwen3-7b \\
    --data_dir ./data \\
    --learning_rate 5e-5 \\
    --num_epochs 3 \\
    --batch_size 4 \\
    --output_dir ./output/rl_training

# 方案B: 首先标准训练
python train.py \\
    --model_path qwen/qwen3-7b \\
    --data_dir ./data \\
    --num_turn 10 \\
    --output_dir ./output/pretrained

# 然后用GRPO微调
python rl/train_rl.py \\
    --model_path ./output/pretrained \\
    --data_dir ./data \\
    --learning_rate 1e-5 \\
    --num_epochs 2 \\
    --output_dir ./output/rl_finetuned
"""

# ============================================================================
# 故障排查
# ============================================================================

"""
常见问题:

1. ImportError: 模块找不到
   解决: 确保 rl/ 目录在Python路径中
   ```python
   import sys
   sys.path.insert(0, '/path/to/llm/dir')
   from rl.environment import CrossNDRLEnvironment
   ```

2. CUDA内存不足
   解决:
   - 减小batch_size
   - 增加gradient_accumulation_steps
   - 使用mixed precision training

3. Loss为NaN
   解决:
   - 检查reward计算是否正常
   - 减小学习率
   - 启用gradient clipping

4. 训练很慢
   解决:
   - num_generations不要设太大
   - 使用多GPU/分布式训练
   - 优化dataloader的num_workers

5. 评估reward不提升
   解决:
   - 调整reward权重
   - 尝试不同的reward_type
   - 增加训练epoch
"""

if __name__ == "__main__":
    print("RL集成指南")
    print("详见本文件的说明和示例代码")
