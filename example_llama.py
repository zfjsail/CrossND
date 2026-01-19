"""
LlamaForCrossND 使用示例
演示如何使用新的Llama模型支持
"""

import torch
from transformers import AutoConfig, AutoTokenizer
from model import LlamaForCrossND, Qwen3ForCrossND


def example_1_basic_loading():
    """示例1：加载Llama模型"""
    print("=" * 50)
    print("示例1：加载LlamaForCrossND模型")
    print("=" * 50)
    
    # 加载配置和分词器
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    
    # 创建模型
    model = LlamaForCrossND(config)
    
    # 配置模型
    model.set_loss_type('ce')
    model.set_header(is_binary_head=False)
    
    print(f"✓ 模型初始化成功")
    print(f"✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ 损失函数类型: {model.loss_type}")
    print()


def example_2_model_comparison():
    """示例2：比较Qwen3和Llama模型"""
    print("=" * 50)
    print("示例2：Qwen3 vs LlamaForCrossND模型对比")
    print("=" * 50)
    
    models_info = []
    
    try:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-7B")
        qwen_model = Qwen3ForCrossND(config)
        models_info.append(("Qwen3ForCrossND", qwen_model))
    except:
        print("⚠ Qwen3 模型加载失败（可能未安装）")
    
    try:
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b")
        llama_model = LlamaForCrossND(config)
        models_info.append(("LlamaForCrossND", llama_model))
    except:
        print("⚠ Llama 模型加载失败（可能未安装）")
    
    for model_name, model in models_info:
        print(f"\n{model_name}:")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 隐藏大小: {model.config.hidden_size}")
        print(f"  - 词汇表大小: {model.vocab_size}")
    print()


def example_3_loss_functions():
    """示例3：测试不同的损失函数"""
    print("=" * 50)
    print("示例3：支持的损失函数类型")
    print("=" * 50)
    
    loss_types = [
        'ce',           # 交叉熵
        'kl',           # KL散度
        'ls',           # 软标签
        'ce_temperature',  # 温度缩放CE
        'ce_fl',        # CE + Focal Loss
        'ls_ranking',   # 软标签 + 排序
        'kl_ranking',   # KL + 排序
        'ce_ranking',   # CE + 排序
        'ranking',      # 排序损失
        'psl',          # PSL v1
        'psl_v2',       # PSL v2
        'psl_v3',       # PSL v3（推荐）
    ]
    
    print("支持的损失函数：")
    for i, loss_type in enumerate(loss_types, 1):
        print(f"  {i}. {loss_type}")
    print()


def example_4_forward_pass():
    """示例4：前向传播"""
    print("=" * 50)
    print("示例4：执行前向传播")
    print("=" * 50)
    
    # 模拟输入
    batch_size = 2
    seq_len = 64
    vocab_size = 32000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[labels == 0] = -100  # 填充标记
    
    print(f"输入形状: input_ids {input_ids.shape}, labels {labels.shape}")
    
    try:
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b")
        model = LlamaForCrossND(config)
        model.set_loss_type('ce')
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输出类型: {type(outputs)}")
        print(f"  - Loss: {outputs.loss}")
        print(f"  - Logits shape: {outputs.logits.shape if outputs.logits is not None else 'None'}")
    except Exception as e:
        print(f"⚠ 前向传播失败: {str(e)}")
    print()


def example_5_lora_operations():
    """示例5：LoRA操作"""
    print("=" * 50)
    print("示例5：LoRA参数管理")
    print("=" * 50)
    
    try:
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b")
        model = LlamaForCrossND(config)
        
        print("✓ LoRA 方法可用:")
        print("  - freeze_lora(): 冻结所有LoRA参数")
        print("  - unfreeze_lora(): 解冻所有LoRA参数")
        print("  - monkey_patch_cls_head(): 启用二分类头模式")
        
        # 获取模型中的LoRA参数数
        lora_params = [name for name, _ in model.named_parameters() if 'lora' in name]
        print(f"\n当前模型中的LoRA参数数: {len(lora_params)}")
        
    except Exception as e:
        print(f"⚠ LoRA 操作失败: {str(e)}")
    print()


def example_6_metadata_for_psl():
    """示例6：PSL损失的元数据格式"""
    print("=" * 50)
    print("示例6：PSL损失元数据格式")
    print("=" * 50)
    
    # 构建元数据
    metadata = [[
        {
            'p_out_sim': 0.8,      # 旧模型预测的匹配概率
            'author_sim': 0.7,     # 两个作者的相似度
            'p_in_sim': 0.75       # 当前模型预测的匹配概率（可选）
        },
        {
            'p_out_sim': 0.6,
            'author_sim': 0.5,
            'p_in_sim': 0.65
        },
    ]]
    
    print("元数据结构示例:")
    print(f"metadata = {metadata}")
    print("\n每个样本的字段说明:")
    print("  - p_out_sim: 旧模型预测的匹配概率 [0, 1]")
    print("  - author_sim: 两个作者的相似度 [0, 1]")
    print("  - p_in_sim: 当前模型预测的匹配概率 [0, 1]（可选）")
    print()


def example_7_model_switching():
    """示例7：在Qwen3和Llama之间切换"""
    print("=" * 50)
    print("示例7：模型切换示例")
    print("=" * 50)
    
    def create_model(model_type='llama'):
        if model_type == 'llama':
            config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b")
            return LlamaForCrossND(config)
        else:
            config = AutoConfig.from_pretrained("Qwen/Qwen3-7B")
            return Qwen3ForCrossND(config)
    
    print("模型创建代码:")
    print("""
    # 使用Llama
    model = create_model('llama')
    
    # 使用Qwen3
    model = create_model('qwen3')
    
    # 其余代码保持不变
    model.set_loss_type('psl_v3')
    output = model(input_ids=input_ids, labels=labels)
    """)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("LlamaForCrossND 使用示例")
    print("=" * 50 + "\n")
    
    # 运行示例
    example_1_basic_loading()
    example_2_model_comparison()
    example_3_loss_functions()
    example_4_forward_pass()
    example_5_lora_operations()
    example_6_metadata_for_psl()
    example_7_model_switching()
    
    print("=" * 50)
    print("所有示例完成！")
    print("=" * 50)



