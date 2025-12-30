#!/usr/bin/env python3
"""
验证 LoRA 是否真的被训练了
通过对比训练前后的 LoRA 权重，检测是否有实际的参数更新
"""

import sys
sys.path.insert(0, '.')

import torch
import os
from transformers import AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from utils import special_token_dict, DEFAULT_PAD_TOKEN
from model import Qwen3ForCrossND
from safetensors.torch import safe_open
import json

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

print("=" * 80)
print("验证方案：对比训练前后的 LoRA 权重变化")
print("=" * 80)

model_path = "/workspace/pangyunhe/models/Qwen/Qwen3-8B"
checkpoint_path = "/workspace/pangyunhe/project/crossnd/llm/output/kddcup/gen_psl_v2_turn/checkpoint-1200"

# =====================================
# 方案 1：计算权重 hash，看是否真的改变了
# =====================================
print("\n【方案 1】检查 adapter 权重的 hash 值是否变化")
print("-" * 80)

adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')

# 读取保存的 adapter 权重
saved_weights = {}
with safe_open(adapter_model_path, framework='pt', device='cpu') as f:
    for key in f.keys():
        if 'lora_' in key:  # 只关心 LoRA 权重
            saved_weights[key] = f.get_tensor(key)

print(f"\n✓ 从 checkpoint 读取了 {len(saved_weights)} 个 LoRA 权重")

# 计算所有 LoRA 权重的统计信息
print("\n【保存的 LoRA 权重统计】")
for key in sorted(list(saved_weights.keys())[:3]):  # 只显示前3个
    w = saved_weights[key]
    print(f"\n{key}:")
    print(f"  shape: {w.shape}")
    print(f"  mean: {w.mean().item():.6f}")
    print(f"  std: {w.std().item():.6f}")
    print(f"  min: {w.min().item():.6f}")
    print(f"  max: {w.max().item():.6f}")

# =====================================
# 方案 2：和基础模型的初始化权重对比
# =====================================
print("\n\n【方案 2】对比初始化的 LoRA 权重 vs 保存的权重")
print("-" * 80)

# 创建一个新的模型（未训练）
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model_fresh = Qwen3ForCrossND.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    config=config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# 应用特殊tokens
special_token_dict_copy = {'additional_special_tokens': special_token_dict.get('additional_special_tokens', [])}
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_token_dict_copy,
    tokenizer=tokenizer,
    model=model_fresh,
)

# 应用 LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    task_type="SEQ_CLS",
    modules_to_save=["lm_head", "embed_tokens"]
)
model_fresh = get_peft_model(model_fresh, peft_config)

# 获取初始化的 LoRA 权重
print("\n初始化（未训练）的 LoRA 权重统计：")
init_weights = {}
for name, param in model_fresh.named_parameters():
    if 'lora_' in name:
        init_weights[name] = param.data.clone().detach()

for key in sorted(list(init_weights.keys())[:3]):  # 只显示前3个
    w = init_weights[key]
    print(f"\n{key}:")
    print(f"  mean: {w.mean().item():.6f}")
    print(f"  std: {w.std().item():.6f}")

# 对比：计算权重变化量
print("\n\n【关键对比：权重是否有实质改变】")
print("-" * 80)

total_params = 0
changed_params = 0
total_change = 0.0

for key in saved_weights.keys():
    if 'lora_' in key:
        saved = saved_weights[key]
        # 对应的初始化权重（需要匹配名称）
        init_key = key.replace('base_model.model.', '')
        
        if init_key in init_weights:
            init = init_weights[init_key]
            
            # 计算权重差异
            diff = (saved - init).abs()
            change = diff.mean().item()
            
            total_params += saved.numel()
            total_change += change * saved.numel()
            
            if change > 1e-6:  # 如果有显著改变
                changed_params += saved.numel()
                print(f"\n✓ {key}")
                print(f"  平均改变量: {change:.8f}")
                print(f"  最大改变: {diff.max().item():.8f}")
            
print(f"\n\n【最终统计】")
print(f"总参数数: {total_params:,}")
print(f"改变的参数数: {changed_params:,}")
print(f"改变比例: {100 * changed_params / total_params:.2f}%")
print(f"平均改变量: {total_change / total_params:.8f}")

if changed_params / total_params < 0.01:
    print("\n🔴 【警告】LoRA 权重几乎没有改变！")
    print("   这表明 save_only_model 可能导致权重没有被正确保存")
    print("   或者 LoRA 在训练时根本没有被激活")
elif changed_params / total_params > 0.5:
    print("\n✅ LoRA 权重有显著改变，训练正常")
else:
    print("\n⚠️ LoRA 权重有部分改变，但改变比例偏低")

# =====================================
# 方案 3：检查特定层的 LoRA 权重活跃度
# =====================================
print("\n\n【方案 3】按层检查 LoRA 权重变化分布")
print("-" * 80)

layer_changes = {}
for key in saved_weights.keys():
    if 'lora_B' in key:  # 只看 lora_B（更容易看出变化）
        saved = saved_weights[key]
        init_key = key.replace('base_model.model.', '')
        
        if init_key in init_weights:
            init = init_weights[init_key]
            diff = (saved - init).abs()
            change = diff.mean().item()
            
            # 提取层号
            layer_match = None
            if 'layers.' in key:
                import re
                match = re.search(r'layers\.(\d+)', key)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num not in layer_changes:
                        layer_changes[layer_num] = []
                    layer_changes[layer_num].append(change)

if layer_changes:
    print("\n按层的 LoRA_B 权重变化：")
    for layer in sorted(layer_changes.keys()):
        avg_change = sum(layer_changes[layer]) / len(layer_changes[layer])
        print(f"  Layer {layer:2d}: {avg_change:.8f}")
    
    # 检查是否有某些层完全没有改变
    zero_layers = [l for l, changes in layer_changes.items() 
                   if all(c < 1e-8 for c in changes)]
    if zero_layers:
        print(f"\n🔴 这些层的 LoRA 权重基本没改变: {zero_layers}")
        print("   可能是这些层在训练中没有被正确访问")
else:
    print("未找到层级信息")

# =====================================
# 方案 4：检查 lm_head 和 embed_tokens
# =====================================
print("\n\n【方案 4】检查 modules_to_save 的权重变化")
print("-" * 80)

if 'base_model.model.lm_head.weight' in saved_weights:
    lm_head_saved = saved_weights['base_model.model.lm_head.weight']
    print(f"\nlm_head.weight:")
    print(f"  shape: {lm_head_saved.shape}")
    print(f"  mean: {lm_head_saved.mean().item():.6f}")
    print(f"  std: {lm_head_saved.std().item():.6f}")
    
    # 检查是否有训练迹象（比较初始化和保存）
    init_lm_head = model_fresh.base_model.lm_head.weight.data
    diff = (lm_head_saved - init_lm_head).abs()
    print(f"  vs init 的平均改变: {diff.mean().item():.8f}")
    
    if diff.mean().item() < 1e-6:
        print(f"  ⚠️ 基本没有改变 - 检查 modules_to_save 是否正确")

if 'base_model.model.model.embed_tokens.weight' in saved_weights:
    embed_saved = saved_weights['base_model.model.model.embed_tokens.weight']
    print(f"\nembed_tokens.weight:")
    print(f"  shape: {embed_saved.shape}")
    print(f"  mean: {embed_saved.mean().item():.6f}")
    print(f"  std: {embed_saved.std().item():.6f}")
    
    init_embed = model_fresh.base_model.model.embed_tokens.weight.data
    diff = (embed_saved - init_embed).abs()
    print(f"  vs init 的平均改变: {diff.mean().item():.8f}")
    
    if diff.mean().item() < 1e-6:
        print(f"  ⚠️ 基本没有改变 - 检查 modules_to_save 是否正确")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)

