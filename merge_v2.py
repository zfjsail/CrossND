import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====== 配置 ======
base_model_path = "/workspace/pangyunhe/models/Qwen/Qwen3-8B"
lora_model_path = "outputs/sft_turn20/global_step_140"
output_path = "lora"

torch_dtype = torch.float16  # 或 bfloat16 / float32
device_map = "auto"
# ==================

def main():
    # 1. 加载 tokenizer（一般用 base 的）
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # 2. 加载 base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # 3. 加载 LoRA
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map=device_map
    )

    # 4. 合并 LoRA 并卸载 adapter
    model = model.merge_and_unload()

    # 5. 保存合并后的模型
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print(f"✅ LoRA merged model saved to: {output_path}")


if __name__ == "__main__":
    main()


# from safetensors.torch import load_file, save_file

# # 原始 safetensors 文件路径
# input_file = "/workspace/pangyunhe/project/crossnd/llm/output/kddcup/gen_psl_v2_turn_v3/global_step_140/adapter_model.safetensors"

# # 目标输出文件
# output_file = "/workspace/pangyunhe/project/crossnd/llm/output/kddcup/gen_psl_v2_turn_v3/global_step_140/filtered_lora.safetensors"

# # 要排除的 key（可以用 startswith 过滤）
# exclude_prefixes = (
#     "base_model.model.lm_head",
#     "base_model.model.model.embed_tokens",
# )

# # 1️⃣ 加载 safetensors
# state_dict = load_file(input_file)

# # 2️⃣ 过滤掉指定 key
# filtered_state_dict = {
#     k: v
#     for k, v in state_dict.items()
#     if not k.startswith(exclude_prefixes)
# }
# breakpoint()
# # 3️⃣ 保存为新的 safetensors
# save_file(filtered_state_dict, output_file)

# print(f"Filtered safetensors saved to {output_file}, kept {len(filtered_state_dict)} tensors")


# CUDA_VISIBLE_DEVICES=6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/project/crossnd/llm/lora --tensor_parallel_size 2 --batch_size 32  --save_dir outputs/multiturn_grpo_v5/eval.txt
