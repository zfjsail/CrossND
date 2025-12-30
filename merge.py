import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====== 配置 ======
base_model_path = "/path/to/base_model"
lora_model_path = "/path/to/lora_adapter"
output_path = "/path/to/merged_model"

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