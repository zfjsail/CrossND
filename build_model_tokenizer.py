from transformers import  AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import torch
from typing import Dict
import transformers
from argparse import ArgumentParser
# python build_model_tokenizer.py --lora_path /workspace/pangyunhe/project/crossnd/llm/output/kddcup/gen_psl_v2_turn_v3/checkpoint-300 --output_dir /workspace/pangyunhe/models/custom/qwen3-8b-multiturn
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="/workspace/pangyunhe/models/Qwen/Qwen3-8B")
parser.add_argument("--lora_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default= "/workspace/pangyunhe/models/custom/qwen3-8b-multiturn")
args = parser.parse_args()

from utils import (
    DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN,
    special_token_dict
)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    From https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 加载模型和tokenizer
model_path = args.model_path
lora_path = args.lora_path

print(f"Loading base model from {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

print(f"Loading tokenizer from {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 调整tokenizer和embedding大小（与train.py保持一致）
if tokenizer.pad_token is None:
    special_token_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_token_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_token_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_token_dict["unk_token"] = DEFAULT_UNK_TOKEN

print(f"Tokenizer vocab size before resizing: {len(tokenizer)}")
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_token_dict,
    tokenizer=tokenizer,
    model=model,
)
print(f"Tokenizer vocab size after resizing: {len(tokenizer)}")

# 现在加载PEFT adapter
print(f"Loading PEFT adapter from {lora_path}")
peft_model = PeftModel.from_pretrained(model, lora_path)

# 合并adapter到基础模型
print("Merging adapter into base model")
merged_model = peft_model.merge_and_unload()

# 保存合并后的模型和tokenizer
save_path = args.output_dir
print(f"Saving merged model to {save_path}")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Done!")

