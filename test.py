# from transformers import AutoModel, AutoTokenizer
# from transformers import Qwen3ForCausalLM

# import torch
# model_path = "/home/zhipuai/zhangfanjin-15T/pyh/models/models/Qwen/Qwen3-4B"
# device = torch.device('cuda:0')
# model = Qwen3ForCausalLM.from_pretrained("/home/zhipuai/zhangfanjin-15T/pyh/models/models/Qwen/Qwen3-4B",torch_dtype=torch.bfloat16,  trust_remote_code=True,attn_implementation="flash_attention_2").to(device)

# breakpoint()

# #deepspeed   test.py 

from zhipuai import ZhipuAI
client = ZhipuAI(api_key="e310028bdb02414e8b0514217917aab1.OLfjktT6m1jCseiU") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4-plus",  # 填写需要调用的模型编码
    messages=[
        {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
        {"role": "user", "content": "农夫需要把狼、羊和白菜都带过河，但每次只能带一样物品，而且狼和羊不能单独相处，羊和白菜也不能单独相处，问农夫该如何过河。"}
    ],
)
print(response.choices[0].message)