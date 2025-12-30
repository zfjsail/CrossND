# # from transformers import AutoModel, AutoTokenizer
# # from transformers import Qwen3ForCausalLM

# # import torch
# # model_path = "/home/zhipuai/zhangfanjin-15T/pyh/models/models/Qwen/Qwen3-4B"
# # device = torch.device('cuda:0')
# # model = Qwen3ForCausalLM.from_pretrained("/home/zhipuai/zhangfanjin-15T/pyh/models/models/Qwen/Qwen3-4B",torch_dtype=torch.bfloat16,  trust_remote_code=True,attn_implementation="flash_attention_2").to(device)

# # breakpoint()

# # #deepspeed   test.py 

# # from zhipuai import ZhipuAI
# # client = ZhipuAI(api_key="e310028bdb02414e8b0514217917aab1.OLfjktT6m1jCseiU") # 填写您自己的APIKey
# # response = client.chat.completions.create(
# #     model="glm-4-plus",  # 填写需要调用的模型编码
# #     messages=[
# #         {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
# #         {"role": "user", "content": "农夫需要把狼、羊和白菜都带过河，但每次只能带一样物品，而且狼和羊不能单独相处，羊和白菜也不能单独相处，问农夫该如何过河。"}
# #     ],
# # )
# # print(response.choices[0].message)
# import torch
# import pickle
# data = pickle.load(open('/workspace/pangyunhe/project/crossnd/llm/res.pkl','rb'))
# from collections import defaultdict
# from sklearn.metrics import roc_auc_score, average_precision_score

# def compute_metrics(predictions, labels, metadata):
#     """
#     计算评估指标，使用metadata和预测结果
    
#     Args:
#         eval_preds: 包含预测结果、标签和metadata的CrossNDEvalPrediction对象
        
#     Returns:
#         metrics: 包含各项评估指标的字典
#     """
#     def flatten_metadata(meta):
#         result = []
#         if not isinstance(meta, list):
#             # 如果是 dict，直接保存
#             result.append(meta)
#         elif isinstance(meta, list):
#             # 如果是 list，递归处理每个元素
#             for item in meta:
#                 result.extend(flatten_metadata(item))
#         # 其他类型忽略
#         return result
#     predictions = torch.cat(flatten_metadata(predictions)).tolist()
#     labels = torch.cat(flatten_metadata(labels)).tolist()
#     metadata = flatten_metadata(metadata)
#     author_data = defaultdict(lambda: {'preds': [], 'labels': []})
#     # 处理嵌套结构的预测结果、标签和元数据
#     for pred, label,meta in zip(predictions, labels, metadata):

#         aid = meta["aid1"]
#         # 确保pred是标量
#         # if type(probs) == list:
#         #     author_data[aid]['preds'].extend(probs)
#         #     author_data[aid]['labels'].extend(label)
#         # else:
#         author_data[aid]['preds'].append(pred)
#         author_data[aid]['labels'].append(label)

#     # 计算宏平均AUC和MAP
#     maps = []
#     aucs = []
    
#     print(f"开始按作者计算指标，共有 {len(author_data)} 个作者")
    
#     for author_id, data in author_data.items():
#         probs = data['preds']
#         labels = data['labels']
          
#         # 注意：原始标签中1表示正样本，0表示负样本
#         # 根据eval.py的处理方式调整标签和预测值

#         # 计算正样本比例

#         pos_ratio = (sum(labels) / len(labels))
#         # 跳过正样本比例≥50%或全为负样本的作者
#         if pos_ratio == 1 or pos_ratio < 0.5:
#             continue

#         adjusted_probs = [1-p for p in probs]
#         adjusted_labels = [1-l for l in labels]   



#         author_ap = average_precision_score(adjusted_labels, adjusted_probs)
#         author_auc = roc_auc_score(adjusted_labels, adjusted_probs)
        
#         maps.append(author_ap)
#         aucs.append(author_auc)
#         n_authors += 1

#     print(f"完成评估的有效作者数量: {n_authors}")
#     # 计算最终宏平均
#     final_map = sum(maps) / len(maps)
#     final_auc = sum(aucs) / len(aucs)
    
#     return {
#         'MAP': float(final_map),
#         'AUC': float(final_auc),
#         'n_authors': n_authors
#     }


from safetensors.torch import load_file

state_dict = load_file("/workspace/pangyunhe/project/crossnd/llm/output/kddcup/gen_psl_v2/checkpoint-240/adapter_model.safetensors")

for key in state_dict.keys():
    if "lm_head" in key:
        print(f"{key}: {state_dict[key].shape}")

breakpoint()