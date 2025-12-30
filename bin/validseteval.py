import pickle
import json

data = json.load(open('/workspace/pangyunhe/project/crossnd/data/kddcup/eval_na_checking_triplets_valid.json','r'))
label_dict = {}
for i in data:
    aid1 = i['aid1']
    pid = i['pid'].split('-')[0]
    label_dict[f'{aid1}-{pid}'] = i['label']

pred = pickle.load(open('/workspace/pangyunhe/project/crossnd/llm/output/Qwen8B/cls-hard-outer-hybrid/res/eval.pkl','rb'))


all_metadata =pred['all_metadata']
def flatten_metadata(meta):
    result = []
    if not isinstance(meta, list):
        # 如果是 dict，直接保存
        result.append(meta)
    elif isinstance(meta, list):
        # 如果是 list，递归处理每个元素
        for item in meta:
            result.extend(flatten_metadata(item))
    # 其他类型忽略
    return result
metadata = flatten_metadata(all_metadata)
pred_dict = {}
for i in metadata:
    aid1 = i['aid1']
    pid = i['pid'].split('-')[0]
    pred_dict[f'{aid1}-{pid}'] = i['label']

label_set = set(label_dict.keys())

pred_set = set(pred_dict.keys())
for i in label_dict.keys():
    if i not in pred_set:
        print(i)
breakpoint()