#!/usr/bin/env python
# coding=utf-8

import json
import os
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import random



data_dir = "/home/zhipuai/zhangfanjin-15T/pyh/pangyunhe1/git/crossnd-202211/data/kddcup"

paper_path = os.path.join(data_dir,"paper_dict_mag.json")
in_name2pid_path = os.path.join(data_dir,'name_aid_to_pids_in.json')
out_name2pid_path = os.path.join(data_dir,'name_aid_to_pids_out.json')
sims_path = os.path.join(data_dir,"kddcup_cmp_res_cps.json")



# 系统提示模板
system_prompt = """
你要进行一个学者论文检测的任务, 每个作者有两个源可以使用, 一个是内部源, 另一个是外部源, 使用外部源来支持内部源论文的错误分配检测,每个源是由论文的集合组成的, 每篇论文包题目,作者,机构信息, 现在需要基于两个源进行异常分配的论文检测,即检测给定论文是否应该属于该作者,不属于该作者的论文是异常论文, 现在有一批论文, 你需要根据内部源和外部源的论文与该论文的相似性,来判断该论文是否属于内部源
你的返回应该是"0"或者"1",其中"0"表示异常数据,"1"表示正常数据,不包含其他内容:
"""

system_prompt_v8 = """
你要进行一个学者论文检测的任务, 每个作者有两个源可以使用, 一个是内部源, 另一个是外部源,使用外部源来支持内部源论文的错误分配检测,每个源是由论文的集合组成的, 同时每个源都有可能存在一定的噪声,同时外部源不一定是当前的目标学者, 而可能是与其同名的学者,两个源之间的相似性可以一定程度上反映这一问题, 每篇论文包题目,作者,机构信息, 现在需要基于两个源进行异常分配的论文检测,即检测给定论文是否应该属于该作者,不属于该作者的论文是异常论文, 现在有一批论文, 你需要根据内部源和外部源的论文与该论文的相似性(在0,1之间),来判断该论文是否属于内部源,其判断的依据主要考虑以下几个方面:
1.对于内部源相似性比较高的情况,表明一致性较高,那么根据目标论文和内部或外部论文的相似性来判断该论文是否应该属于该作者
2.对于内部源相似性较低的情况,表明一致性较低,说明内部或者外部论文可能存在错误分配,则如果目标论文和外部源相似性较高, 则说明外部源是正确的,目标论文可能是异常,如果目标论文和内部源相似性较高,则说明内部源是正确的,目标论文可能是异常
3.相似性是根据论文间题目所属的领域,作者和机构之间的相匹配程度三个特征来计算的,需要考虑三方面特征
4.不能根据该论文存在于内部源或者外部源,就判断该论文是否属于该作者,因为要预测的论文就是从其中采样出来的
5.不能简单地根据目标学者名字属于内部源或者外部源,就判断该论文是否属于该作者,因为目标其原本论文分配错误的原因就是因为学者同名而错误分配的
6.可能存在一些属性缺失的情况,但是这不应该成为判断是正确还是异常的标准
7.首先你要分别归纳一下内部源和外部源的内容, 判断其领域信息, 作者和机构等信息, 随后逐条目标论文来分析, 最终得到最后的结果, 这些只应该出现在思考中
8.因为内部源或者外部源中可能会存在一些噪声的论文, 所以你不应该只凭借目标论文与内部或外部源的一篇论文得出最终结果

你的返回应该按照 "id":"label" 的形式返回json数据,你要返回一个0和1之间的浮点数,用以表示其是否属于该作者,值越接近1则表明该论文越属于该作者,值越接近0则表明该论文不属于该作者,json不包含其他内容:
"""

global_prompt = """
目标学者名字: {name} 
内部源: {inner} 
外部源: {outer} 
两个源之间的相似性是: {similarity} 
下面是需要异常检测的论文: {targets} 
""" 

multi_turn_prompt = """
第 {index} 篇: {paper} 
"""

def fetch_single_paper_input(paper):
    text = ""
    
    text += f"Title: {paper['title']}\n"
    authors = [i.get('OriginalAuthor','') for i in paper['authors']]
    organizations = [i.get('OriginalAffiliation','') for i in paper['authors']]
    if len(authors)>10:
        authors = authors[:8]+ authors[-2:]
        organizations = organizations[:8]+ organizations[-2:]
    
    # 首先处理每个组织名称
    processed_orgs = []
    for org in organizations:
        processed_org = ' '.join(org.split(' ')[:10])  # 只取前10个单词
        processed_orgs.append(processed_org)
    
    # 然后去除重复项
    organizations = list(set(processed_orgs))
    
    text += f"Authors: {', '.join(authors)}\n"
    text += f"Organizations: {', '.join(organizations)}\n"

    return text 

def select_related_papers(target_paper, papers, type = "random"):
    #两种方式，一种是根据年份选出最接近的20个， 另一种是随机抽取20个
    if type == "random":
        return random.sample(papers,20) if len(papers)>20 else papers
    elif type == "year":
        target_year = target_paper.get("year",0)
        if target_year == 0:
            return random.sample(papers,20) if len(papers)>20 else papers
        else:
            # 计算每篇论文与目标论文年份的差值
            year_diff = [(i, abs(i.get('year', 0) - target_year)) for i in papers]
            # 按照年份差值排序
            sorted_papers = [paper for paper, _ in sorted(year_diff, key=lambda x: x[1])]
            # 返回年份最接近的20篇论文
            return sorted_papers[:20] if len(sorted_papers) > 20 else sorted_papers

def build_multiturn_input(inner_papers, outer_papers, similarity, target_papers, name, slct_type= "random"):
    inner = "\n".join([fetch_single_paper_input(paper) for paper in (random.sample(inner_papers,20) if len(inner_papers)>20 else inner_papers)])
    outer = "\n".join([fetch_single_paper_input(paper) for paper in (random.sample(outer_papers,20) if len(outer_papers)>20 else outer_papers) ])
    multi_turn_input = '; '.join([multi_turn_prompt.format(index = index, paper = fetch_single_paper_input(p)) for index,p in enumerate(target_papers)])
    return global_prompt.format(name=name, inner=inner, outer=outer, similarity=similarity,targets=multi_turn_input)

def process_all(input_path, output_path):
    """
    一次性处理所有步骤，减少中间文件的读写操作
    """
    print(f'开始处理数据：{input_path} -> {output_path}')
    
    # 固定路径


    # 加载数据
    print("加载所需数据...")
    with open(in_name2pid_path, 'r', encoding='utf-8') as f:
        in_name2pid = json.load(f)
    with open(out_name2pid_path, 'r', encoding='utf-8') as f:
        out_name2pid = json.load(f)
    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_data = json.load(f)
    with open(sims_path, 'r', encoding='utf-8') as f:
        sims = json.load(f)
    
    print(f'{len(test_data)} 总数据量')
    test_data = [i for i in test_data if i['author_sim']!=0]
    print(f'{len(test_data)} 删除 author_sim=0 后的数据量')
    
    # 步骤1：构建数据
    print("\n=== 步骤1：构建数据 ===")
    
    # 预处理数据
    data = {}
    for raw in test_data:
        pid, aid1, aid2, name, label = raw['pid'], raw['aid1'], raw['aid2'], raw['name'], raw['label']
        k = f'{aid1}-{aid2}'
        if name not in data:
            data[name] = {}
        
        if k not in data[name]:
            data[name][k] = []
        data[name][k].append(raw)
    
    processed_batch_data = []
    batch_size = 1  # 设置批处理大小
    for _, author in data.items():
        for k, v in author.items():
            random.shuffle(v)
            for batch in range(0, len(v), batch_size):
                processed_batch_data.append(v[batch:batch+batch_size])

    # 构建提示词数据
    data_batches = []
    def fetch_paper_id(src_id):
        return src_id.split('-')[0]

    for batch in tqdm(processed_batch_data, desc="构建数据批次"):
        aid1, aid2, name = batch[0]['aid1'], batch[0]['aid2'], batch[0]['name']
        batch_pid = [fetch_paper_id(i['pid']) for i in batch]
        
        batch_label = [i['label'] for i in batch]
        batch_papers = [paper_data[pid] for pid in batch_pid]
        
        # 随机打乱论文和标签
        combined = list(zip(batch_papers, batch_label))
        random.shuffle(combined)
        batch_papers, batch_label = zip(*combined)

        inner_papers = [paper_data[fetch_paper_id(i)] for i in in_name2pid[name][aid1]]
        outer_papers = [paper_data[fetch_paper_id(i)] for i in out_name2pid[name][aid2]]

        similarity = 0
        if sims.get(name, 0):
            if sims[name].get(aid1, 0):
                if sims[name][aid1].get(aid2, 0):
                    similarity = sims[name][aid1][aid2]

        batch_input = build_multiturn_input(
            inner_papers=inner_papers,
            outer_papers=outer_papers,
            similarity=similarity,
            name=name,
            target_papers=batch_papers
        )

        data_batches.append({
            'system_input': system_prompt,
            'input': batch_input,
            'label': batch_label,
            'src': batch
        })
    
    # 步骤2：平衡数据
    print("\n=== 步骤2：平衡数据 ===")
    
    positive = []
    negative = []
    for itm in data_batches:
        label = itm['label'][0]

        if label == 1:
            positive.append(itm)
        elif label == 0:
            negative.append(itm)
        else:
            raise NotImplementedError

    print(f"原始数据集: 正样本数量: {len(positive)}, 负样本数量: {len(negative)}")

    # 平衡正负样本
    if len(positive) > len(negative):
        # 计算需要重复采样的次数
        repeat_times = len(positive) // len(negative)
        remainder = len(positive) % len(negative)
        
        # 完整重复采样
        balanced_neg = negative * repeat_times
        
        # 处理余数部分
        if remainder > 0:
            balanced_neg += random.sample(negative, remainder)
        
        # 确保负样本数量与正样本数量相同
        assert len(balanced_neg) == len(positive)
        
        balanced_data = positive + balanced_neg
    # 如果负样本比正样本多，则从负样本中随机选择与正样本相同数量
    elif len(negative) > len(positive):
        balanced_neg = random.sample(negative, len(positive))
        balanced_data = positive + balanced_neg
    # 如果已经平衡，则直接使用
    else:
        balanced_data = positive + negative

    print(f"平衡后的数据集大小: {len(balanced_data)}")
    print(f"正样本数量: {len(positive)}, 负样本数量: {len(balanced_data) - len(positive)}")

    # 随机打乱数据
    random.shuffle(balanced_data)
    
    # 步骤3：提取和构建
    print("\n=== 步骤3：提取和构建训练数据 ===")
    
    # strfy result, only for 1 target paper per time
    for d in balanced_data:
        label = d['label']
        d['results'] = str(label[0])

    # transfer to system & user & assistant format
    sft_format = []
    for d in balanced_data:
        system = d['system_input']
        prompt = d['input']
        completion = d['results']
        
        sft_format.append({
            "id": d["src"],
            "messages": [
                {
                    "role": "system", 
                    "content": system
                },
                {
                    "role": "user", 
                    "content": prompt
                },
                {
                    "role": "assistant", 
                    "content": completion
                },            
            ]
        })
    
    # 步骤4：转换数据格式
    print("\n=== 步骤4：转换数据格式 ===")
    
    final_data = []
    for itm in sft_format:
        input_text = itm['messages'][0]["content"] + itm['messages'][1]["content"]
        label = int(itm["messages"][2]["content"])
        cur_itm = {
            "text": input_text,
            "label": label,
            "id": itm['id'][0]
        }
        final_data.append(cur_itm)

    # 保存最终结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n所有处理已完成，最终输出: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据处理流程')
    parser.add_argument('--input_path', type=str, required=True, help='输入数据路径')
    parser.add_argument('--output_path', type=str, default='./output/final_data.json', help='输出文件路径')
    
    args = parser.parse_args()
    
    process_all(args.input_path, args.output_path) 