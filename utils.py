#!/usr/bin/env python
# coding=utf-8

import json
import os
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union, Tuple
import copy
from collections import defaultdict
from os.path import join
LABEL_TOKEN = '<label_token>'

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# TO BE ADDED
END_OF_TEXT = '<eot>'
END_OF_GRAPH = '<eog>'
END_OF_EMB = '<eoe>'
TRAINABLE_SPECIAL_TOKENS = [LABEL_TOKEN]
special_token_dict = {'additional_special_tokens': TRAINABLE_SPECIAL_TOKENS}


def paper_overlap_ratio(pids1, pids2):
    common_pids = set(pids1) & set(pids2)
    n_common_pubs = len(common_pids)
    pubs_overlap_a = n_common_pubs / len(pids1)
    pubs_overlap_m = n_common_pubs / len(pids2)
    return min(pubs_overlap_a, pubs_overlap_m), max(pubs_overlap_a, pubs_overlap_m)


def build_turn_data(papers, num_turn, pred_scores=None, strategy='random'):
    """
    将论文列表按 num_turn 分批，并根据策略对每批内部进行排序。
    
    Args:
        papers: 同一个 aid1-aid2 组合下的论文列表
        num_turn: 每批的大小
        pred_scores: pid -> pred_score 的映射字典，None 时退化为随机
        strategy: 排序策略
            ===== 全局排序后分批 =====
            - 'random': 随机打乱后分批（默认）
            - 'desc': 全局按分数降序后分批
            - 'asc': 全局按分数升序后分批
            - 'confidence': 全局按置信度(|score-0.5|)降序后分批
            
            ===== turn 粒度排序 =====
            - 'turn_desc': 每批内部按分数降序
            - 'turn_asc': 每批内部按分数升序
            - 'turn_confidence': 每批内部按置信度降序（距0.5远的在前）
            - 'turn_confidence_first': 每批首位放高置信度样本，其余随机
            - 'turn_interleave': 类希尔排序，高低分交替排列，各批分数分布均匀
            - 'turn_interleave_confidence_first': 类希尔排序+置信度，按置信度高低交替排列，各批置信度分布均匀
            - 'turn_interleave_v2': 类希尔排序v2，每批内高中低分三段交替
            - 'turn_interleave_reverse': 类希尔排序反向，低高分交替（先低后高）
            - 'turn_interleave_confidence_v2': 类希尔排序+置信度v2，每批内高中低置信度三段交替
            - 'turn_shell_sort': 真·希尔排序，使用希尔增量序列分组交替
            - 'turn_balanced': 均匀分配，使每批的平均分数尽量接近（类似发牌）
            - 'turn_zigzag': 锯齿形分配，奇数批降序、偶数批升序
            - 'turn_confidence_desc': 均匀分配后，每批内部按置信度降序排列
    
    Returns:
        List[List[dict]]: 分批后的论文列表，每个元素是一个 turn 的论文列表
    """
    n = len(papers)
    num_batches = (n + num_turn - 1) // num_turn
    
    def get_score(p):
        return pred_scores.get(p['pid'], 0.5) if pred_scores else 0.5
    
    def get_confidence(p):
        return abs(get_score(p) - 0.5)
    
    def simple_split(sorted_papers):
        """将已排序的列表按 num_turn 分批"""
        return [sorted_papers[i:i+num_turn] for i in range(0, n, num_turn)]
    
    # ===== 全局排序后分批 =====
    if strategy == 'random' or pred_scores is None:
        v_ = copy.deepcopy(papers)
        random.shuffle(v_)
        return simple_split(v_)
    
    elif strategy == 'desc':
        # 全局降序后分批
        v_ = sorted(papers, key=get_score, reverse=True)
        return simple_split(v_)
    
    elif strategy == 'asc':
        # 全局升序后分批
        v_ = sorted(papers, key=get_score, reverse=False)
        return simple_split(v_)
    
    elif strategy == 'confidence':
        # 全局按置信度降序后分批
        v_ = sorted(papers, key=get_confidence, reverse=True)
        return simple_split(v_)
    
    # ===== turn 粒度排序：先随机分批，再对每批内部排序 =====
    elif strategy == 'turn_desc':
        # 先随机分批，然后每批内部按分数降序
        v_ = copy.deepcopy(papers)
        random.shuffle(v_)
        batches = simple_split(v_)
        return [sorted(batch, key=get_score, reverse=True) for batch in batches]
    
    elif strategy == 'turn_asc':
        # 先随机分批，然后每批内部按分数升序
        v_ = copy.deepcopy(papers)
        random.shuffle(v_)
        batches = simple_split(v_)
        return [sorted(batch, key=get_score, reverse=False) for batch in batches]
    
    elif strategy == 'turn_confidence':
        # 先随机分批，然后每批内部按置信度降序
        v_ = copy.deepcopy(papers)
        random.shuffle(v_)
        batches = simple_split(v_)
        return [sorted(batch, key=get_confidence, reverse=True) for batch in batches]
    
    elif strategy == 'turn_confidence_first':
        # 每批首位放高置信度样本，其余随机
        # 例：50篇，num_turn=10 → 5批 → 取top5高置信度样本各放一批首位
        v_ = sorted(papers, key=get_confidence, reverse=True)
        anchors = v_[:num_batches]       # 前 num_batches 个高置信度样本
        rest = v_[num_batches:]           # 剩余样本
        random.shuffle(rest)              # 剩余样本随机打乱
        
        batches = []
        rest_idx = 0
        for b in range(num_batches):
            batch = [anchors[b]]          # 首位是高置信度样本
            fill_count = min(num_turn - 1, n - num_batches - rest_idx)
            batch.extend(rest[rest_idx:rest_idx + fill_count])
            rest_idx += fill_count
            batches.append(batch)
        return batches
    
    elif strategy == 'turn_interleave':
        # 类希尔排序：先蛇形发牌使各批均匀，再每批内高低分交替排列
        # 步骤1: 蛇形发牌（同 turn_balanced）
        v_ = sorted(papers, key=get_score, reverse=True)
        batches = [[] for _ in range(num_batches)]
        for idx, paper in enumerate(v_):
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        # 步骤2: 每批内部高低分交替排列
        result = []
        for batch in batches:
            if len(batch) <= 2:
                result.append(batch)
                continue
            batch_sorted = sorted(batch, key=get_score, reverse=True)
            interleaved = []
            left, right = 0, len(batch_sorted) - 1
            while left <= right:
                interleaved.append(batch_sorted[left])
                if left != right:
                    interleaved.append(batch_sorted[right])
                left += 1
                right -= 1
            result.append(interleaved)
        return result
    
    elif strategy == 'turn_interleave_confidence_first':
        # 类希尔排序+置信度：先按置信度蛇形发牌使各批均匀，再每批内高低置信度交替排列
        # 步骤1: 按置信度降序排序后蛇形发牌
        v_ = sorted(papers, key=get_confidence, reverse=True)
        batches = [[] for _ in range(num_batches)]
        for idx, paper in enumerate(v_):
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        # 步骤2: 每批内部按置信度高低交替排列
        result = []
        for batch in batches:
            if len(batch) <= 2:
                result.append(batch)
                continue
            batch_sorted = sorted(batch, key=get_confidence, reverse=True)
            interleaved = []
            left, right = 0, len(batch_sorted) - 1
            while left <= right:
                interleaved.append(batch_sorted[left])  # 高置信度
                if left != right:
                    interleaved.append(batch_sorted[right])  # 低置信度
                left += 1
                right -= 1
            result.append(interleaved)
        return result
    
    elif strategy == 'turn_interleave_v2':
        # 类希尔排序v2：三段交替（高-低-中）
        # 步骤1: 蛇形发牌
        v_ = sorted(papers, key=get_score, reverse=True)
        batches = [[] for _ in range(num_batches)]
        for idx, paper in enumerate(v_):
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        # 步骤2: 每批内部按高-低-中三段交替
        result = []
        for batch in batches:
            if len(batch) <= 3:
                result.append(batch)
                continue
            batch_sorted = sorted(batch, key=get_score, reverse=True)
            n = len(batch_sorted)
            high = batch_sorted[:n//3]
            mid = batch_sorted[n//3:2*n//3]
            low = batch_sorted[2*n//3:]
            
            interleaved = []
            for i in range(max(len(high), len(mid), len(low))):
                if i < len(high):
                    interleaved.append(high[i])
                if i < len(low):
                    interleaved.append(low[i])
                if i < len(mid):
                    interleaved.append(mid[i])
            result.append(interleaved)
        return result
    
    elif strategy == 'turn_interleave_reverse':
        # 类希尔排序反向：低高分交替（先低后高）
        # 步骤1: 蛇形发牌
        v_ = sorted(papers, key=get_score, reverse=True)
        batches = [[] for _ in range(num_batches)]
        for idx, paper in enumerate(v_):
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        # 步骤2: 每批内部低高分交替排列（反向）
        result = []
        for batch in batches:
            if len(batch) <= 2:
                result.append(batch)
                continue
            batch_sorted = sorted(batch, key=get_score, reverse=True)
            interleaved = []
            left, right = 0, len(batch_sorted) - 1
            while left <= right:
                interleaved.append(batch_sorted[right])  # 先低分
                if left != right:
                    interleaved.append(batch_sorted[left])  # 后高分
                left += 1
                right -= 1
            result.append(interleaved)
        return result
    
    elif strategy == 'turn_interleave_confidence_v2':
        # 类希尔排序+置信度v2：三段交替（高-低-中置信度）
        # 步骤1: 按置信度蛇形发牌
        v_ = sorted(papers, key=get_confidence, reverse=True)
        batches = [[] for _ in range(num_batches)]
        for idx, paper in enumerate(v_):
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        # 步骤2: 每批内部按高-低-中置信度三段交替
        result = []
        for batch in batches:
            if len(batch) <= 3:
                result.append(batch)
                continue
            batch_sorted = sorted(batch, key=get_confidence, reverse=True)
            n = len(batch_sorted)
            high = batch_sorted[:n//3]
            mid = batch_sorted[n//3:2*n//3]
            low = batch_sorted[2*n//3:]
            
            interleaved = []
            for i in range(max(len(high), len(mid), len(low))):
                if i < len(high):
                    interleaved.append(high[i])
                if i < len(low):
                    interleaved.append(low[i])
                if i < len(mid):
                    interleaved.append(mid[i])
            result.append(interleaved)
        return result
    
    elif strategy == 'turn_shell_sort':
        # 真·希尔排序：使用希尔增量序列
        # 步骤1: 按分数排序
        v_ = sorted(papers, key=get_score, reverse=True)
        
        # 步骤2: 使用希尔增量序列进行分组
        # 增量序列: n/2, n/4, n/8, ..., 1
        gap = num_turn // 2
        if gap < 1:
            gap = 1
        
        batches = [[] for _ in range(num_batches)]
        
        # 按增量分组交替放置
        for idx, paper in enumerate(v_):
            # 使用希尔增量计算批次索引
            batch_idx = (idx * gap) % num_batches
            batches[batch_idx].append(paper)
        
        # 每批内部保持原有顺序（已按分数排序）
        return batches
    
    elif strategy == 'turn_balanced':
        # 均匀分配（发牌策略）：按分数排序后，轮流发给各批
        # 使得每批的平均分数尽量接近
        # 例：排序后 [s1, s2, ..., s50]（降序）
        #   s1→batch0, s2→batch1, ..., s5→batch4
        #   s6→batch4, s7→batch3, ..., s10→batch0  （蛇形发牌）
        #   s11→batch0, s12→batch1, ...
        v_ = sorted(papers, key=get_score, reverse=True)
        batches = [[] for _ in range(num_batches)]
        
        for idx, paper in enumerate(v_):
            # 蛇形发牌：第0轮 0,1,2,...,k-1; 第1轮 k-1,...,1,0; 第2轮 0,1,...
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        # 每批内部按分数降序排列
        return [sorted(batch, key=get_score, reverse=True) for batch in batches]
    
    elif strategy == 'turn_zigzag':
        # 锯齿形：奇数批内部降序，偶数批内部升序
        # 先均匀分配（发牌），再对奇偶批做不同方向排序
        v_ = sorted(papers, key=get_score, reverse=True)
        batches = [[] for _ in range(num_batches)]
        
        for idx, paper in enumerate(v_):
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        result = []
        for b_idx, batch in enumerate(batches):
            if b_idx % 2 == 0:
                result.append(sorted(batch, key=get_score, reverse=True))   # 降序
            else:
                result.append(sorted(batch, key=get_score, reverse=False))  # 升序
        return result
    
    elif strategy == 'turn_confidence_desc':
        # 先均匀分配（发牌），然后每批内部按置信度降序排列
        v_ = sorted(papers, key=get_confidence, reverse=True)
        batches = [[] for _ in range(num_batches)]
        
        for idx, paper in enumerate(v_):
            cycle = idx // num_batches
            pos_in_cycle = idx % num_batches
            if cycle % 2 == 0:
                batch_idx = pos_in_cycle
            else:
                batch_idx = num_batches - 1 - pos_in_cycle
            batches[batch_idx].append(paper)
        
        # 每批内部按置信度降序排列
        return [sorted(batch, key=get_confidence, reverse=True) for batch in batches]
    
    else:
        print(f"Warning: Unknown TTS strategy '{strategy}', using random shuffle")
        v_ = copy.deepcopy(papers)
        random.shuffle(v_)
        return simple_split(v_)


def add_author_overlap(data):
    all_data = []
    name_aid_to_pids_in = json.load(
        open("whoiswho_data/name_aid_to_pids_in.json"))
    name_aid_to_pids_out = json.load(
        open("whoiswho_data/name_aid_to_pids_out.json"))
    mag_name_pid_to_aid = defaultdict(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                mag_name_pid_to_aid[name][pid] = aid

    for item in data:
        pid = item['pid']
        aid = item['aid1']
        name = item['name']
        pids1 = name_aid_to_pids_in[name].get(aid, [])
        # if len(pids1) < 5:
        #     continue
        try:
            aid_map = mag_name_pid_to_aid[name][pid]
            pids_m = name_aid_to_pids_out[name][aid_map]
            # if len(pids_m) < 5:
            #     continue
        except:
            continue
        cur_sim, cur_sim_max = paper_overlap_ratio(pids1, pids_m)
        item['author_sim'] = cur_sim
        item['author_sim_max'] = cur_sim_max
        all_data.append(item)
    return all_data


class CrossNDDataset(Dataset):
    """学者论文异常检测数据集"""

    def __init__(self,
                 data_dir,
                 tokenizer,
                 model_args,
                 data_args,
                 num_turn,
                 mode="train",
                 ):

        self.use_label_token = model_args.use_label_token
        # self.label_type = model_args.label_type
        self.num_turn = num_turn
        self.apply_chat_template = data_args.apply_chat_template
        self.mode = mode.lower()  # 确保模式字符串为小写
        self.use_outer = model_args.use_outer
        self.model_args = model_args
        self.data_args = data_args
        self.max_seq_length = model_args.max_seq_length
        # 保存原始的 all_data 用于重新构建数据集时使用
        self.all_data_raw = None
        self.yes_token = "Yes"
        self.no_token = "No"
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS, self.LABEL_TOKEN_IDS = tokenizer.convert_tokens_to_ids(
            [self.yes_token, self.no_token, LABEL_TOKEN])
        
        # 读取多轮推理的预测分数（用于TTS优化）
        self.pred_scores = None
        self.tts_strategy = model_args.tts_strategy if hasattr(model_args, 'tts_strategy') else 'desc'
        if model_args.multiturn_path is not None and os.path.exists(model_args.multiturn_path):
            print(f"Loading prediction scores from {model_args.multiturn_path} for TTS optimization")
            print(f"TTS Strategy: {self.tts_strategy}")
            pred_data = json.load(open(model_args.multiturn_path))
            # 构建 pid -> pred_score 的映射
            self.pred_scores = {item['pid']: item['pred'] for item in pred_data}
            print(f"Loaded {len(self.pred_scores)} prediction scores")

        if data_args.dataset == 'whoiswho':
            self.in_name2pid = json.load(
                open(os.path.join(data_dir,"cleaned_name_aid_to_pids_in.json")))
            self.out_name2pid = json.load(
                open(os.path.join(data_dir,"cleaned_name_aid_to_pids_out.json")))
            self.paper_data = json.load(open(os.path.join(data_dir,"pub_dict.json")))

            def flatten_name_dict(data):
                temp_dict = {}
                for k, v in data.items():
                    for aid1, pids in v.items():
                        temp_dict[aid1] = pids
                return temp_dict
            self.in_name2pid = flatten_name_dict(self.in_name2pid)
            self.out_name2pid = flatten_name_dict(self.out_name2pid)

        self.data = []

        if self.use_outer:
            self.system_prompt = """你要进行一个学者论文检测的任务, 每个学者有两个源可以使用, 一个是内部源, 另一个是外部源, 使用外部源来支持内部源论文的错误分配检测,每个源是由论文的集合组成的, 每篇论文包题目,学者,机构信息, 现在需要基于两个源进行异常分配的论文检测,即检测给定论文是否应该属于该学者,不属于该学者的论文是异常论文, 现在有一批论文, 你需要根据内部源和外部源的论文以及两个源的相似性,来判断该论文是否属于内部源"""
            self.global_prompt = """目标学者名字: {name} 
            内部源: {inner} 
            外部源: {outer} 
            内部源和外部源的相似度是: {author_sim}
            """
            if self.apply_chat_template:
                self.user_prompt = """ {paper} 是该学者的论文吗? """
                self.assistant_prompt = """{label_token}"""
            else:
                self.multi_turn_prompt = """ {paper} 是该学者的论文。 {label_token}\n"""
        else:
            self.system_prompt = """你要进行一个学者论文检测的任务,每篇论文包题目,学者,机构信息,现在需要进行异常分配的论文检测,即检测给定论文是否应该属于该学者,不属于该学者的论文是异常论文,判断给定的论文是否属于该学者, 如果属于则返回 Yes ,否则返回 No。"""
            self.global_prompt = """目标学者名字: {name} 
            该学者的论文,可能存在错误分配论文: {inner} 
            """

            if self.apply_chat_template:
                self.user_prompt = """ {paper} 结果是什么? """
                self.assistant_prompt = """{label_token}"""
                self.multi_turn_prompt = """ {paper} 结果是 {label_token}\n"""
            else:
                self.multi_turn_prompt = """ 论文 {paper} 是该学者的论文。 {label_token}\n"""

        self.tokenizer = tokenizer
        if self.mode == "train":
            data = json.load(open(model_args.src))
            if self.data_args.dataset == "whoiswho" and 'author_sim' not in data:
                data = add_author_overlap(data)
            if self.model_args.label_thr is not None and self.mode == 'train':
                for i in data:
                    i['ori_label'] = i['label']
                    if i['soft_label'] >= self.model_args.label_thr:
                        i['label'] = 1
                    else:
                        i['label'] = 0
            pos = [i for i in data if i['label'] == 1]
            neg = [i for i in data if i['label'] == 0]
            neg = neg * (len(pos)//len(neg))  # 平衡正负样本
            all_data = pos + neg
            random.shuffle(all_data)
        elif self.mode == "eval":
            if self.data_args.dataset == "whoiswho":
                all_data = json.load(
                    open(join(data_dir,"eval_na_checking_triplets_valid.json")))
                if 'author_sim' not in all_data[0]:
                    all_data = add_author_overlap(all_data)
        else:
            if self.data_args.dataset == "whoiswho":
                all_data = json.load(
                    open(join(data_dir,"eval_na_checking_triplets_test.json")))
                if 'author_sim' not in all_data[0]:
                    all_data = add_author_overlap(all_data)

        # 保存原始 all_data 用于后续重新构建
        self.all_data_raw = copy.deepcopy(all_data)

        if self.num_turn == 1:
            random.shuffle(all_data)
            self.data = [[i] for i in all_data]
        else:
            data = []
            data_dd = defaultdict(list)
            for item in all_data:
                aid1 = item['aid1']
                aid2 = item.get('aid2', 'None')
                data_dd[f'{aid1}-{aid2}'].append(item)

            for v in data_dd.values():
                v_ = copy.deepcopy(v)
                if self.mode == "test" and self.pred_scores is not None:
                    batches = build_turn_data(v_, self.num_turn, self.pred_scores, self.tts_strategy)
                else:
                    batches = build_turn_data(v_, self.num_turn)
                data.extend(batches)
            self.data = data

    def rebuild_dataset(self, num_turn):
        """
        根据新的num_turn重新构建数据集

        Args:
            num_turn: 新的轮数参数
        """
        # 更新num_turn参数
        self.num_turn = num_turn
        print(f"rebuild_dataset num_turn: {num_turn}")
        # 使用保存的原始数据重新构建
        all_data = copy.deepcopy(self.all_data_raw)

        if num_turn > 1:
            data = []
            data_dd = defaultdict(list)
            if self.use_outer:
                for item in all_data:
                    aid1 = item['aid1']
                    aid2 = item['aid2']
                    data_dd[f'{aid1}-{aid2}'].append(item)
            else:
                for item in all_data:
                    aid1 = item['aid1']
                    data_dd[f'{aid1}'].append(item)

            for v in data_dd.values():
                v_ = copy.deepcopy(v)
                if self.mode == "test" and self.pred_scores is not None:
                    batches = build_turn_data(v_, self.num_turn, self.pred_scores, self.tts_strategy)
                else:
                    batches = build_turn_data(v_, self.num_turn)
                data.extend(batches)
            self.data = data
        else:
            random.shuffle(all_data)
            self.data = [[i] for i in all_data]

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        aid1, aid2, name = data[0]['aid1'], data[0].get(
            'aid2', None), data[0]['name']

        aid1 = str(aid1)
        aid2 = str(aid2) if aid2 is not None else None

        if 'similarity' in data[0]:
            similarity = data[0]['similarity']
        elif 'author_sim' in data[0]:
            similarity = data[0]['author_sim']
        else:
            similarity = None

        label_list = [i['label'] for i in data]
        labels = torch.tensor(label_list, dtype=torch.long)

        def fetch_paper_id(src_id):
            return src_id.split('-')[0]
        # 获取论文信息
        pids = [fetch_paper_id(i['pid']) for i in data]
        papers = [self.paper_data[i] for i in pids]

        papers_in = [i for i in self.in_name2pid[aid1] if i != data[0]['pid']]
        inner_papers = [self.paper_data[fetch_paper_id(i)] for i in papers_in]
        random.shuffle(inner_papers)

        # selected_inner_papers = self._select_related_papers(papers[0], inner_papers, type='random',num=self.model_args.paper_slct_num)
        selected_inner_papers = inner_papers[:self.model_args.paper_slct_num]
        inner = "\n".join([self._fetch_single_paper_input(paper)
                          for paper in selected_inner_papers])
        if self.use_outer and aid2 is not None:
            # 两种情况加入outer：1) similarity >= author_sim  2) author_sim_lower_bound非None且similarity < lower_bound
            should_include_outer = (similarity >= self.model_args.author_sim) or \
                                   (self.model_args.author_sim_lower_bound is not None and similarity <
                                    self.model_args.author_sim_lower_bound)
            if not self.model_args.hybrid_train or (self.model_args.hybrid_train and should_include_outer):
                papers_out = [i for i in self.out_name2pid[aid2]
                              if i != data[0]['pid']]
                outer_papers = [
                    self.paper_data[fetch_paper_id(i)] for i in papers_out]
                random.shuffle(outer_papers)
                # selected_outer_papers = self._select_related_papers(papers[0], outer_papers, type='random',num=self.model_args.paper_slct_num)
                selected_outer_papers = outer_papers[:
                                                     self.model_args.paper_slct_num]
                outer_papers = [self._fetch_single_paper_input(
                    paper) for paper in selected_outer_papers]
                pred_paper_inputs_len = len(self.tokenizer(' ; '.join(
                    [self._fetch_single_paper_input(paper) for paper in papers]))['input_ids'])

                if self.max_seq_length is None:
                    outer = "\n".join(outer_papers)
                else:
                    inner_token_len = len(self.tokenizer(inner)['input_ids'])
                    outer_token_len = self.max_seq_length - \
                        inner_token_len - pred_paper_inputs_len - 500
                    outer_curr_len = 0
                    num_outer_papers = 0
                    cuted_outer_papers = []
                    random.shuffle(outer_papers)
                    while outer_curr_len < outer_token_len and num_outer_papers < len(outer_papers):
                        cuted_outer_papers.append(
                            outer_papers[num_outer_papers])
                        outer_curr_len += len(self.tokenizer(
                            outer_papers[num_outer_papers])['input_ids'])
                        num_outer_papers += 1
                    outer = "\n".join(cuted_outer_papers)
                    # print(f"Original outer papers: {len(outer_papers)}, Cut outer papers: {len(cuted_outer_papers)}")
            else:
                outer = None
                similarity = None
        else:
            outer = None
            similarity = None

        # 构建模型输入
        if self.apply_chat_template:
            chat = []
            chat.append({"role": "system", "content": self.system_prompt})

            # 构建全局提示

            global_prompt = self.global_prompt.format(**{
                "name": name,
                "inner": inner,
                "outer": outer if outer is not None else 'none',
                "author_sim": round(similarity, 2) if similarity is not None else 'none'
            })

            # 第一轮对话
            first_paper_prompt = global_prompt + \
                self.user_prompt.format(
                    paper=self._fetch_single_paper_input(papers[0]))
            chat.append({"role": "user", "content": first_paper_prompt})

            chat.append({"role": "assistant", "content": self.assistant_prompt.format(
                label_token=LABEL_TOKEN)})

            # 如果有多轮对话，添加后续轮次
            for paper in papers[1:]:
                chat.append({"role": "user", "content": self.user_prompt.format(
                    paper=self._fetch_single_paper_input(paper))})
                chat.append({"role": "assistant", "content": self.assistant_prompt.format(
                    label_token=LABEL_TOKEN)})
            # 应用聊天模板
            inputs = self.tokenizer.apply_chat_template(
                chat, return_tensors="pt")
            # 确保attention_mask与input_ids形状一致
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
            # ### 将label_token的位置的attention_mask设置为0
            # label_token_id = self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN)
            # attention_mask[inputs['input_ids'] == label_token_id] = 0
            label_pos_mask = inputs.squeeze() == self.LABEL_TOKEN_IDS
            prefilled_labels = torch.full_like(inputs, -100).squeeze()
            label_values = torch.tensor([self.YES_TOKEN_IDS if label == 1 else self.NO_TOKEN_IDS for label in labels])
            prefilled_labels[label_pos_mask] = label_values
            if not self.use_label_token:
                inputs[label_pos_mask.unsqueeze(0)] = label_values
            assert self.YES_TOKEN_IDS in prefilled_labels or self.NO_TOKEN_IDS in prefilled_labels, "label_token is not in the inputs"
            inputs = {"input_ids":inputs,"attention_mask":attention_mask} 
        else:
            raise ValueError('apply_chat_template is not True')
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": prefilled_labels.unsqueeze(0) if prefilled_labels.dim() == 1 else prefilled_labels,
            "metadata": data
        }

    def _fetch_single_paper_input(self, paper, feature = 'title_year_author_org'):
        """获取单个论文的文本表示"""
        text = ""
        text += f"标题: {paper['title']}\n"
        authors_list = paper['authors']
        text += f"年份: {paper['year']}\n"
        if paper.get('venue', None) is not None:
            text += f"会议: {paper.get('venue')}\n"

        if "AuthorSequenceNumber" in authors_list[0]:
            authors_list = sorted(
                authors_list, key=lambda x: x['AuthorSequenceNumber'])
        # sorted_data = sorted(authors_list, key=lambda x: x['AuthorSequenceNumber'])

        if "name" in authors_list[0]:
            authors = [i['name'] for i in authors_list]
        elif "OriginalAuthor" in authors_list[0]:
            authors = [i.get('OriginalAuthor', '') for i in authors_list]
        if "org" in authors_list[0]:
            organizations = [i.get('org', '') for i in authors_list]
        elif "OriginalAffiliation" in authors_list[0]:
            organizations = [i.get('OriginalAffiliation', '')
                             for i in authors_list]
        organizations = list(set(organizations))
        if len(authors) > 10:
            authors = authors[:8] + authors[-2:]
            organizations = organizations[:8] + organizations[-2:]

        # 处理组织名称
        processed_orgs = []
        for org in organizations:
            processed_org = ' '.join(org.split(' ')[:10])  # 只取前10个单词
            processed_orgs.append(processed_org)
        processed_authors = []
        for author in authors:
            processed_author = ' '.join(author.split(' ')[:5])  # 只取前5个单词
            processed_authors.append(processed_author)
        authors = list(set(processed_authors))
        organizations = list(set(processed_orgs))
        # 去除重复项
        # organizations = list(set(processed_orgs))
        if "author" in feature:
            text += f"作者: {', '.join(authors)}\n"
        if "org" in feature:
            text += f"机构: {', '.join(organizations)}\n"
        # text += f"学者: {', '.join(authors)}\n"
        # text += f"机构: {', '.join(organizations)}\n"
        return text

    def _select_related_papers(self, target_paper, papers, type="random", num=20):
        """选择相关论文"""
        if type == "random":
            return random.sample(papers, num) if len(papers) > num else papers
        elif type == "year":
            target_year = target_paper.get("year", 0)
            if target_year == 0:
                return random.sample(papers, num) if len(papers) > num else papers
            else:
                # 计算每篇论文与目标论文年份的差值
                year_diff = [(i, abs(i.get('year', 0) - target_year))
                             for i in papers]
                # 按年份差值排序
                sorted_papers = [paper for paper, _ in sorted(
                    year_diff, key=lambda x: x[1])]
                # 返回年份最接近的20篇论文
                return sorted_papers[:num] if len(sorted_papers) > num else sorted_papers

    def truncate_to(self, text, length):
        """
        将输入文本截断到指定长度的token数量

        参数:
        text: str, 需要截断的输入文本
        length: int, 目标token长度

        返回:
        str: 截断后再decode成的原文本格式
        """
        # 使用tokenizer对文本进行tokenize
        tokens = self.tokenizer.tokenize(text)

        # 截断tokens到指定长度
        truncated_tokens = tokens[:length]

        # 将截断后的tokens解码回文本
        truncated_text = self.tokenizer.convert_tokens_to_string(
            truncated_tokens)

        return truncated_text

    def _build_multiturn_input(self, inner_papers, outer_papers, similarity, target_papers, name, slct_type="random", num=20):
        """构建多轮输入"""
        # 系统提示和全局提示模板

        inner = "\n".join([self._fetch_single_paper_input(paper) for paper in
                           (random.sample(inner_papers, num) if len(inner_papers) > num else inner_papers)])
        outer = "\n".join([self._fetch_single_paper_input(paper) for paper in
                           (random.sample(outer_papers, num) if len(outer_papers) > num else outer_papers)])

        # 构建多轮输入，确保每个目标论文对应的标签位置使用特殊的<label_token>
        multi_turn_input = ""
        for index, p in enumerate(target_papers):
            paper_input = self._fetch_single_paper_input(p)
            turn_prompt = self.multi_turn_prompt.format(
                index=index,
                paper=paper_input,
                label_token=LABEL_TOKEN  # 使用特殊的标签token
            )
            multi_turn_input += turn_prompt

        input_text = self.global_prompt.format(
            name=name,
            inner=inner,
            outer=outer,
            targets=multi_turn_input
        )

        return self.system_prompt, input_text


class CrossNDCollator:
    """
    为CrossNDDataset自定义的数据整理器
    处理不同长度的输入并创建batch
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples):
        # 从examples中提取input_ids, attention_mask和labels
        input_ids = [example["input_ids"] for example in examples]
        attention_mask = [example["attention_mask"] for example in examples]
        labels = [example["labels"] for example in examples]
        labels_dtype = labels[0].dtype
        # 找出这个batch中最长的序列长度
        max_length = max(ids.shape[-1] for ids in input_ids)
        label_max_length = max(lbl.shape[-1] for lbl in labels)
        # 创建填充后的张量
        batch_input_ids = torch.full(
            (len(input_ids), max_length),
            self.pad_token_id,
            dtype=torch.long
        )
        batch_attention_mask = torch.zeros(
            (len(input_ids), max_length),
            dtype=torch.long
        )
        batch_labels = torch.full(
            (len(input_ids), label_max_length),
            -100,  # -100是PyTorch中忽略的标签值
            dtype=labels_dtype
        )

        # 填充每个样本
        for i, (ids, mask, lbl) in enumerate(zip(input_ids, attention_mask, labels)):
            length = ids.shape[-1]
            batch_input_ids[i, :length] = ids
            batch_attention_mask[i, :length] = mask
            batch_labels[i, :length] = lbl

        # 检查是否所有样本都包含metadata
        has_metadata = all("metadata" in example for example in examples)

        if has_metadata:
            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels,
                "metadata": [example["metadata"] for example in examples],
            }
        else:
            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels,
            }


def min_max_normalize(data_list, v_min, v_max):
    """
    将输入的列表按照最大最小值标准化到指定范围[v_min, v_max]

    参数:
    data_list: list, 需要标准化的数据列表
    v_min: float, 目标范围的最小值
    v_max: float, 目标范围的最大值

    返回:
    list: 标准化后的数据列表

    异常处理:
    - 如果输入列表为空，返回空列表
    - 如果输入列表所有值相同，会避免除零错误
    """

    # 处理空列表的情况
    if not data_list:
        return []

    # 获取列表中的最小值和最大值
    data_min = min(data_list)
    data_max = max(data_list)

    # 处理所有值相同的情况（避免除零错误）
    if data_max == data_min:
        # 所有值标准化为目标范围的中点
        normalized_value = (v_min + v_max) / 2
        return [normalized_value] * len(data_list)

    # 执行标准化计算
    normalized_list = []
    for value in data_list:
        # 标准化公式: (x - min) / (max - min) * (v_max - v_min) + v_min
        normalized_value = (value - data_min) / \
            (data_max - data_min) * (v_max - v_min) + v_min
        normalized_list.append(normalized_value)

    return normalized_list
