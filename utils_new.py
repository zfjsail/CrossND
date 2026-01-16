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

# LABEL_TOKEN = '<label_token>'
# # LABEL_TOKEN = "<|fim_middle|>"
# EMBED_TOKEN = '<emb_token>'
# GRAPH_TOKEN = '<graph_token>' 
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"

# #TO BE ADDED
# END_OF_TEXT = '<eot>'
# END_OF_GRAPH = '<eog>'
# END_OF_EMB = '<eoe>'
# TRAINABLE_SPECIAL_TOKENS = [END_OF_TEXT,END_OF_GRAPH,END_OF_EMB,LABEL_TOKEN]

# special_token_dict = {'additional_special_tokens':TRAINABLE_SPECIAL_TOKENS+[EMBED_TOKEN,GRAPH_TOKEN]}


def paper_overlap_ratio(pids1, pids2):
    common_pids = set(pids1) & set(pids2)
    n_common_pubs = len(common_pids)
    pubs_overlap_a = n_common_pubs / len(pids1)
    pubs_overlap_m = n_common_pubs / len(pids2)
    return min(pubs_overlap_a, pubs_overlap_m), max(pubs_overlap_a, pubs_overlap_m)

def add_author_overlap(data):

    all_data = []
    name_aid_to_pids_in = json.load(open("/workspace/pangyunhe/project/crossnd/api/whoiswho/data/whoiswho/name_aid_to_pids_in.json"))
    name_aid_to_pids_out = json.load(open("/workspace/pangyunhe/project/crossnd/api/whoiswho/data/whoiswho/name_aid_to_pids_out.json"))
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
        pids1 = name_aid_to_pids_in[name].get(aid,[])
        if len(pids1) < 5:
            continue
        if pid in mag_name_pid_to_aid.get(name, {}):
            aid_map = mag_name_pid_to_aid[name][pid]
            pids_m = name_aid_to_pids_out[name][aid_map]
            if len(pids_m) < 5:
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
                 mode = "train",
                 noise_ratio = 0.0,
                 drop = False,
                 ):
        self.drop = drop
        self.noise_ratio = noise_ratio
        self.num_turn = model_args.num_turn
        self.mode = mode.lower()  # 确保模式字符串为小写
        self.use_outer = model_args.use_outer
        self.model_args = model_args

        if model_args.dataset == 'kddcup':
            in_name2pid = json.load(open('/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup/aid_to_pids_in.json'))
            out_name2pid = json.load(open('/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup/aid_to_pids_out.json'))
            clean_in_name2pid = json.load(open('/workspace/pangyunhe/project/crossnd/api/ark_batch_inf/ind_output/inner_output.json'))
            clean_out_name2pid = json.load(open('/workspace/pangyunhe/project/crossnd/api/ark_batch_inf/ind_output/outer_output.json'))
            for k1,v1 in in_name2pid.items():
                if k1 not in clean_in_name2pid:
                    clean_in_name2pid[k1] = v1
            for k1, v1 in out_name2pid.items():
                if k1 not in clean_out_name2pid:
                    clean_out_name2pid[k1] = v1
            self.in_name2pid = clean_in_name2pid
            self.out_name2pid = clean_out_name2pid
            paper_path = os.path.join(data_dir, "paper_dict_mag.json")
            self.paper_data = json.load(open(paper_path,'r'))
        elif model_args.dataset == 'whoiswho':
            self.in_name2pid = json.load(open("/workspace/pangyunhe/project/crossnd/api/whoiswho/self_clean/cleaned_name_aid_to_pids_in.json"))
            self.out_name2pid = json.load(open("/workspace/pangyunhe/project/crossnd/api/whoiswho/self_clean/cleaned_name_aid_to_pids_out.json"))
            self.paper_data = json.load(open("/workspace/pangyunhe/project/crossnd/api/whoiswho/data/whoiswho/pub_dict.json"))
            def flatten_name_dict(data):
                temp_dict = {}
                for k,v in data.items():
                    for aid1,pids in v.items():
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
            self.user_prompt = """ {paper} 是该学者的论文吗? """
            self.assistant_prompt = """{label_token}"""
        else:
            self.system_prompt = """你要进行一个学者论文检测的任务,每篇论文包题目,学者,机构信息,现在需要进行异常分配的论文检测,即检测给定论文是否应该属于该学者,不属于该学者的论文是异常论文,判断给定的论文是否属于该学者, 如果属于则返回 Yes ,否则返回 No。"""
            self.global_prompt = """目标学者名字: {name} 
            该学者的论文,可能存在错误分配论文: {inner} 
            """
            
            self.user_prompt = """ {paper} 结果是什么? """
            self.assistant_prompt = """{label_token}"""
            self.multi_turn_prompt = """ {paper} 结果是 {label_token}\n"""     
    
        self.tokenizer = tokenizer
        if self.mode == "train":
            data = json.load(open(model_args.src))
            if self.model_args.dataset == "whoiswho" and 'author_sim' not in data:
                data = add_author_overlap(data)
            if self.model_args.label_thr is not None and self.mode == 'train':
                for i in data:
                    i['ori_label'] = i['label']
                    if i['soft_label']>=self.model_args.label_thr:
                        i['label'] = 1
                    else:
                        i['label'] =0
            # random.shuffle(data)
            # all_data = data
            pos = [i for i in data if i['label'] ==1 ]
            neg = [i for i in data if i['label'] ==0 ]
            if self.model_args.upsample:
                if len(pos) > len(neg) and len(pos)/len(neg) > 1.5:
                    neg = neg * (len(pos)//len(neg)) # 平衡正负样本
                elif len(pos) < len(neg) and len(neg)/len(pos) > 1.5:
                    pos = pos * (len(neg)//len(pos)) # 平衡正负样本
            all_data = pos + neg
            random.shuffle(all_data)
        elif self.mode == "eval":
            if self.model_args.dataset == "kddcup":
                data_file = os.path.join(data_dir, "valid_with_sim.json")
                # data_file = os.path.join(data_dir, "test_with_sim.json")
                all_data = json.load(open(data_file))
            elif self.model_args.dataset == "whoiswho":
                all_data =json.load(open("/workspace/pangyunhe/project/crossnd/api/whoiswho/data/whoiswho/eval_na_checking_triplets_valid.json"))
                if 'author_sim' not in all_data[0]: 
                    all_data = add_author_overlap(all_data)
        else:
            if self.model_args.dataset == "kddcup":
                data_file = os.path.join(data_dir, "test_with_sim.json")
                all_data = json.load(open(data_file))
            elif self.model_args.dataset == "whoiswho":
                all_data =json.load(open("/workspace/pangyunhe/project/crossnd/api/whoiswho/data/whoiswho/eval_na_checking_triplets_test.json"))
                if 'author_sim' not in all_data[0]:
                    all_data = add_author_overlap(all_data)
        if model_args.num_turn > 1:
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
            
            # 调试打印
            print(f"\n=== 调试信息 ===")
            print(f"num_turn: {self.num_turn}")
            print(f"all_data 原始大小: {len(all_data)}")
            print(f"按学者分组后的组数: {len(data_dd)}")
            for k, v in list(data_dd.items())[:5]:  # 打印前5个学者的数据
                print(f"  学者 {k}: {len(v)} 条记录")
            
            for v in data_dd.values():
                v_ = copy.deepcopy(v)
                random.shuffle(v_)
                for i in range(0, len(v_), self.num_turn):
                    data.append(v_[i:i+self.num_turn])
            
            # 调试打印
            print(f"最终 self.data 的大小: {len(data)}")
            print(f"=== 调试信息结束 ===\n")
            
            self.data = data
        else:
            random.shuffle(all_data)
            self.data = [[i] for i in all_data]

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        aid1, aid2, name = data[0]['aid1'], data[0].get('aid2',None), data[0]['name']

        aid1 = str(aid1)
        aid2 = str(aid2) if aid2 is not None else None
        
        if 'similarity' in data[0]:
            similarity = data[0]['similarity']
        elif 'author_sim' in data[0]:
            similarity = data[0]['author_sim']
        else:
            raise ValueError('no similarity found')

        label_list = [i['label'] for i in data]
        labels = torch.tensor(label_list, dtype=torch.long)
        if self.noise_ratio > 0:
            for i in range(len(labels)):
                if random.random() < self.noise_ratio:
                    labels[i] = 1 - labels[i]
        def fetch_paper_id(src_id):
            return src_id.split('-')[0]
        
        # 获取论文信息
        pids = [fetch_paper_id(i['pid']) for i in data]
        papers = [self.paper_data[i] for i in pids]

        if self.drop:
            drop_num = random.randint(0, len(data)-1)
            if drop_num != 0:
                data = data[:len(data)-drop_num]
                labels = labels[:len(labels)-drop_num]
                pids = pids[:len(pids)-drop_num]
                papers = papers[:len(papers)-drop_num]
 
        papers_in = [i for i in self.in_name2pid[aid1] if i != data[0]['pid']]
        inner_papers = [self.paper_data[fetch_paper_id(i)] for i in papers_in]
        selected_inner_papers = self._select_related_papers(papers[0], inner_papers, type='random',num=self.model_args.paper_slct_num)
        inner = "\n".join([self._fetch_single_paper_input(paper) for paper in selected_inner_papers])
        if self.use_outer and aid2 is not None:
            papers_out = [i for i in self.out_name2pid[aid2] if i != data[0]['pid']]
            outer_papers = [self.paper_data[fetch_paper_id(i)] for i in papers_out]
            # selected_outer_papers = self._select_related_papers(papers[0], outer_papers, type='random',num=self.model_args.paper_slct_num)
            selected_outer_papers = self._select_related_papers(papers[0], outer_papers, type='random',num=self.model_args.paper_slct_num)
            outer_papers = [self._fetch_single_paper_input(paper) for paper in selected_outer_papers]
            pred_paper_inputs_len = len(self.tokenizer(' ; '.join([self._fetch_single_paper_input(paper) for paper in papers]))['input_ids'])

            if self.model_args.max_seq_length is None:
                outer = "\n".join(outer_papers)
            else:
                inner_token_len = len(self.tokenizer(inner)['input_ids'])
                outer_token_len = self.model_args.max_seq_length - inner_token_len - pred_paper_inputs_len - 500 

                outer_curr_len = 0
                num_outer_papers = 0
                cuted_outer_papers = []
                random.shuffle(outer_papers)
                while outer_curr_len < outer_token_len and num_outer_papers < len(outer_papers):
                    cuted_outer_papers.append(outer_papers[num_outer_papers])
                    outer_curr_len += len(self.tokenizer(outer_papers[num_outer_papers])['input_ids'])
                    num_outer_papers += 1
                outer = "\n".join(cuted_outer_papers)

        else:
            outer = None
            similarity = None

        # 构建模型输入
        chat = []
        
        
        # 构建全局提示
        
        global_prompt = self.global_prompt.format(**{
            "name": name,
            "inner": inner, 
            "outer": outer if outer is not None else 'none',
            "author_sim":round(similarity,2) if similarity is not None else 'none'
        })
        chat.append({"role": "system", "content": self.system_prompt +global_prompt})
        # 第一轮对话
        for i in range(len(papers)):
            chat.append({"role": "user", "content": self.user_prompt.format(paper=self.truncate_to(self._fetch_single_paper_input(papers[i]),200))})
            chat.append({"role": "assistant", "content": 'Yes' if labels[i] == 1 else 'No'})
        # include metadata to calculate similarity and etc
        return {
            "messages": chat[:2],
            "data_source":"multiturnnd",
            "reward_model":{
                "style":"rule",
                "ground_truth":['Yes' if label == 1 else 'No' for label in labels]
            },
            "extra_info":{
                "interaction_kwargs":{
                    "name": "multiturn",
                    "messages":chat[1:],
                    "ground_truth":['Yes' if label == 1 else 'No' for label in labels]
                },
                "metadata":{
                    "name": name,
                    "aid1": aid1,
                    "aid2": aid2,
                    "pids": pids,
                    "similarity": similarity
                }
            },
            # "interaction_kwargs":{
            #         "name": "multiturn",
            #         "messages":chat,
            #         "ground_truth":['Yes' if label == 1 else 'No' for label in labels]
            #     }
        }
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
        truncated_text = self.tokenizer.convert_tokens_to_string(truncated_tokens)
        
        return truncated_text

    def resample_dataset(self,):
        self.data = []
        for k,v in self.tmp_data.items():
            v = copy.deepcopy(v)
            random.shuffle(v)
            for i in range(0, len(v), self.num_turn):
                self.data.append(v[i:i+self.num_turn]) 

    def _fetch_single_paper_input(self, paper, feature = 'title_author_org'):
        """获取单个论文的文本表示"""
        text = ""
        text += f"标题: {self.truncate_to(paper['title'],100)}\n"
        authors_list = paper['authors']
        if "AuthorSequenceNumber" in authors_list[0]:
            authors_list = sorted(authors_list, key=lambda x: x['AuthorSequenceNumber']) 
        # sorted_data = sorted(authors_list, key=lambda x: x['AuthorSequenceNumber'])

        if "name" in authors_list[0]:
            authors = [i['name'] for i in authors_list]
        elif "OriginalAuthor" in authors_list[0]:
            authors = [i.get('OriginalAuthor','') for i in authors_list]
        if "org" in authors_list[0]:
            organizations = [i.get('org','') for i in authors_list]
        elif "OriginalAffiliation" in authors_list[0]:
            organizations = [i.get('OriginalAffiliation','') for i in authors_list]
        organizations = list(set(organizations))
        if len(authors) > 10:
            authors = authors[:8] + authors[-2:]
            organizations = organizations[:8] + organizations[-2:]
        
        # 处理组织名称
        processed_orgs = []
        for org in organizations:
            processed_org = self.truncate_to(org,50)  # 只取前10个单词
            processed_orgs.append(processed_org)
        processed_authors = []
        for author in authors:
            processed_author = self.truncate_to(author,50) # 只取前5个单词
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
    
    def _select_related_papers(self, target_paper, papers, type="random",num=20):
        
        """选择相关论文"""
        if type == "random":
            return random.sample(papers, num) if len(papers) > num else papers
        elif type == "year":
            target_year = target_paper.get("year", 0)
            if target_year == 0:
                return random.sample(papers, num) if len(papers) > num else papers
            else:
                # 计算每篇论文与目标论文年份的差值
                year_diff = [(i, abs(i.get('year', 0) - target_year)) for i in papers]
                # 按年份差值排序
                sorted_papers = [paper for paper, _ in sorted(year_diff, key=lambda x: x[1])]
                # 返回年份最接近的20篇论文
                return sorted_papers[:num] if len(sorted_papers) > num else sorted_papers
    
