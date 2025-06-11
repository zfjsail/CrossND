#!/usr/bin/env python
# coding=utf-8

import json
import os
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union, Tuple
import copy


LABEL_TOKEN = '<label_token>'
EMBED_TOKEN = '<emb_token>'
GRAPH_TOKEN = '<graph_token>' 
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

#TO BE ADDED
END_OF_TEXT = '<eot>'
END_OF_GRAPH = '<eog>'
END_OF_EMB = '<eoe>'
TRAINABLE_SPECIAL_TOKENS = [END_OF_TEXT,END_OF_GRAPH,END_OF_EMB,LABEL_TOKEN]

special_token_dict = {'additional_special_tokens':TRAINABLE_SPECIAL_TOKENS+[EMBED_TOKEN,GRAPH_TOKEN]}



class CrossNDDataset(Dataset):
    """学者论文异常检测数据集"""
    
    def __init__(self, 
                 data_dir,
                 tokenizer,
                 num_turn = 1,
                 mode = "train",
                 apply_chat_template = True,
                 use_outer = True
                 ):
        self.num_turn = num_turn
        self.apply_chat_template = apply_chat_template
        self.mode = mode.lower()  # 确保模式字符串为小写
        self.use_outer = use_outer
        if use_outer:
            self.system_prompt = """
            你要进行一个学者论文检测的任务, 每个作者有两个源可以使用, 一个是内部源, 另一个是外部源, 使用外部源来支持内部源论文的错误分配检测,每个源是由论文的集合组成的, 每篇论文包题目,作者,机构信息, 现在需要基于两个源进行异常分配的论文检测,即检测给定论文是否应该属于该作者,不属于该作者的论文是异常论文, 现在有一批论文, 你需要根据内部源和外部源的论文与该论文的相似性,来判断该论文是否属于内部源
            """
            # self.global_prompt = """目标学者名字: {name} 
            # 内部源: {inner} 
            # 外部源: {outer} 
            # 两个源之间的相似性是: {similarity}\n
            # """
            self.global_prompt = """目标学者名字: {name} 
            内部源: {inner} 
            外部源: {outer} 
            """
            if self.apply_chat_template:
                self.user_prompt = """ {paper} 是该作者的论文吗? """
                self.assistant_prompt = """{label_token}"""
            else:
                self.multi_turn_prompt = """ {paper} 结果是 {label_token}\n"""   

            self.tokenizer = tokenizer
            
            # 根据模式加载不同的数据文件
            if self.mode == "train":
                # data_file = os.path.join(data_dir, "deleted_train_triplets.json")
                # pos_file = "/workspace/pangyunhe/pangyunhe1/git/crossnd-202211/data/kddcup/positive_paper_author_pairs.json"
                # neg_file = "/workspace/pangyunhe/pangyunhe1/git/crossnd-202211/data/kddcup/negative_paper_author_pairs.json"
                # pos_data = json.load(open(pos_file))
                # neg_data = json.load(open(neg_file))
                # for i in pos_data:
                #     i['label'] = 1
                #     i['aid1'] = i['aid']
                #     i['similarity'] = 0

                # for i in neg_data:
                #     i['label'] = 0
                #     i['aid1'] = i['aid']
                #     i['similarity'] = 0
                # all_data = pos_data + neg_data
                all_data = json.load(open("/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup/train_samples.json"))
                random.shuffle(all_data)
                
            elif self.mode == "eval":
                data_file = os.path.join(data_dir, "valid_with_sim.json")
                all_data = json.load(open(data_file))
            else:
                data_file = os.path.join(data_dir, "test_with_sim.json")
                all_data = json.load(open(data_file))


            paper_path = os.path.join(data_dir, "paper_dict_mag.json")
            in_name2pid_path = os.path.join(data_dir,"name_aid_to_pids_in.json")
            out_name2pid_path =  os.path.join(data_dir,"name_aid_to_pids_out.json")
            with open(in_name2pid_path, 'r', encoding='utf-8') as f:
                self.in_name2pid = json.load(f)
            with open(out_name2pid_path, 'r', encoding='utf-8') as f:
                self.out_name2pid = json.load(f)
            self.paper_data = json.load(open(paper_path,'r'))

            # 构建数据
            self.data = []
            # 处理数据，根据不同模式可能有不同的处理方式
            self.tmp_data = {}
            # 对于单回合设置
            if num_turn == 1:       
            
                if self.mode == "train":
                    # 只加载一次valid数据
                    data_file = os.path.join(data_dir, "valid_with_sim.json")
                    valid_data = json.load(open(data_file)) 
                    # 预先构建valid_keys列表
                    valid_keys = {}
                    for v in valid_data:
                        valid_keys[f"{v['aid1']}-{v['pid']}"] = v['label']
                        
                    for raw in all_data:
                        if f"{raw['aid1']}-{raw['pid']}" not in valid_keys:
                            self.data.append([raw])
                        else:
                            raw['label'] = valid_keys[f"{raw['aid1']}-{raw['pid']}"]
                            self.data.append([raw])
                else:
                    for raw in all_data:
                        self.data.append([raw])








            else:  # 对于多回合设置
                for item in all_data:
                    pid, aid1, aid2, name, label, sim = item['pid'], item['aid1'], item['aid2'], item['name'], item['label'], item['similarity']
                    key = f"{aid1}-{aid2}"
                    if key not in self.tmp_data:
                        self.tmp_data[key] = []
                    self.tmp_data[key].append(item)
                
                for k, v in self.tmp_data.items():
                    v = copy.deepcopy(v)
                    if self.mode == "train":
                        random.shuffle(v)  # 只在训练模式下随机打乱
                    for i in range(len(v) // self.num_turn):
                        self.data.append(v[i:i+self.num_turn])
                self.tokenizer = tokenizer
                
                # 根据模式加载不同的数据文件
                if self.mode == "train":
                    data_file = os.path.join(data_dir, "deleted_train_triplets.json")
                elif self.mode == "eval":
                    data_file = os.path.join(data_dir, "valid_with_sim.json")
                else:
                    data_file = os.path.join(data_dir, "test_with_sim.json")
            









            # if self.mode == "eval":  # debug
            #     self.data = self.data[:50]
        else:
            self.system_prompt = """
            你要进行一个学者论文检测的任务,每篇论文包题目,作者,机构信息,现在需要进行异常分配的论文检测,即检测给定论文是否应该属于该作者,不属于该作者的论文是异常论文,判断给定的论文是否属于该作者, 如果属于则返回 Yes ,否则返回 No。
            """
            self.global_prompt = """目标学者名字: {name} 
            该作者的论文,可能存在错误分配论文: {inner} 
            """
            self.multi_turn_prompt = """ {paper} 结果是 {label_token}\n"""     
            if self.apply_chat_template:          
                self.user_prompt = """ {paper} 结果是什么? """
                self.assistant_prompt = """{label_token}"""
            else:
                self.multi_turn_prompt = """ {paper} 结果是 {label_token}\n"""  
            
            
            self.tokenizer = tokenizer
            
            # 根据模式加载不同的数据文件
            if self.mode == "train":
                # data_file = os.path.join(data_dir, "deleted_train_triplets.json")
                pos_file = "/workspace/pangyunhe/pangyunhe1/git/crossnd-202211/data/kddcup/positive_paper_author_pairs.json"
                neg_file = "/workspace/pangyunhe/pangyunhe1/git/crossnd-202211/data/kddcup/negative_paper_author_pairs.json"
                pos_data = json.load(open(pos_file))
                neg_data = json.load(open(neg_file))
                for i in pos_data:
                    i['label'] = 1
                    i['aid1'] = i['aid']
                    i['similarity'] = 0

                for i in neg_data:
                    i['label'] = 0
                    i['aid1'] = i['aid']
                    i['similarity'] = 0
                all_data = pos_data + neg_data
                random.shuffle(all_data)
                
            elif self.mode == "eval":
                data_file = os.path.join(data_dir, "valid_with_sim.json")
                all_data = json.load(open(data_file))
            else:
                data_file = os.path.join(data_dir, "test_with_sim.json")
                all_data = json.load(open(data_file))
                
            paper_path = os.path.join(data_dir, "paper_dict_mag.json")
            in_name2pid_path = os.path.join(data_dir,"name_aid_to_pids_in.json")
            out_name2pid_path =  os.path.join(data_dir,"name_aid_to_pids_out.json")
            with open(in_name2pid_path, 'r', encoding='utf-8') as f:
                self.in_name2pid = json.load(f)
            with open(out_name2pid_path, 'r', encoding='utf-8') as f:
                self.out_name2pid = json.load(f)
            self.paper_data = json.load(open(paper_path,'r'))

            
            
            self.data = []
            if self.mode == "train":
                # 只加载一次valid数据
                data_file = os.path.join(data_dir, "valid_with_sim.json")
                valid_data = json.load(open(data_file)) 
                # 预先构建valid_keys列表
                valid_keys = {}
                for v in valid_data:
                    valid_keys[f"{v['aid1']}-{v['pid']}"] = v['label']
                    
                for raw in all_data:
                    if f"{raw['aid1']}-{raw['pid']}" not in valid_keys:
                        self.data.append([raw])
                    else:
                        raw['label'] = valid_keys[f"{raw['aid1']}-{raw['pid']}"]
                        self.data.append([raw])
            else:
                for raw in all_data:
                    self.data.append([raw])


    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]

        aid1, aid2, name, similarity = data[0]['aid1'], data[0].get('aid2',None), data[0]['name'], data[0].get('similarity',None)
        label_list = [i['label'] for i in data]
        def fetch_paper_id(src_id):
            return src_id.split('-')[0]
        
        # 获取论文信息
        pids = [fetch_paper_id(i['pid']) for i in data]
        papers = [self.paper_data[i] for i in pids]

        papers_in = [i for i in self.in_name2pid[name][aid1] if i != data[0]['pid']]
        inner_papers = [self.paper_data[fetch_paper_id(i)] for i in papers_in]
        selected_inner_papers = self._select_related_papers(papers[0], inner_papers, type='random')
        inner = "\n".join([self._fetch_single_paper_input(paper) for paper in selected_inner_papers])
                
        if self.use_outer:
            if aid2 is not None:
                papers_out = [i for i in self.out_name2pid[name][aid2] if i != data[0]['pid']]
                outer_papers = [self.paper_data[fetch_paper_id(i)] for i in papers_out]
                selected_outer_papers = self._select_related_papers(papers[0], outer_papers, type='random')
                outer = "\n".join([self._fetch_single_paper_input(paper) for paper in selected_outer_papers])
            else:
                outer = "无"
        else:
            outer = None
        


        # 构建模型输入
        if self.apply_chat_template:
            chat = []
            chat.append({"role": "system", "content": self.system_prompt})
            
            # 构建全局提示
            global_prompt = self.global_prompt.format(**{
                "name": name,
                "inner": inner, 
                "outer": outer
            })
            
            # 第一轮对话
            first_paper_prompt = global_prompt + self.user_prompt.format(paper=self._fetch_single_paper_input(papers[0]))
            chat.append({"role": "user", "content": first_paper_prompt})
            chat.append({"role": "assistant", "content": self.assistant_prompt.format(label_token=LABEL_TOKEN)})
            
            # 如果有多轮对话，添加后续轮次
            for paper in papers[1:]:
                chat.append({"role": "user", "content": self.user_prompt.format(paper=self._fetch_single_paper_input(paper))})
                chat.append({"role": "assistant", "content": self.assistant_prompt.format(label_token=LABEL_TOKEN)})
            
            # 应用聊天模板
            inputs = self.tokenizer.apply_chat_template(chat,return_tensors="pt")
            # 确保attention_mask与input_ids形状一致
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
            inputs = {"input_ids":inputs,"attention_mask":attention_mask}
        else:
            # 不使用聊天模板的情况
            inputs = self.system_prompt + "\n"
            inputs += self.global_prompt.format(**{
                "name": name,
                "inner": inner, 
                "outer": outer
            })
            
            # 添加所有论文
            for paper in papers:
                inputs += self.multi_turn_prompt.format(paper=self._fetch_single_paper_input(paper), label_token=LABEL_TOKEN)
            
            # 分词 - 删除重复的代码行，确保返回attention_mask
            inputs = self.tokenizer.encode_plus(inputs, return_tensors="pt", truncation=False, padding=False)
        
        labels= torch.tensor(label_list, dtype=torch.long)
        if self.mode != "train":
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels.unsqueeze(0) if labels.dim() == 1 else labels,
                "metadata": data
            }
        else:
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels.unsqueeze(0) if labels.dim() == 1 else labels,
            }
    
    def resample_dataset(self,):
        self.data = []
        for k,v in self.tmp_data.items():
            v = copy.deepcopy(v)
            random.shuffle(v)
            for i in range(0, len(v), self.num_turn):
                self.data.append(v[i:i+self.num_turn]) 

    def _fetch_single_paper_input(self, paper):
        """获取单个论文的文本表示"""
        text = ""
        
        text += f"Title: {paper['title']}\n"
        authors = [i.get('OriginalAuthor','') for i in paper['authors']]
        organizations = [i.get('OriginalAffiliation','') for i in paper['authors']]
        if len(authors) > 10:
            authors = authors[:8] + authors[-2:]
            organizations = organizations[:8] + organizations[-2:]
        
        # 处理组织名称
        processed_orgs = []
        for org in organizations:
            processed_org = ' '.join(org.split(' ')[:10])  # 只取前10个单词
            processed_orgs.append(processed_org)
        
        # 去除重复项
        organizations = list(set(processed_orgs))
        
        text += f"Authors: {', '.join(authors)}\n"
        text += f"Organizations: {', '.join(organizations)}\n"

        return text
    
    def _select_related_papers(self, target_paper, papers, type="random"):
        
        """选择相关论文"""


        if type == "random":
            return random.sample(papers, 40) if len(papers) > 40 else papers
        elif type == "year":
            target_year = target_paper.get("year", 0)
            if target_year == 0:
                return random.sample(papers, 40) if len(papers) > 40 else papers
            else:
                # 计算每篇论文与目标论文年份的差值
                year_diff = [(i, abs(i.get('year', 0) - target_year)) for i in papers]
                # 按年份差值排序
                sorted_papers = [paper for paper, _ in sorted(year_diff, key=lambda x: x[1])]
                # 返回年份最接近的20篇论文
                return sorted_papers[:40] if len(sorted_papers) > 40 else sorted_papers
    
    def _build_multiturn_input(self, inner_papers, outer_papers, similarity, target_papers, name, slct_type="random"):
        """构建多轮输入"""
        # 系统提示和全局提示模板

        
        inner = "\n".join([self._fetch_single_paper_input(paper) for paper in 
                        (random.sample(inner_papers, 40) if len(inner_papers) > 40 else inner_papers)])
        outer = "\n".join([self._fetch_single_paper_input(paper) for paper in 
                        (random.sample(outer_papers, 40) if len(outer_papers) > 40 else outer_papers)])
        
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
            dtype=torch.long
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

