system_prompt = """你是一个异常论文检测模型, 你需要判断一篇论文是否是因为同名学者等原因错误地分配给该学者的论文,由于原本集合中本来就存在一些错误的论文,因此你需要根据论文的共同作者,研究领域,发表期刊会议,机构等因素,你的返回应该按照 "id":"label" 的形式返回json数据,label是0或者1,例如{"1":"0", "2":"1", "3":"0"...}"""


multi_turn_prompt = """第 {index} 篇: {paper} \n"""

import json
import random
from tqdm import tqdm
from argparse import ArgumentParser
from openai import OpenAI
import re
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from LLM import LLMClass


def fetch_single_paper_input(paper, feature_list):
    """Format paper information based on the feature list."""
    title, authors, orgs, venue, year, keywords = '', '', '', '', '', ''
    feats = feature_list.split('_')
    
    if 'title' in feats:
        title = f"Title: {paper['title']}\n"
    
    if 'author' in feats:
        authors = [i.get('name','') for i in paper['authors']]
        if len(authors) > 10:
            authors = authors[:8] + authors[-2:]
        authors = f"Authors: {', '.join(authors)}\n"
    
    if 'organization' in feats:
        organizations = [i.get('org','') for i in paper['authors']]
        if len(organizations) > 10:
            organizations = organizations[:8] + organizations[-2:]
        processed_orgs = []
        for org in organizations:
            processed_org = ' '.join(org.split(' ')[:10])
            processed_orgs.append(processed_org)
        organizations = list(set(processed_orgs))
        orgs = f"Organizations: {', '.join(organizations)}\n"
    
    if 'venue' in feats:
        venue = f"Venue: {paper.get('venue','None')}\n"

    if 'year' in feats:
        year = f"Year: {str(paper.get('year','None'))}\n"

    if 'keywords' in feats:
        keywords = f"Keywords: {' ;'.join(paper.get('keywords',[]))}\n"
    
    text = (title if 'title' in feats else '') + \
           (authors if 'author' in feats else '') + \
           (orgs if 'organization' in feats else '') + \
           (venue if 'venue' in feats else '') + \
           (year if 'year' in feats else '') + \
           (keywords if 'keywords' in feats else '')
    
    return text

def fetch_paper_id(src_id):
    """Extract the paper ID from the source ID."""
    return src_id.split('-')[0]

def build_ind_input(collection_papers, target_papers, feature_list):
    """Build the input for the anomaly detection model."""
    collection_papers = random.sample(collection_papers, min(len(collection_papers), 40))
    collection = "\n".join([fetch_single_paper_input(paper, feature_list) for paper in collection_papers])
    multi_turn_input = "\n".join([multi_turn_prompt.format(index=index, paper=fetch_single_paper_input(paper, feature_list)) for index, paper in enumerate(target_papers)])
    prompt = f"""论文集合: {collection} \n 需要判断的论文: {multi_turn_input} \n 请判断这些论文是否属于给定学者,0表示不属于,1表示属于。"""
    return prompt

def prepare_inference_data(args, paper_data, target_data, data):
    
    all_pred_aids = {}
    for item in data:
        
        name = item['name']
        if 'name_aid_to_pids_in' in args.target and 'name_aid_to_pids_out' not in args.target:
            aid = item['aid1']
        elif 'name_aid_to_pids_out' in args.target and 'name_aid_to_pids_in' not in args.target:
            aid = item['aid2']
        else:
            raise ValueError 
        if aid in all_pred_aids:
            all_pred_aids[name].append(aid)
        else:
            all_pred_aids[name] = [aid]
    all_papers = defaultdict(dict)
    for name,aids in all_pred_aids.items():
        for aid in aids:
            if len(target_data[name][aid])>5:
                all_papers[aid] = target_data[name][aid]
    inference_data = []
    for k,v in all_papers.items():
        paper_ids = [fetch_paper_id(p) for p in v]
        papers = [paper_data[pid] for pid in paper_ids if pid in paper_data]
        random.shuffle(papers)
        for i in range(0,len(papers), args.batch_size):
            batch_papers = papers[i:i+args.batch_size]
            batch_input = build_ind_input(
                collection_papers=papers,
                target_papers=batch_papers,
                feature_list=args.feature_list
            )
            inference_data.append({'aid':k,'pids':v,'inputs':batch_input})
    return inference_data

def main_inference(args):
    save_name = f"self_clean/{args.model}_{args.target.split('/')[-1]}.json"
    print(f'to be saved to {save_name}')
    if os.path.exists(save_name):
        data = json.load(open(save_name,'r'))
        finished_data = []
        nonfinished_data = []
        for item in data:
            if item['content'] =="":
                nonfinished_data.append(item)
            else:
                finished_data.append(item)
        print(f'totally {len(data)} items, finished: {len(finished_data)}, none finished: {len(nonfinished_data)}')
        llm  = LLMClass()      
        results = llm.process_batch_with_pool([i['inputs'] for i in nonfinished_data],
                                            model = args.model,
                                            system_prompt = system_prompt
                                            )                  
        try:
            for inf, res in zip(nonfinished_data, results):
                inf['content'] = res.content
                inf['reasoning_content'] = res.reasoning_content
            inference_data = nonfinished_data + finished_data
        except:
            breakpoint()
    else:         
        # Load paper data
        paper_data = json.load(open(args.paper_dict))

        target_data  = json.load(open(args.target))
        data = json.load(open(args.triplets_train)) \
        + json.load(open(args.triplets_test)) \
        + json.load(open(args.triplets_valid))
        for item in data:
            item['aid1'] = str(item['aid1'])   
            item['aid2'] = str(item['aid2'])
        inference_data = prepare_inference_data(args, paper_data, target_data, data)
        print(len(inference_data))
        llm  = LLMClass()
        results = llm.process_batch_with_pool([i['inputs'] for i in inference_data],
                                                model = args.model,
                                                system_prompt = system_prompt
                                                )
        for inf, res in zip(inference_data, results):
            inf['content'] = res.content
            inf['reasoning_content'] = res.reasoning_content

        with open(save_name, 'w', encoding='utf-8') as f:
            json.dump(inference_data, f, indent=4, ensure_ascii=False)
        
        print(f"Results saved to {save_name}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--feature_list", type=str, default="title_author_organization_venue_year_keywords", 
                      help="Features to include, separated by '_'")
    parser.add_argument("--model", type=str, default="your-model-name")
    parser.add_argument("--batch_size", type=int, default=40,
                      help="Batch size for inference")
    parser.add_argument("--target", type=str, required=True,
                      help="Path to name_aid_to_pids_in.json or name_aid_to_pids_out.json")
    parser.add_argument("--paper_dict", type=str, required=True,
                      help="Path to pub_dict.json")
    parser.add_argument("--triplets_train", type=str, required=True,
                      help="Path to triplets_train_author_subset.json")
    parser.add_argument("--triplets_test", type=str, required=True,
                      help="Path to eval_na_checking_triplets_test.json")
    parser.add_argument("--triplets_valid", type=str, required=True,
                      help="Path to eval_na_checking_triplets_valid.json")
    args = parser.parse_args()
    main_inference(args)
