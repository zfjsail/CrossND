"""
Batch Refine V2 for WhoIsWho Dataset

与 batch_refine.py 的区别：
- 原版将所有批次的论文合并为一个大列表后整体送给 LLM
- V2 保留原始批次结构，prompt 中按批次组织，格式为：
    第 X 批:
      第 1 篇: <论文信息>, 分数: <score>
      第 2 篇: <论文信息>, 分数: <score>
      ...
  LLM 需要对每批每篇论文分别打分，返回格式为：
    {"batch_0": {"0": score, "1": score, ...}, "batch_1": {...}, ...}
"""

import json
import random
import re
import os
from collections import defaultdict
from argparse import ArgumentParser
from LLM import LLMClass
from utils import extract_and_parse_json, fetch_single_paper_input, fetch_paper_id, crossnd_global_prompt_wo_similarity

system_prompt = """你要进行一个学者论文异常分配检测的任务, 输入的每一篇论文包含题目，作者，机构，会议等信息，并且包含一个额外的0到1之间的分数信息，是之前分批次的异常分配检测的分数，越接近1分表明该论文越像是该学者的论文，但其中可能存在一些噪声。你可以根据整体论文的信息来判断学者大概的研究领域、合作学者、机构等信息。输入的论文按批次组织，每批包含若干篇带分数的论文。给定所有批次的论文及其打分，你需要综合所有论文信息，逐篇重新判断每篇论文是否属于该学者，并重新打分（仍然是越接近1分越像该学者的论文）。你的返回应该严格按照以下JSON格式：{"batch_0": {"0": score, "1": score, ...}, "batch_1": {"0": score, ...}, ...}，其中batch编号与输入一致，每批内部的键为该批内的论文序号（从0开始），score为0到1之间的浮点数，越接近0表明该论文不属于该作者，越接近1表明该论文属于该作者。仅输出json并且不包含其他内容，并且json应该能够使用json.loads来解析。"""

batch_header_prompt = """第 {batch_index} 批（共 {batch_size} 篇）:\n"""
multi_turn_prompt = """  第 {index} 篇: {paper} , 分数: {score}\n"""


def build_batch_input_v2(data, paper_data, in_name2pid, out_name2pid, args):
    """
    构建用于批量推理的输入（V2：保留批次结构）
    """
    aid1 = data['aid1']
    aid2 = data['aid2']
    name = data['name']
    batches = data['batches']

    ordered_batch_info = []
    target_papers_text = ""

    for batch_idx, batch in enumerate(batches):
        pids = batch['pids']
        preds = batch['preds']
        papers = [paper_data[fetch_paper_id(pid)] for pid in pids]

        zipped = list(zip(papers, preds, pids))
        random.shuffle(zipped)

        batch_pids_ordered = []
        batch_preds_ordered = []

        target_papers_text += batch_header_prompt.format(
            batch_index=batch_idx,
            batch_size=len(zipped)
        )
        for i, (paper, pred, pid) in enumerate(zipped):
            target_papers_text += multi_turn_prompt.format(
                index=i,
                paper=fetch_single_paper_input(paper, args.feature_list),
                score=pred
            )
            batch_pids_ordered.append(pid)
            batch_preds_ordered.append(pred)

        ordered_batch_info.append({
            'batch_idx': batch_idx,
            'pids': batch_pids_ordered,
            'preds': batch_preds_ordered,
        })

    inner_papers = [paper_data[fetch_paper_id(
        i)] for i in in_name2pid.get(aid1, [])]
    outer_papers = [paper_data[fetch_paper_id(
        i)] for i in out_name2pid.get(aid2, [])]

    inner_papers = random.sample(inner_papers, min(len(inner_papers), 20))
    outer_papers = random.sample(outer_papers, min(len(outer_papers), 20))

    inner_papers_ctx = "\n".join([fetch_single_paper_input(
        paper, args.feature_list) for paper in inner_papers])
    outer_papers_ctx = "\n".join([fetch_single_paper_input(
        paper, args.feature_list) for paper in outer_papers])

    batch_inputs = crossnd_global_prompt_wo_similarity.format(
        name=name,
        inner=inner_papers_ctx,
        outer=outer_papers_ctx,
        targets=target_papers_text
    )

    return batch_inputs, ordered_batch_info


def prepare_inference_data(overall_results, paper_data, in_name2pid, out_name2pid, args):
    """
    准备推理数据
    """
    batch_data = []

    for k, v in overall_results.items():
        batch_input, ordered_batch_info = build_batch_input_v2(
            v, paper_data, in_name2pid, out_name2pid, args)

        all_inputs = {}
        all_inputs['inputs'] = batch_input
        all_inputs['ordered_batch_info'] = ordered_batch_info
        all_inputs['aid1'] = v['aid1']
        all_inputs['aid2'] = v['aid2']
        all_inputs['name'] = v['name']

        batch_data.append(all_inputs)

    return batch_data


def main_inference(args):
    """主推理函数"""
    print("加载论文数据...")
    paper_data = json.load(open(args.paper_dict))

    print("加载内部源和外部源数据...")
    in_name2pid = json.load(open(args.in_name2pid))
    out_name2pid = json.load(open(args.out_name2pid))

    clean_in_name2pid = json.load(open(args.clean_in_name2pid))
    clean_out_name2pid = json.load(open(args.clean_out_name2pid))

    for k1, v1 in in_name2pid.items():
        if k1 not in clean_in_name2pid:
            clean_in_name2pid[k1] = v1

    for k1, v1 in out_name2pid.items():
        if k1 not in clean_out_name2pid:
            clean_out_name2pid[k1] = v1

    in_aid2pid = {}
    for name, aids in clean_in_name2pid.items():
        for aid in aids:
            in_aid2pid[aid] = aids[aid]

    out_aid2pid = {}
    for name, aids in clean_out_name2pid.items():
        for aid in aids:
            out_aid2pid[aid] = aids[aid]

    print(f"加载预测文件: {args.src}")
    predict_file = json.load(open(args.src))

    print("按 aid1-aid2 对分组（保留批次结构）...")
    overall_results = defaultdict(dict)

    for item in predict_file:
        aid1 = item['aid1']
        aid2 = item['aid2']
        name = item.get('name', 'unknown')
        key = f"{aid1}-{aid2}"

        try:
            if "</think>" in item['content']:
                item['content'] = item['content'].split("</think>")[-1].strip()
            res = extract_and_parse_json(item['content'])
        except:
            res = {}
            for i in range(len(item['pids'])):
                res[str(i)] = 0.5

        if len(item['pids']) != len(res.keys()):
            print(f"警告: aid1={aid1}, aid2={aid2} 的结果长度不匹配，使用默认分数 0.5")
            res = {str(i): 0.5 for i in range(len(item['pids']))}

        preds = [v for k, v in sorted(res.items(), key=lambda x: int(x[0]))]

        if 'batches' not in overall_results[key]:
            overall_results[key]['aid1'] = aid1
            overall_results[key]['aid2'] = aid2
            overall_results[key]['name'] = name
            overall_results[key]['batches'] = []

        overall_results[key]['batches'].append({
            'pids': item['pids'],
            'preds': preds,
        })

    total_papers = sum(
        sum(len(b['pids']) for b in v['batches'])
        for v in overall_results.values()
    )
    print(f"共有 {len(overall_results)} 个 aid1-aid2 对")
    print(f"总论文数: {total_papers}")

    print("准备推理数据...")
    inference_data = prepare_inference_data(
        overall_results, paper_data, in_aid2pid, out_aid2pid, args)

    print(f"使用模型: {args.model}")
    llm_client = LLMClass()

    print("开始批量推理...")
    results = llm_client.process_batch_with_pool(
        messages=[i['inputs'] for i in inference_data],
        model=args.model,
        system_prompt=system_prompt,
        pool_size=args.pool_size,
        validate_json=True
    )

    print("合并结果...")
    for inf, res in zip(inference_data, results):
        inf['content'] = res.content
        inf['reasoning_content'] = res.reasoning_content if res.reasoning_content else ""
        inf['json_data'] = res.json_data if res.json_data else {}

    os.makedirs('refine_output', exist_ok=True)
    save_name = f'refine_output/batch_refine_v2_{args.src.split("/")[-1].split(".")[0]}_{args.model}_{args.save_name}.json'

    print(f"保存结果到: {save_name}")
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(inference_data, f, indent=4, ensure_ascii=False)

    print("完成!")

    print("解析并保存扁平化结果...")
    flat_results = []

    for item in inference_data:
        aid1 = item['aid1']
        aid2 = item['aid2']
        name = item['name']
        ordered_batch_info = item['ordered_batch_info']

        try:
            if "</think>" in item['content']:
                content_text = item['content'].split("</think>")[-1].strip()
            else:
                content_text = item['content']

            res = extract_and_parse_json(content_text)

            for batch_info in ordered_batch_info:
                batch_idx = batch_info['batch_idx']
                batch_key = f"batch_{batch_idx}"
                pids = batch_info['pids']
                old_preds = batch_info['preds']

                batch_res = res.get(batch_key, {})

                for i, (pid, old_pred) in enumerate(zip(pids, old_preds)):
                    new_pred = float(batch_res.get(str(i), old_pred))
                    flat_results.append({
                        'name': name,
                        'aid1': aid1,
                        'aid2': aid2,
                        'pid': pid,
                        'batch_idx': batch_idx,
                        'old_pred': old_pred,
                        'new_pred': new_pred,
                    })

        except Exception as e:
            print(f"解析 {aid1}-{aid2} 时出错: {e}，使用旧预测值")
            for batch_info in ordered_batch_info:
                batch_idx = batch_info['batch_idx']
                for pid, old_pred in zip(batch_info['pids'], batch_info['preds']):
                    flat_results.append({
                        'name': name,
                        'aid1': aid1,
                        'aid2': aid2,
                        'pid': pid,
                        'batch_idx': batch_idx,
                        'old_pred': old_pred,
                        'new_pred': old_pred,
                    })

    flat_save_name = f'refine_output/batch_refine_v2_flat_{args.src.split("/")[-1].split(".")[0]}_{args.model}_{args.save_name}.json'
    print(f"保存扁平化结果到: {flat_save_name}")
    with open(flat_save_name, 'w', encoding='utf-8') as f:
        json.dump(flat_results, f, indent=4, ensure_ascii=False)

    print(f"\n统计信息:")
    print(f"  总论文数: {len(flat_results)}")
    print(f"  aid1-aid2 对数: {len(inference_data)}")

    score_changes = [abs(r['new_pred'] - r['old_pred']) for r in flat_results]
    if score_changes:
        print(f"  平均分数变化: {sum(score_changes) / len(score_changes):.4f}")
        print(f"  最大分数变化: {max(score_changes):.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Batch Refine V2 for WhoIsWho Dataset")

    parser.add_argument("--src", type=str, required=True,
                        help="输入文件路径，即 refine_crossnd.py 的输出结果")
    parser.add_argument("--model", type=str, required=True,
                        help="使用的模型名称")
    parser.add_argument("--save_name", type=str, required=True,
                        help="保存结果的名称后缀")

    parser.add_argument("--paper_dict", type=str, required=True,
                        help="论文字典文件路径 (pub_dict.json)")
    parser.add_argument("--in_name2pid", type=str, required=True,
                        help="内部源数据文件路径 (name_aid_to_pids_in.json)")
    parser.add_argument("--out_name2pid", type=str, required=True,
                        help="外部源数据文件路径 (name_aid_to_pids_out.json)")
    parser.add_argument("--clean_in_name2pid", type=str, required=True,
                        help="清洗后的内部源数据文件路径 (cleaned_name_aid_to_pids_in.json)")
    parser.add_argument("--clean_out_name2pid", type=str, required=True,
                        help="清洗后的外部源数据文件路径 (cleaned_name_aid_to_pids_out.json)")

    parser.add_argument("--feature_list", type=str,
                        default="title_author_organization",
                        help="使用的特征列表，用下划线分隔")
    parser.add_argument("--pool_size", type=int, default=100,
                        help="线程池大小")

    args = parser.parse_args()

    print("=" * 80)
    print("Batch Refine V2 for WhoIsWho Dataset")
    print("=" * 80)
    print(f"输入文件: {args.src}")
    print(f"模型: {args.model}")
    print(f"保存名称: {args.save_name}")
    print(f"特征列表: {args.feature_list}")
    print(f"线程池大小: {args.pool_size}")
    print("=" * 80)

    main_inference(args)
