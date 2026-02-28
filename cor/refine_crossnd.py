import json
import random
import re
import os
from collections import defaultdict, Counter
from argparse import ArgumentParser
from LLM import LLMClass
from utils import extract_and_parse_json, fetch_single_paper_input, fetch_paper_id, build_crossnd_input
from utils import multi_turn_prompt, crossnd_system_prompt, crossnd_global_prompt, prompt_crossnd
from LLM import LLMClass


def load_progress(progress_file):
    """加载进度文件"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            print(f'发现进度文件,已处理{len(progress_data["completed_results"])}个aid对')
            return progress_data
        except Exception as e:
            print(f'加载进度文件失败: {e}')
            return None
    return None


def save_progress(progress_file, completed_results, processed_keys, correction_data):
    """保存进度文件"""
    try:
        progress_data = {
            'completed_results': completed_results,
            'processed_keys': list(processed_keys),
            'total_items': len(correction_data)
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4, ensure_ascii=False)
        print(f'进度已保存: {len(completed_results)}/{len(correction_data)}')
    except Exception as e:
        print(f'保存进度失败: {e}')


def save_checkpoint(checkpoint_file, results):
    """保存检查点文件"""
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f'检查点已保存: {checkpoint_file}')
    except Exception as e:
        print(f'保存检查点失败: {e}')


def main(args):

    paper_data = json.load(open(args.paper_dict))
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
    clean_in_name2pid = in_aid2pid
    clean_out_name2pid = out_aid2pid

    predata = json.load(open(args.src, 'r', encoding='utf-8'))

    data = defaultdict(list)
    for item in predata:
        aid1, aid2, pid = item['aid1'], item['aid2'], item['pid']
        data[f'{aid1}-{aid2}'].append(item)
    for k,v in data.items():
        seen_pids = set()
        unique_items = []
        for item in v:
            if item['pid'] not in seen_pids:
                seen_pids.add(item['pid'])
                unique_items.append(item)
        data[k] = unique_items
    print(f'共有{len(data)}个aid1-aid2对')
    correction_data = build_crossnd_input(paper_data=paper_data, data=data, in_name2pid=clean_in_name2pid,
                                          out_name2pid=clean_out_name2pid, num_turn=20, feature_list='title_author_organization')

    save_path = f"refine_output/crossnd_{args.src.split('/')[-1].split('.')[0]}_{args.model}_{args.save_name}.json"
    progress_file = f"refine_output/progress_{args.src.split('/')[-1].split('.')[0]}_{args.model}_{args.save_name}.json"
    checkpoint_dir = "refine_output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    llm_client = LLMClass()

    progress_data = load_progress(progress_file)
    if progress_data:
        completed_results = progress_data['completed_results']
        processed_keys = set(progress_data['processed_keys'])
        print(f'从进度文件恢复,跳过已完成的{len(processed_keys)}个aid对')
    else:
        completed_results = []
        processed_keys = set()

    pending_data = []
    for idx, item in enumerate(correction_data):
        item_key = f"{item.get('aid1', '')}-{item.get('aid2', '')}"
        if item_key not in processed_keys:
            pending_data.append((idx, item))

    print(f'待处理: {len(pending_data)}个aid对')

    save_interval = 100
    processed_count = 0

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pool_size = args.pool_size

        print(f'\n开始流式处理 {len(pending_data)} 个任务 (线程池大小: {pool_size})...')

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_data = {}
            for idx, item in pending_data:
                future = executor.submit(
                    llm_client.call_llm,
                    args.model,
                    item['inputs'],
                    prompt_crossnd,
                    True
                )
                future_to_data[future] = (idx, item)

            for future in as_completed(future_to_data):
                original_idx, inf = future_to_data[future]

                try:
                    res = future.result()

                    inf['content'] = res.content
                    inf['reasoning_content'] = res.reasoning_content if res.reasoning_content else ""
                    inf['json_data'] = res.json_data if res.json_data else {}

                    completed_results.append(inf)
                    item_key = f"{inf.get('aid1', '')}-{inf.get('aid2', '')}"
                    processed_keys.add(item_key)

                    processed_count += 1

                    if processed_count % 10 == 0 or processed_count == len(pending_data):
                        print(
                            f'已完成: {processed_count}/{len(pending_data)} ({processed_count*100//len(pending_data)}%)')

                    if processed_count % save_interval == 0:
                        save_progress(progress_file, completed_results,
                                      processed_keys, correction_data)

                        checkpoint_file = f"{checkpoint_dir}/checkpoint_{args.src.split('/')[-1].split('.')[0]}_{args.model}_{args.save_name}_{len(completed_results)}.json"
                        save_checkpoint(checkpoint_file, completed_results)

                except Exception as e:
                    print(f'\n处理任务 {original_idx} 时出错: {e}')
                    continue

        if processed_count % save_interval != 0:
            save_progress(progress_file, completed_results,
                          processed_keys, correction_data)
            checkpoint_file = f"{checkpoint_dir}/checkpoint_{args.src.split('/')[-1].split('.')[0]}_{args.model}_{args.save_name}_{len(completed_results)}.json"
            save_checkpoint(checkpoint_file, completed_results)

        print('\n所有任务处理完成,保存结果...')

        result_dict = {}
        for r in completed_results:
            pids_key = '-'.join(sorted(r.get('pids', [])))
            key = f"{r.get('aid1', '')}-{r.get('aid2', '')}-{pids_key}"
            result_dict[key] = r
        
        final_results = []
        for item in correction_data:
            pids_key = '-'.join(sorted(item.get('pids', [])))
            key = f"{item.get('aid1', '')}-{item.get('aid2', '')}-{pids_key}"
            if key in result_dict:
                final_results.append(result_dict[key])
            else:
                print(f"警告: 未找到处理结果 {key}")
                final_results.append(item)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        print(f'最终结果已保存到: {save_path}, 共 {len(final_results)} 条记录')

        if os.path.exists(progress_file):
            os.remove(progress_file)
            print('进度文件已删除')

    except KeyboardInterrupt:
        print('\n\n检测到中断信号,保存当前进度...')
        save_progress(progress_file, completed_results,
                      processed_keys, correction_data)
        checkpoint_file = f"{checkpoint_dir}/checkpoint_interrupt_{args.src.split('/')[-1].split('.')[0]}_{args.model}_{args.save_name}.json"
        save_checkpoint(checkpoint_file, completed_results)
        print('进度已保存,下次运行将自动继续')
        raise
    except Exception as e:
        print(f'\n\n发生错误: {e}')
        print('保存当前进度...')
        save_progress(progress_file, completed_results,
                      processed_keys, correction_data)
        checkpoint_file = f"{checkpoint_dir}/checkpoint_error_{args.src.split('/')[-1].split('.')[0]}_{args.model}_{args.save_name}.json"
        save_checkpoint(checkpoint_file, completed_results)
        print('进度已保存,下次运行将自动继续')
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='your-model-name')
    parser.add_argument('--num_turn', type=int, default=20)
    parser.add_argument('--feature_list', type=str,
                        default="title_author_organization")
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--src', type=str, required=True,
                        help="输入三元组数据文件路径")
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--paper_dict', type=str, required=True,
                        help="论文字典文件路径 (pub_dict.json)")
    parser.add_argument('--in_name2pid', type=str, required=True,
                        help="内部源数据文件路径 (name_aid_to_pids_in.json)")
    parser.add_argument('--out_name2pid', type=str, required=True,
                        help="外部源数据文件路径 (name_aid_to_pids_out.json)")
    parser.add_argument('--clean_in_name2pid', type=str, required=True,
                        help="清洗后的内部源数据文件路径 (cleaned_name_aid_to_pids_in.json)")
    parser.add_argument('--clean_out_name2pid', type=str, required=True,
                        help="清洗后的外部源数据文件路径 (cleaned_name_aid_to_pids_out.json)")
    args = parser.parse_args()
    main(args)
