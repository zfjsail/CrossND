from utils import load_json
from collections import defaultdict as dd
import utils
import settings

# def gen_pa_pair_to_input_mat_dict_mp(emb_type="w2v"):
#     # file_dir = settings.DATASET_DIR
#     file_dir = "/home/zhipuai/zhangfanjin-15T/pyh/pangyunhe1/git/crossnd-202211/data/kddcup"
#     total = 0
#     err = 0
#     # if settings.data_source == "whoiswho" or settings.data_source == "kddcup":
#     name_aid_to_pids_in = load_json(file_dir, "aminer_name_aid_to_pids.json")
#     name_aid_to_pids_out = load_json(file_dir, "name_aid_to_pids_out_noisy_0.1.json")
#     # pairs_valid = load_json(file_dir, "eval_na_checking_triplets_valid.json")
#     # pairs_test = load_json(file_dir, "eval_na_checking_triplets_test.json")
#     # pos_pairs_train = load_json(file_dir, "positive_paper_author_pairs.json")
#     # neg_pairs_train = load_json(file_dir, "negative_paper_author_pairs.json")
#     # paper_dict = load_json(file_dir, "paper_dict_mag.json")  # str: dict
#     # else:
#     #     raise NotImplementedError

#     name_pid_to_aid_out = dd(dict)
#     for name in name_aid_to_pids_out:
#         cur_name_dict = name_aid_to_pids_out[name]
#         for aid in cur_name_dict:
#             cur_pids = cur_name_dict[aid]
#             for pid in cur_pids:
#                 name_pid_to_aid_out[name][pid] = aid
#     breakpoint()
#     triplets = []
#     for i, name in enumerate(name_aid_to_pids_in):
#         # logger.info("name %d: %s", i, name)
#         cur_name_dict = name_aid_to_pids_in[name]
#         for aid in cur_name_dict:
#             cur_pids = cur_name_dict[aid]
#             for pid in cur_pids:
#                 total += 1
#                 if pid in name_pid_to_aid_out.get(name, {}):
#                     aid_map = name_pid_to_aid_out[name][pid]
#                     triplets.append({"aid1": aid, "aid2": aid_map, "pid": pid, "name": name})
#                 else:
#                     err += 1
#     print(err, total)

# gen_pa_pair_to_input_mat_dict_mp()
def build_train_triplets():
    """
    构建训练数据集的三元组(aid1, aid2, pid)和二元组(aid1, pid)
    
    返回:
        train_samples: 包含训练集三元组和二元组的列表
    """
    file_dir = "/workspace/pangyunhe/project/crossnd/data/datasets--canalpang--crossnd/snapshots/fe8fc58f86dce28120151da0f110e286b947e7ba/kddcup"

    # 加载数据
    name_aid_to_pids_in = utils.load_json(file_dir, "name_aid_to_pids_in.json")
    name_aid_to_pids_out = utils.load_json(file_dir, "name_aid_to_pids_out.json")
    pos_pairs_train = utils.load_json(file_dir, "positive_paper_author_pairs.json")
    neg_pairs_train = utils.load_json(file_dir, "negative_paper_author_pairs.json")
    
    # 构建名称-论文ID到外部作者ID的映射
    name_pid_to_aid_out = dd(dict)
    for name in name_aid_to_pids_out:
        cur_name_dict = name_aid_to_pids_out[name]
        for aid in cur_name_dict:
            cur_pids = cur_name_dict[aid]
            for pid in cur_pids:
                name_pid_to_aid_out[name][pid] = aid
    
    # 构建训练集样本（三元组和二元组）
    train_samples = []
    n_pairs = len(pos_pairs_train) + len(neg_pairs_train)

    n_triplets = 0
    n_pairs_only = 0
    
    # 处理正样本和负样本
    for pair in pos_pairs_train + neg_pairs_train:
        pid = pair["pid"]
        aid1 = pair["aid"]
        name = pair["name"]
        is_positive = pair in pos_pairs_train
        
        sample = {
            "aid1": aid1, 
            "pid": pid, 
            "name": name,
            "label": 1 if is_positive else 0
        }
        
        # 检查论文是否在外部数据集中
        if pid in name_pid_to_aid_out.get(name, {}):
            aid2 = name_pid_to_aid_out[name][pid]
            sample["aid2"] = aid2  # 添加外部作者ID
            n_triplets += 1
        else:
            n_pairs_only += 1
        
        train_samples.append(sample)
    
    print(f"训练集原本有 {n_pairs} 个样本")
    print(f"训练集映射后有 {n_triplets} 个三元组")
    # 可选：保存样本到文件
    utils.dump_json(train_samples, settings.OUT_DATASET_DIR, "train_samples.json")
    return train_samples

res = build_train_triplets()

