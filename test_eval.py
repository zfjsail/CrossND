import pickle
from train import *
from dataclasses import dataclass
import numpy as np
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载生成的预测文件
with open("eval_preds.pkl", "rb") as f:
    data = pickle.load(f)

# 创建EvalPrediction对象
def compute_metrics_v1(eval_preds):
    """
    计算评估指标，使用metadata和预测结果
    
    Args:
        eval_preds: 包含预测结果、标签和metadata的CrossNDEvalPrediction对象
        
    Returns:
        metrics: 包含各项评估指标的字典
    """
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    
    # 直接从CrossNDEvalPrediction对象中获取metadata
    metadata = eval_preds.metadata if hasattr(eval_preds, "metadata") else None

    i, j = len(metadata), len(metadata[0])
    pred_batch_size = len(predictions)
    if i != pred_batch_size and j == pred_batch_size:
        metadata = [list(row) for row in zip(*metadata)]

    # 递归展平嵌套列表，提取所有metadata字典
    def flatten_metadata(items, result=None):
        if result is None:
            result = []
        
        if items is None:
            return result
            
        if isinstance(items, dict):
            # 如果是字典，直接添加
            result.append(items)
        elif isinstance(items, (list, tuple)):
            # 如果是列表或元组，递归处理每个元素
            for item in items:
                flatten_metadata(item, result)
        
        return result
    all_pred, all_label, all_meta = [], [], []
    for pred, label, meta in zip(predictions, labels, metadata):
        for p,l,m in zip(pred,label,meta):
            all_pred.append(p)
            all_label.append(l)
            while isinstance(m, list):
                m = m[0]
            all_meta.append(m)

    # # 展平metadata
    # flattened_metadata = []
    # if metadata:
    #     for m in metadata:
    #         flatten_metadata(m, flattened_metadata)
    
    # 整理作者ID、预测结果和标签
    author_data = defaultdict(lambda: {'preds': [], 'labels': []})

    for p,l,m in zip(all_pred, all_label, all_meta):
        author_data[m['aid1']]['preds'].append(float(p))
        author_data[m['aid1']]['labels'].append(int(l))

    
    # 计算宏平均AUC和MAP
    map_sum = 0.0
    auc_sum = 0.0
    n_authors = 0
    
    # 每个作者的评估结果
    author_metrics = {}
    
    logger.info(f"开始按作者计算指标，共有 {len(author_data)} 个作者")
    
    for author_id, data in author_data.items():
        probs = data['preds']
        labels = data['labels']
        
        if len(labels) < 2 or len(probs) < 2:
            logger.warning(f"作者 {author_id} 没有足够的数据进行评估")
            continue
        adjusted_probs = [1-p for p in probs]
        adjusted_labels = [1-l for l in labels]
        
        # 计算正样本比例
        pos_ratio = 1 - (sum(adjusted_labels) / len(adjusted_labels))
        
        # 跳过正样本比例≥50%或全为负样本的作者
        if pos_ratio == 1 or pos_ratio < 0.5:
            logger.info(f"作者 {author_id} 的正样本比例不符合要求: {pos_ratio}, 跳过评估")
            continue
        

        author_ap = average_precision_score(adjusted_labels, adjusted_probs)
        author_auc = roc_auc_score(adjusted_labels, adjusted_probs)
        
        # 宏平均: 直接累加每个作者的指标
        map_sum += author_ap
        auc_sum += author_auc
        n_authors += 1
        
        author_metrics[author_id] = {
            'AP': author_ap,
            'AUC': author_auc,
            'sample_count': len(labels),
            'pos_ratio': pos_ratio
        }

    
    logger.info(f"完成评估的有效作者数量: {n_authors}")
    
    # 计算最终宏平均
    final_map = map_sum / n_authors if n_authors > 0 else 0
    final_auc = auc_sum / n_authors if n_authors > 0 else 0
    
    return {
        'MAP': float(final_map),
        'AUC': float(final_auc),
        'n_authors': n_authors
    }

if __name__ == "__main__":
    from trainer import compute_metrics
    metrics = compute_metrics(data)
    print(metrics)