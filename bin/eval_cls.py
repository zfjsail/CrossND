from datasets import load_dataset
from transformers import Qwen3ForSequenceClassification, AutoTokenizer
from peft import PeftModel
import logging
import os
import json
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, average_precision_score
from collections import defaultdict

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', type=str, default="/workspace/pangyunhe/models/models/Qwen/Qwen3-4B-Base")
parser.add_argument('--model_path', type=str, default="/workspace/pangyunhe/project/crossnd/qwen3b-cls-lora") 
parser.add_argument('--test_data_path', type=str, default="./data/kddcup_cls_test.json")
parser.add_argument('--output_file', type=str, default="./cls_evaluation_results.json")
parser.add_argument('--predictions_file', type=str, default="./model_predictions.json")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_length', type=int, default=20000)
args = parser.parse_args()

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_dataset(dataset, dataset_name="dataset"):
    """
    处理数据集，确保数据符合序列分类任务的输入格式
    
    序列分类任务的数据格式应为：
    {"text": "输入文本", "label": 标签值}
    
    Args:
        dataset: 要处理的数据集
        dataset_name: 数据集名称，用于日志输出
        
    Returns:
        处理后的数据集
    """
    # 确定数据集切片
    split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
    
    # 获取数据集大小
    dataset_size = len(dataset[split_name])
    logger.info(f"{dataset_name} 数据集大小: {dataset_size} 个样本")
    
    # 验证数据格式是否符合要求
    sample = dataset[split_name][0]
    if 'text' not in sample or 'label' not in sample:
        raise ValueError(f"{dataset_name} 数据集必须包含'text'和'label'字段")
    
    # 检查所有样本的格式
    for idx in range(min(100, dataset_size)):  # 只检查前100个样本以节省时间
        sample = dataset[split_name][idx]
        if not isinstance(sample['text'], str):
            raise ValueError(f"{dataset_name} 数据集中第{idx}个样本的'text'字段必须是字符串")
        if not isinstance(sample['label'], (int, float)):
            raise ValueError(f"{dataset_name} 数据集中第{idx}个样本的'label'字段必须是数字")
    
    # 统计标签分布
    labels = [sample['label'] for sample in dataset[split_name]]
    unique_labels = set(labels)
    label_counts = {label: labels.count(label) for label in unique_labels}
    logger.info(f"{dataset_name} 数据集标签分布: {label_counts}")
    
    return dataset

def preprocess_function(examples, tokenizer, max_length=20000):
    """
    对文本数据进行预处理和tokenize
    
    Args:
        examples: 数据集样本
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        处理后的数据
    """
    # 对文本进行tokenize
    tokenized = tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 添加标签
    tokenized["labels"] = examples["label"]
    
    return tokenized

def predict_batch(model, batch_input_ids, batch_attention_mask):
    """批量进行预测"""
    device = model.device
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_mask = batch_attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        
    logits = outputs.logits
    # 转换为float32类型，以避免BFloat16类型的问题
    # logits = logits.to(torch.bfloat16)
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()  # 获取概率分布
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    return predictions, probabilities

def evaluate_model():
    """加载模型、执行推理并评估性能"""
    # 加载评估数据集
    logger.info("正在加载评估数据集...")
    eval_dataset = load_dataset("json", data_files=args.test_data_path)
    eval_dataset = process_dataset(eval_dataset, "评估")
    
    # 检查数据集结构
    split_name = 'train' if 'train' in eval_dataset else list(eval_dataset.keys())[0]
    sample = eval_dataset[split_name][0]
    logger.info(f"数据集示例: {sample}")
    logger.info(f"数据集特征: {eval_dataset[split_name].features}")
    
    # 加载分词器和模型
    logger.info(f"正在加载分词器: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        logger.info("设置pad_token为eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"正在加载基础模型: {args.base_model_path}")
    # 获取标签数量
    num_labels = len(set([sample['label'] for sample in eval_dataset['train']]))
    logger.info(f"分类任务标签数量: {num_labels}")
    
    base_model = Qwen3ForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 确保模型的pad_token_id与tokenizer一致
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info(f"正在加载微调后的模型: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    
    # 将模型移到 CUDA 0（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model = model.to(device)
    
    model.eval()
    
    # 预处理数据集
    logger.info("对评估数据集进行预处理...")
    batch_size = args.batch_size
    all_predictions = []
    all_labels = []
    all_probs = []
    all_sample_ids = []
    all_author_ids = []
    
    # 遍历数据集
    split_name = 'train' if 'train' in eval_dataset else list(eval_dataset.keys())[0]
    eval_data = eval_dataset[split_name]
    
    for i in tqdm(range(0, len(eval_data), batch_size), desc="批量预测进度"):
        # 获取当前批次
        batch_samples = eval_data[i:min(i+batch_size, len(eval_data))]
        
        # 提取文本、标签和作者ID（如果有）
        batch_texts = batch_samples["text"]
        batch_labels = batch_samples["label"]
        
        # 尝试获取样本ID和作者ID（如果数据集中有这些字段）
        batch_sample_ids = []
        batch_author_ids = []
        
        # 修复：batch_samples是一个包含多个特征的字典，不能按索引访问单个样本
        # 通过特征名称直接访问并获取相应长度的数据
        for j in range(len(batch_texts)):
            # 给每个样本分配一个默认ID
            sample_id = str(i + j)
            # 默认作者ID
            author_id = "unknown"
            
            # 处理嵌套的id字段
            if "id" in batch_samples:
                if j < len(batch_samples["id"]):
                    sample_id_info = batch_samples["id"][j]
                    # 如果id是字典，尝试从中提取aid1
                    if isinstance(sample_id_info, dict):
                        # 从id字典中提取作者ID，优先使用aid1
                        if "aid1" in sample_id_info:
                            author_id = sample_id_info["aid1"]
                        elif "aid2" in sample_id_info:
                            author_id = sample_id_info["aid2"]
                        elif "aid" in sample_id_info:
                            author_id = sample_id_info["aid"]
                        
                        # 从id字典中提取样本ID，优先使用pid
                        if "pid" in sample_id_info:
                            sample_id = sample_id_info["pid"]
                        else:
                            # 如果没有pid，使用整个id字典转换为字符串作为样本ID
                            sample_id = str(i + j)
                    else:
                        # 如果id不是字典，直接使用它作为样本ID
                        sample_id = str(sample_id_info)
            
            # 如果没有从id字段找到作者ID，尝试其他可能的字段
            if author_id == "unknown":
                if "author_id" in batch_samples and j < len(batch_samples["author_id"]):
                    author_id = batch_samples["author_id"][j]
                elif "aid1" in batch_samples and j < len(batch_samples["aid1"]):
                    author_id = batch_samples["aid1"][j]
                elif "aid" in batch_samples and j < len(batch_samples["aid"]):
                    author_id = batch_samples["aid"][j]
            
            batch_sample_ids.append(sample_id)
            batch_author_ids.append(author_id)
        
        # 对文本进行tokenize
        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        
        # 进行预测，获取预测结果和概率
        batch_predictions, batch_probs = predict_batch(
            model, 
            encoded_inputs["input_ids"], 
            encoded_inputs["attention_mask"]
        )
        
        # 收集预测结果、概率和标签
        all_predictions.extend(batch_predictions.tolist())
        all_probs.extend(batch_probs.tolist())
        all_labels.extend(batch_labels)
        all_sample_ids.extend(batch_sample_ids)
        all_author_ids.extend(batch_author_ids)
        
        # 定期输出日志
        if (i + batch_size) % (batch_size * 10) == 0 or i == 0:
            logger.info(f"已完成 {min(i+batch_size, len(eval_data))}/{len(eval_data)} 个样本的预测")
    
    # 保存所有的预测结果
    prediction_data = []
    for idx in range(len(all_labels)):
        prediction_data.append({
            "id": all_sample_ids[idx],
            "author_id": all_author_ids[idx],
            "label": all_labels[idx],
            "prediction": all_predictions[idx],
            "probabilities": all_probs[idx]
        })
    
    # 打印一些数据样本，用于调试
    logger.info(f"预测结果样本数量: {len(prediction_data)}")
    if len(prediction_data) > 0:
        logger.info(f"第一个预测结果样本: {prediction_data[0]}")
    
    # 保存预测结果到文件
    logger.info(f"正在保存预测结果到 {args.predictions_file}")
    with open(args.predictions_file, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, ensure_ascii=False, indent=2)
    
    # 计算标准评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # 生成详细的分类报告
    class_report = classification_report(all_labels, all_predictions, output_dict=True)
    
    # 打印结果
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"宏平均 F1 (Macro F1): {macro_f1:.4f}")
    logger.info(f"加权平均 F1 (Weighted F1): {weighted_f1:.4f}")
    
    # 按照eval.py中的方法计算AUC和MAP
    author_metrics, final_auc, final_map, n_authors = calculate_author_metrics(all_author_ids, all_predictions, all_labels)
    
    logger.info(f"评估的作者数量: {n_authors}")
    logger.info(f"宏平均 MAP: {final_map:.4f}")
    logger.info(f"宏平均 AUC: {final_auc:.4f}")
    
    # 保存结果
    evaluation_results = {
        'metrics': {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'MAP': final_map,
            'AUC': final_auc,
            'n_authors': n_authors,
            'avg_type': 'macro'
        },
        'classification_report': class_report,
        'author_metrics': author_metrics
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    return accuracy, macro_f1, weighted_f1, final_map, final_auc

def calculate_author_metrics(author_ids, predictions, labels):
    """
    按照作者ID计算MAP和AUC指标
    
    Args:
        author_ids: 作者ID列表
        predictions: 预测结果列表
        labels: 真实标签列表
    
    Returns:
        author_metrics: 每个作者的评估指标
        final_auc: 最终的宏平均AUC
        final_map: 最终的宏平均MAP
        n_authors: 有效作者数量
    """
    # 按作者分组的预测和标签
    author_predictions = defaultdict(list)
    author_labels = defaultdict(list)
    
    # 将预测结果和标签按作者ID分组
    for author_id, pred, label in zip(author_ids, predictions, labels):
        author_predictions[author_id].append(pred)
        author_labels[author_id].append(label)
    
    # 计算宏平均AUC和MAP
    map_sum = 0.0
    auc_sum = 0.0
    n_authors = 0
    
    # 每个作者的评估结果
    author_metrics = {}
    
    logger.info(f"开始按作者计算指标，共有 {len(author_predictions)} 个作者")
    
    for author_id, labels in author_labels.items():
        predictions = author_predictions[author_id]
        
        if len(labels) == 0 or len(predictions) == 0:
            logger.warning(f"作者 {author_id} 没有足够的数据进行评估")
            continue
            
        # 注意：原始标签中1表示正样本，0表示负样本
        # 根据eval.py的处理方式调整标签和预测值
        adjusted_predictions = [1-p for p in predictions]
        adjusted_labels = [1-l for l in labels]
        
        # 计算正样本比例
        pos_ratio = 1 - (sum(adjusted_labels) / len(adjusted_labels))
        
        # 检查正样本比例
        logger.debug(f"作者 {author_id} 的正样本比例: {pos_ratio}, 样本数: {len(labels)}")
        
        # 跳过正样本比例≥50%或全为负样本的作者（与eval.py保持一致）
        if pos_ratio == 1 or pos_ratio < 0.5:
            logger.info(f"作者 {author_id} 的正样本比例不符合要求: {pos_ratio}, 跳过评估")
            continue
        
        n_authors += 1
        sample_count = len(labels)
        
        # 计算AP和AUC
        try:
            # 确保每个类别至少有一个样本
            unique_labels = set(adjusted_labels)
            if len(unique_labels) < 2:
                logger.warning(f"作者 {author_id} 的样本只有一个类别，跳过评估")
                continue
                
            author_ap = average_precision_score(adjusted_labels, adjusted_predictions)
            author_auc = roc_auc_score(adjusted_labels, adjusted_predictions)
            
            # 宏平均: 直接累加每个作者的指标，不使用样本数量加权
            map_sum += author_ap
            auc_sum += author_auc
            
            author_metrics[author_id] = {
                'AP': author_ap,
                'AUC': author_auc,
                'sample_count': sample_count,
                'pos_ratio': pos_ratio
            }
            
            logger.debug(f"作者 {author_id} - AUC: {author_auc:.4f}, AP: {author_ap:.4f}, 样本数: {sample_count}")
            
        except Exception as e:
            logger.warning(f"计算作者 {author_id} 的指标时出错: {e}")
    
    logger.info(f"完成评估的有效作者数量: {n_authors}")
    
    # 计算最终宏平均 - 对每个作者的指标取算术平均
    final_map = map_sum / n_authors if n_authors > 0 else 0
    final_auc = auc_sum / n_authors if n_authors > 0 else 0
    
    return author_metrics, final_auc, final_map, n_authors

if __name__ == "__main__":
    accuracy, macro_f1, weighted_f1, map_score, auc_score = evaluate_model()
    logger.info(f"最终评估结果:")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"宏平均 F1 (Macro F1): {macro_f1:.4f}")
    logger.info(f"加权平均 F1 (Weighted F1): {weighted_f1:.4f}")
    logger.info(f"宏平均 MAP: {map_score:.4f}")
    logger.info(f"宏平均 AUC: {auc_score:.4f}") 