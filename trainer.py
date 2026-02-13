import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

# Importing relevant modules from the Transformers library
from transformers import Trainer, TrainerCallback
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging
# from transformers.trainer import Trainer
from transformers.trainer import *
from transformers.trainer_utils import EvalPrediction

# Importing utility functions and constants
from utils import CrossNDDataset, CrossNDCollator, LABEL_TOKEN

# Setup logging
logger = logging.get_logger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


# 自定义EvalPrediction类，支持metadata参数
class CrossNDEvalPrediction(EvalPrediction):
    """
    扩展的EvalPrediction类，支持metadata参数
    
    Parameters:
        predictions (`np.ndarray`): 模型的预测结果
        label_ids (`np.ndarray`): 目标标签
        inputs (`np.ndarray`, *optional*): 传递给模型的输入数据
        losses (`np.ndarray`, *optional*): 评估期间计算的损失值
        metadata (`list`, *optional*): 评估数据的元数据
    """
    def __init__(
        self,
        predictions: Union[np.ndarray, tuple[np.ndarray]],
        label_ids: Union[np.ndarray, tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, tuple[np.ndarray]]] = None,
        metadata: Optional[list] = None,
    ):
        super().__init__(predictions, label_ids, inputs, losses)
        self.metadata = metadata
        if self.metadata is not None:
            self.elements += (self.metadata,)


def calculate_author_metrics(author_ids, logits, labels):
    """
    按照作者ID计算MAP和AUC指标，使用logits而不是二值预测
    
    Args:
        author_ids: 作者ID列表
        logits: 模型输出的logits
        labels: 真实标签列表
    
    Returns:
        author_metrics: 每个作者的评估指标
        final_auc: 最终的宏平均AUC
        final_map: 最终的宏平均MAP
        n_authors: 有效作者数量
    """
    # 按作者分组的logits和标签
    author_logits = defaultdict(list)
    author_labels = defaultdict(list)
    
    # 将logits和标签按作者ID分组
    for author_id, logit, label in zip(author_ids, logits, labels):
        # 使用softmax获取类别1的概率
        probs = torch.softmax(torch.tensor(logit), dim=-1)[1].item()
        author_logits[author_id].append(probs)
        author_labels[author_id].append(label)
    
    # 计算宏平均AUC和MAP
    map_sum = 0.0
    auc_sum = 0.0
    n_authors = 0
    
    # 每个作者的评估结果
    author_metrics = {}
    
    logging.info(f"开始按作者计算指标，共有 {len(author_logits)} 个作者")
    
    for author_id, labels in author_labels.items():
        probs = author_logits[author_id]
        
        if len(labels) == 0 or len(probs) == 0:
            logging.warning(f"作者 {author_id} 没有足够的数据进行评估")
            continue
            
        # 注意：原始标签中1表示正样本，0表示负样本
        # 根据eval.py的处理方式调整标签和预测值
        adjusted_probs = [1-p for p in probs]
        adjusted_labels = [1-l for l in labels]
        
        # 计算正样本比例
        pos_ratio = 1 - (sum(adjusted_labels) / len(adjusted_labels))
        
        # 跳过正样本比例≥50%或全为负样本的作者
        if pos_ratio == 1 or pos_ratio < 0.5:
            logging.info(f"作者 {author_id} 的正样本比例不符合要求: {pos_ratio}, 跳过评估")
            continue
        
        n_authors += 1
        sample_count = len(labels)
        
        try:
            # 确保每个类别至少有一个样本
            unique_labels = set(adjusted_labels)
            if len(unique_labels) < 2:
                logging.warning(f"作者 {author_id} 的样本只有一个类别，跳过评估")
                continue
                
            author_ap = average_precision_score(adjusted_labels, adjusted_probs)
            author_auc = roc_auc_score(adjusted_labels, adjusted_probs)
            
            # 宏平均: 直接累加每个作者的指标
            map_sum += author_ap
            auc_sum += author_auc
            
            author_metrics[author_id] = {
                'AP': author_ap,
                'AUC': author_auc,
                'sample_count': sample_count,
                'pos_ratio': pos_ratio
            }
            
        except Exception as e:
            logging.warning(f"计算作者 {author_id} 的指标时出错: {e}")
    
    logging.info(f"完成评估的有效作者数量: {n_authors}")
    
    # 计算最终宏平均
    final_map = map_sum / n_authors if n_authors > 0 else 0
    final_auc = auc_sum / n_authors if n_authors > 0 else 0
    
    return author_metrics, final_auc, final_map, n_authors


def eval(model, eval_dataset, tokenizer, batch_size=10):
    """
    评估模型性能
    
    Args:
        model: 要评估的模型
        eval_dataset: 评估数据集
        tokenizer: 分词器
        batch_size: 批次大小
        
    Returns:
        evaluation_results: 包含各项评估指标的字典
    """
    logging.info("开始评估...")
    model.eval()
    device = next(model.parameters()).device
    
    all_logits = []
    all_labels = []
    all_author_ids = []
    
    # 创建数据加载器
    data_collator = CrossNDCollator(tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    # 进行预测
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="评估进度"):
            # 将数据移到对应设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 获取模型输出
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            
            # 只保留有效的预测和标签（非-100的位置）
            valid_mask = labels != -100
            valid_logits = logits[valid_mask].cpu().numpy()
            valid_labels = labels[valid_mask].cpu().numpy()
            
            all_logits.extend(valid_logits)
            all_labels.extend(valid_labels)
            
            # 获取作者ID（如果数据集中包含）
            if hasattr(batch, "metadata") and "aid1" in batch.metadata:
                all_author_ids.extend(batch.metadata["aid1"])
            else:
                # 如果没有作者ID，使用默认值
                all_author_ids.extend(["unknown"] * len(valid_labels))
    
    # 计算作者级别的指标
    author_metrics, final_auc, final_map, n_authors = calculate_author_metrics(
        all_author_ids, all_logits, all_labels
    )
    
    # 计算整体准确率（使用argmax）
    predictions = np.argmax(all_logits, axis=-1)
    accuracy = (predictions == np.array(all_labels)).mean()
    
    # 汇总评估结果
    evaluation_results = {
        'metrics': {
            'accuracy': accuracy,
            'MAP': final_map,
            'AUC': final_auc,
            'n_authors': n_authors,
            'avg_type': 'macro'
        },
        'author_metrics': author_metrics
    }
    
    return evaluation_results
class CrossNDTrainer_v2(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        重载评估循环，保留完整的输入数据用于评估，包括metadata
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # 使用Python列表存储结果，而不是在GPU上累积
        all_losses = []
        all_preds = []
        all_labels = []
        all_inputs = []
        all_metadata = []

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            # 立即将所有结果移至CPU并转换为float32类型
            if losses is not None:
                losses = losses.detach().to(dtype=torch.float32, device='cpu')
            
            if logits is not None:
                logits = logits.detach().to(dtype=torch.float32, device='cpu')
            
            if labels is not None:
                labels = labels.detach().to(device='cpu')
            
            # 获取主要输入名称，用于解码
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = None
            if "inputs" in args.include_for_metrics and main_input_name in inputs:
                inputs_decode = inputs[main_input_name].detach().to(device='cpu')
            
            # 收集metadata
            if "metadata" in inputs:
                metadata = self.accelerator.gather_for_metrics([inputs["metadata"]])
                all_metadata.append(metadata)

            if is_torch_xla_available():
                xm.mark_step()
            
            # 处理并存储结果到CPU列表
            if losses is not None:
                all_losses.append(losses.numpy())
            if logits is not None:
                logits = self.gather_function(logits)
                all_preds.append(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                all_labels.append(labels)
            # 每次迭代后强制清理GPU内存
            del losses, logits, labels, inputs_decode, metadata
            torch.cuda.empty_cache()

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        # self.gather_function = self.accelerator.gather_for_metrics
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        
        # # 将所有结果转换为numpy数组
        # all_preds_np = np.array(all_preds) if all_preds else np.array([])
        # all_labels_np = np.array(all_labels) if all_labels else np.array([])
        # all_losses_np = np.array(all_losses) if all_losses else np.array([])
        # all_inputs_np = np.array(all_inputs) if all_inputs else np.array([])
        os.makedirs(f'{self.args.output_dir}/res', exist_ok=True)
        with open(f'{self.args.output_dir}/res/{metric_key_prefix}.pkl','wb') as f:
            pickle.dump({'all_preds': all_preds, 'all_labels': all_labels, 'all_losses': all_losses, 'all_metadata': all_metadata}, f)
        # exit(0)
        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            # 使用自定义的CrossNDEvalPrediction类
            metrics = self.compute_metrics(
                CrossNDEvalPrediction(
                    predictions=all_preds, 
                    label_ids=all_labels, 
                    # inputs=all_inputs_np if all_inputs else None,
                    losses=all_losses if all_losses else None,
                    metadata=all_metadata if all_metadata else None
                )
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)
        
        # 只在主进程保存评估结果
        if self.args.local_rank <= 0 or not hasattr(self, "accelerator") or self.accelerator.is_main_process:
            logger.info(f"[DEBUG] Saving results to output_dir: {self.args.output_dir}")
            # 将评估结果以JSON格式追加保存到res.txt文件
            res_txt_path = f'{self.args.output_dir}/res.txt'
            result_dict = {
                'save_path': self.args.output_dir,
                'metric_key_prefix': metric_key_prefix,
                'num_samples': num_samples,
                'metrics': metrics
            }
            if metric_key_prefix != 'eval':
                with open(res_txt_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_dict, ensure_ascii=False, indent=2) + '\n')
            
            # 保存详细预测结果（包含metadata）
            try:
                # 使用与compute_metrics相同的flatten_metadata函数
                def flatten_metadata(meta):
                    """递归展平metadata"""
                    result = []
                    if not isinstance(meta, list):
                        # 如果是 dict，直接保存
                        result.append(meta)
                    elif isinstance(meta, list):
                        # 如果是 list，递归处理每个元素
                        for item in meta:
                            result.extend(flatten_metadata(item))
                    # 其他类型忽略
                    return result
                
                # 调试日志
                logger.info(f"开始处理预测结果保存 - {metric_key_prefix}")
                logger.info(f"all_preds 类型: {type(all_preds)}, 长度: {len(all_preds) if isinstance(all_preds, (list, tuple)) else 'N/A'}")
                logger.info(f"all_metadata 类型: {type(all_metadata)}, 长度: {len(all_metadata) if isinstance(all_metadata, (list, tuple)) else 'N/A'}")
                
                # 展平predictions - 使用与compute_metrics相同的方式
                # all_preds 结构: [[tensor([...])], [tensor([...])], ...]
                # 需要先展平列表结构，再 cat tensor
                flat_preds = torch.cat(flatten_metadata(all_preds)).tolist()
                logger.info(f"展平后的 predictions 数量: {len(flat_preds)}")
                
                # 展平metadata
                # all_metadata 结构: [[[[{...}, {...}]]], [[[{...}, ...]]], ...]
                # 需要递归展平到字典列表
                flat_metadata = flatten_metadata(all_metadata)
                logger.info(f"展平后的 metadata 数量: {len(flat_metadata)}")
                
                # 构建结果列表
                predictions_with_metadata = []
                min_len = min(len(flat_preds), len(flat_metadata))
                
                if min_len == 0:
                    logger.warning(f"预测或metadata为空，无法保存结果")
                else:
                    for i in range(min_len):
                        pred = flat_preds[i]
                        meta = flat_metadata[i]
                        if isinstance(meta, dict):
                            result_item = dict(meta)  # 复制metadata
                            result_item['pred'] = float(pred)  # 添加预测值
                            predictions_with_metadata.append(result_item)
                    
                    if len(predictions_with_metadata) > 0:
                        # 保存到JSON文件，如果文件存在则自动添加版本号
                        base_path = f'{self.args.output_dir}/predictions_with_labels_{metric_key_prefix}'
                        predictions_json_path = f'{base_path}.json'
                        
                        # 检查文件是否存在，如果存在则添加版本号
                        if os.path.exists(predictions_json_path):
                            version = 1
                            while os.path.exists(f'{base_path}_v{version}.json'):
                                version += 1
                            predictions_json_path = f'{base_path}_v{version}.json'
                        
                        with open(predictions_json_path, 'w', encoding='utf-8') as f:
                            json.dump(predictions_with_metadata, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"预测结果已保存到: {predictions_json_path}")
                        logger.info(f"共保存 {len(predictions_with_metadata)} 条预测结果")
                    else:
                        logger.warning(f"没有有效的metadata字典，无法保存结果")
            except Exception as e:
                import traceback
                logger.warning(f"保存预测结果时出错: {str(e)}")
                logger.warning(f"错误堆栈: {traceback.format_exc()}")
        
        if all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.mean(all_losses).item()
        
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):  # 保留metadata键不添加前缀
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # 返回标准的EvalLoopOutput
        return EvalLoopOutput(
            predictions=all_preds, 
            label_ids=all_labels, 
            metrics=metrics, 
            num_samples=num_samples
        )
def compute_metrics(eval_preds):
    """
    计算评估指标，使用metadata和预测结果
    
    Args:
        eval_preds: 包含预测结果、标签和metadata的CrossNDEvalPrediction对象
        
    Returns:
        metrics: 包含各项评估指标的字典
    """
    def flatten_metadata(meta):
        result = []
        if not isinstance(meta, list):
            # 如果是 dict，直接保存
            result.append(meta)
        elif isinstance(meta, list):
            # 如果是 list，递归处理每个元素
            for item in meta:
                result.extend(flatten_metadata(item))
        # 其他类型忽略
        return result
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    metadata = eval_preds.metadata if hasattr(eval_preds, "metadata") else None

    predictions = torch.cat(flatten_metadata(predictions)).tolist()
    labels = torch.cat(flatten_metadata(labels)).tolist()
    metadata = flatten_metadata(metadata)

    author_data = defaultdict(lambda: {'preds': [], 'labels': []})
    # 处理嵌套结构的预测结果、标签和元数据
    for pred, label,meta in zip(predictions, labels, metadata):

        aid = meta["aid1"]
        # 确保pred是标量
        # if type(probs) == list:
        #     author_data[aid]['preds'].extend(probs)
        #     author_data[aid]['labels'].extend(label)
        # else:
        author_data[aid]['preds'].append(pred)
        author_data[aid]['labels'].append(label)

    # 计算宏平均AUC和MAP
    maps = []
    aucs = []
    
    logger.info(f"开始按作者计算指标，共有 {len(author_data)} 个作者")
    n_authors = 0
    for author_id, data in author_data.items():
        probs = data['preds']
        labels = data['labels']
          
        # 注意：原始标签中1表示正样本，0表示负样本
        # 根据eval.py的处理方式调整标签和预测值

        # 计算正样本比例

        pos_ratio = (sum(labels) / len(labels))
        # 跳过正样本比例≥50%或全为负样本的作者
        if pos_ratio == 1 or pos_ratio < 0.5:
            continue

        adjusted_probs = [1-p for p in probs]
        adjusted_labels = [1-l for l in labels]
        author_ap = average_precision_score(adjusted_labels, adjusted_probs)
        author_auc = roc_auc_score(adjusted_labels, adjusted_probs)
        
        maps.append(author_ap)
        aucs.append(author_auc)
        n_authors += 1

    logger.info(f"完成评估的有效作者数量: {n_authors}")
    # 计算最终宏平均
    final_map = sum(maps) / len(maps)
    final_auc = sum(aucs) / len(aucs)
    auc_map = final_auc+final_map
    return {
        'MAP': float(final_map),
        'AUC': float(final_auc),
        'AUC_MAP': float(auc_map),
        'n_authors': n_authors
    }



def cal_auc_map(pred,label):
    from sklearn.metrics import roc_auc_score, average_precision_score
    preds,labels = [],[]
    num_profile = 0
    overall_auc = 0
    overall_map = 0
    for aid,itm in pred.items():
        preds.append(itm)
        labels.append(label[aid])
    
    for p,l in zip(preds,labels):
        pos_ratio = sum(l)/len(l)
        if pos_ratio ==1 or pos_ratio<0.5:
            continue
        p = [1-r for r in p]
        l = [1-r for r in l]
        cur_auc = roc_auc_score(l,p)
        cur_map = average_precision_score(l,p)
        num_profile += 1
        overall_auc+= cur_auc
        overall_map+= cur_map
    return overall_auc/num_profile , overall_map/num_profile

class NumTurnScheduler(TrainerCallback):
    """
    在每个 epoch 动态调整 dataset 的 num_turn，并限制每个 epoch 的最大训练步数
    """

    def __init__(self, schedule_type="exponential", max_steps_per_epoch=None, max_num_turn=None):
        self.schedule_type = schedule_type
        self.max_steps_per_epoch = max_steps_per_epoch
        self.internal_epoch =0
        self.trainer = None
        # 记录 Epoch 开始时的 step，初始化为 None 以便处理 Resume 情况
        self.epoch_start_global_step = None 
        self.max_num_turn = max_num_turn
    # def on_train_begin(self, args, state, control, **kwargs):
    #     self.trainer = kwargs.get("trainer")

    #     # --- 修复 Bug 2: 断点续训初始化 ---
    #     # 如果是断点续训，我们将当前 step 视为该(残缺) epoch 的起点
    #     # 避免 steps_in_epoch 计算错误导致立即停止
    #     self.epoch_start_global_step = state.global_step

    #     if state.is_local_process_zero:
    #         logger.info(
    #             f"[NumTurnScheduler] 启动 | "
    #             f"Schedule: {self.schedule_type} | "
    #             f"Max steps/epoch: {self.max_steps_per_epoch}"
    #         )

    def on_epoch_begin(self, args, state, control, **kwargs):
        
        self.epoch_start_global_step = state.global_step
        current_seed = args.seed + self.internal_epoch
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)
        
        # 计算新的 num_turn
        if self.schedule_type == "exponential":
            new_num_turn = min(4 ** self.internal_epoch, self.max_num_turn)
        elif self.schedule_type == "linear":
            new_num_turn = min(self.internal_epoch + 1, self.max_num_turn)
        else:
            raise NotImplementedError(f"Unsupported schedule_type: {self.schedule_type}")
        self.internal_epoch += 1
        if new_num_turn == 1:
            return

        if not self.trainer.train_dataset.num_turn == new_num_turn:
            self.trainer.train_dataset.rebuild_dataset(new_num_turn)
            self.trainer._train_dataloader = None
        
        if not self.trainer.eval_dataset.num_turn == new_num_turn:
            self.trainer.eval_dataset.rebuild_dataset(new_num_turn)
            self.trainer._eval_dataloader = None
        
        if hasattr(self.trainer, "accelerator"):
            self.trainer.accelerator.wait_for_everyone()


    def on_step_end(self, args, state, control, **kwargs):
        # 防御性编程：防止 epoch_start_global_step 为 None
        start_step = self.epoch_start_global_step if self.epoch_start_global_step is not None else 0
        steps_in_epoch = state.global_step - start_step
        if steps_in_epoch >= self.max_steps_per_epoch:
            # print(f"达到每 epoch 最大步数: {self.max_steps_per_epoch}")
            if state.is_local_process_zero:
                logger.info(
                    f"[NumTurnScheduler] Epoch {state.epoch:.2f} | "
                    f"达到限制步数 {self.max_steps_per_epoch}，提前结束 Epoch"
                )
            control.should_epoch_stop = True
        return control



@dataclass
class DataArguments:
    """
    数据集相关的参数
    """
    data_dir: str = field(
        default="/home/zhipuai/zhangfanjin-15T/pyh/pangyunhe1/git/crossnd-202211/data/kddcup",
        metadata={"help": "数据目录"}
    )

    apply_chat_template: bool = field(
        default=True,
        metadata={"help": "是否应用聊天模板"}
    )
    dataset: str = field(
        default = "kddcup"
    )


@dataclass
class ModelArguments:
    """
    LoRA相关的参数
    """
    src: str = field(
        default= "/workspace/pangyunhe/project/crossnd/llm/data/alldata_nd_thr09.json",
        #/workspace/pangyunhe/project/crossnd/llm/data/fuzzyneg.json
        #/workspace/pangyunhe/project/crossnd/llm/data/fuzzyneg_csics.json
        #/workspace/pangyunhe/project/crossnd/llm/alldata_nd.json
        #/workspace/pangyunhe/project/crossnd/llm/alldata_crossnd.json
        metadata={"help":"数据集"}
    )
    model_path: str = field(
        default="/workspace/pangyunhe/models/Qwen/Qwen3-4B",
        metadata={"help": "模型路径"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout率"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "需要应用LoRA的模块"}
    )
    task_type: str = field(
        default="SEQ_CLS",
        metadata={"help": "任务类型"}
    )
    modules_to_save: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "需要保存的模块"}
    )
    loss_type: str = field(
        default='ce',
        metadata={"help": "是否使用label smoothing"}
    )
    use_binary_head: bool = field(
        default=False,
        metadata={"help": "是否使用二分类头"}
    )
    use_outer: bool = field(
        default=True,
        metadata={"help": "是否使用外部数据"}
    )
    num_turn: int = field(
        default=10,
        metadata={"help": "对话轮数"}
    )
    hybrid_train: bool = field(
        default=False,
    )
    paper_slct_num: int = field(
        default=100
    )
    label_thr: float = field(
        default=0.9
    )
    author_sim: float = field(
        default=1.0
    )
    lora_path: str = field(
        default=None,
        metadata={"help": "预训练的LoRA路径"}
    )
    max_seq_length: int = field(
        default=None,
        metadata={"help": "最大序列长度"}
    )
    freeze_header: bool = field(
        default=True
    )
    upsample: bool = field(
        default=True
    )
    use_hybrid_head: bool = field(  
        default=False
    )
    num_turn_schedule_type: str = field(
        default=None,
        metadata={"help": "num_turn增长模式: 'exponential'(1,2,4,8...), 'linear'(1,2,3,4...), 'custom'(自定义列表)"}
    )
    max_steps_per_epoch: int = field(
        default=150,
        metadata={"help": "每个 epoch 最多执行的步数（如果为 None 则不限制）"}
    )
    psi: float = field(
        default=1.0,
        metadata={"help": "PSL的psi参数"}
    )
    alpha: float = field(
        default=0.5,
        metadata={"help": "PSL的alpha参数"}
    )
    multiturn_path: str = field(
        default=None,
        metadata={"help": "多轮推理的预测分数文件路径，用于TTS优化"}
    )
    tts_strategy: str = field(
        default="random",
        metadata={"help": "TTS排序策略: random(随机), desc(降序), asc(升序), confidence(置信度), turn_interleave_confidence_first(置信度交替优先)等"}
    )
