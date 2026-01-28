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
                if -100 in labels:
                    labels = labels[labels != -100]
                #将yes_token_id 和no_token_id 转换为1和0
                labels[labels == self.model.YES_TOKEN_IDS] = 1
                labels[labels == self.model.NO_TOKEN_IDS] = 0
                labels = self.gather_function(torch.tensor(labels).tolist())
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
        
        # 将评估结果以JSON格式追加保存到res.txt文件
        res_txt_path = f'{self.args.output_dir}/res.txt'
        result_dict = {
            'save_path': self.args.output_dir,
            'metric_key_prefix': metric_key_prefix,
            'num_samples': num_samples,
            'metrics': metrics
        }
        with open(res_txt_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=2) + '\n')
        
        if all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.mean(all_losses).item()
        
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"): 
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
    try:
        labels = torch.cat(flatten_metadata(labels)).tolist()
    except:
        labels = flatten_metadata(labels)
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
        # 'n_authors': n_authors
    }


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
            new_num_turn = min(2 ** self.internal_epoch, self.max_num_turn)
        elif self.schedule_type == "linear":
            new_num_turn = min(self.internal_epoch + 1, self.max_num_turn)
        elif self.schedule_type == "constant":
            new_num_turn = self.max_num_turn
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
    use_label_token: bool = field(
        default=True,
        metadata={"help": "是否使用label token"}
    )
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
        default=0.5
    )
    author_sim_lower_bound: float = field(
        default=None
    )
    lora_path: str = field(
        default=None,
        metadata={"help": "预训练的LoRA路径"}
    )
    max_seq_length: int = field(
        default=None,
        metadata={"help": "最大序列长度"}
    )
    num_turn_schedule_type: str = field(
        default=None,
        metadata={"help": "num_turn增长模式: 'exponential'(1,2,4,8...), 'linear'(1,2,3,4...), 'custom'(自定义列表)"}
    )
    max_steps_per_epoch: int = field(
        default=150,
        metadata={"help": "每个 epoch 最多执行的步数（如果为 None 则不限制）"}
    )
    use_clean_data: bool = field(
        default=True,
        metadata={"help": "是否使用clean data"}
    )
    base_model_save_path : str = field(
        default=None,
        metadata={"help": "base model save path"}
    )
    psl_psi : float = field(
        default=1.0,
        metadata={"help": "psl psi"}
    )
    psl_lambda : float = field(
        default=0.5,
        metadata={"help": "psl lambda"}
    )

# class SaveBestCheckpointCallback(TrainerCallback):
#     def __init__(self, output_dir, best_ckpt_dir='best_ckpt'):
#         # 初始化，指定输出目录和最佳检查点目录
#         self.output_dir = os.path.abspath(output_dir)
#         self.best_ckpt_dir = os.path.abspath(os.path.join(output_dir, best_ckpt_dir))

#     def on_train_end(self, args, state, control, **kwargs):
#         # 在训练结束时，获取最佳检查点
#         best_checkpoint = state.best_model_checkpoint
#         if best_checkpoint is not None:
#             best_checkpoint = os.path.abspath(best_checkpoint)  # 转换为绝对路径

#             # 如果最佳检查点目录不存在，则创建
#             if not os.path.exists(self.best_ckpt_dir):
#                 os.makedirs(self.best_ckpt_dir)
            
#             # 复制最佳模型检查点的内容（仅文件，不含父目录）到 best_ckpt_dir
#             for file in os.listdir(best_checkpoint):
#                 src_file = os.path.join(best_checkpoint, file)
#                 dst_file = os.path.join(self.best_ckpt_dir, file)
#                 if os.path.isfile(src_file):
#                     shutil.copy(src_file, dst_file)

#             print(f"Best checkpoint copied to {self.best_ckpt_dir}.")

#             # 删除 output_dir 中所有名字带有 'checkpoint' 的文件夹
#             for sub_item in os.listdir(self.output_dir):
#                 sub_item_path = os.path.join(self.output_dir, sub_item)

#                 # 判断是否是名字包含 'checkpoint' 的目录
#                 if os.path.isdir(sub_item_path) and 'checkpoint' in sub_item:
#                     try:
#                         shutil.rmtree(sub_item_path)  # 递归删除目录
#                         print(f"Deleted checkpoint directory: {sub_item_path}")
#                     except Exception as e:
#                         print(f"Failed to delete {sub_item_path}. Reason: {e}")

