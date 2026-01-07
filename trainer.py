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

class CrossNDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        # if output_dir is None:
        #     output_dir = self.args.output_dir
        # self.model.save_pretrained(output_dir)
        # if self.tokenizer is not None:
        #     self.tokenizer.save_pretrained(output_dir)
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = unwrap_model(self.model)
        
        #for debug
        state_dict = {}
        for k,v in model_to_save.named_parameters():
            for module_to_save in self.modules_to_save:
                if module_to_save in k:
                    state_dict[k] = v.to("cpu")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))

    def predict_without_train(        
        self,
        test_dataset=None,
        ground_truth = None,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        eval_result = None

        """
            borrow from trainer.train, to initialize model 
            ONLY FOR INFERENCE WITHOUT TRAINING
        """
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model
        
        """
            borrow from customized evaluate func
        """
        
        self._memory_tracker.start()
        args = self.args


        dataloader = self.get_eval_dataloader(test_dataset)

        # if self.is_deepspeed_enabled and self.deepspeed is None:
        #     _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            # model = (
            #     self.accelerator.prepare(model)
            #     if self.is_deepspeed_enabled
            #     else self.accelerator.prepare_model(model, evaluation_mode=True)
            # )
            model = self.accelerator.prepare_model(model, evaluation_mode=True)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        eval_result = []
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                if batch_size is None:
                    batch_size = observed_batch_size
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None)
            for b in range(labels.shape[0]):
                mask = labels[b] != -100
                cols = torch.where(mask)[0]
                res = []
                for i in range(len(cols)):
                    cur_label = labels[b][cols[i]]
                    cur_logits = logits[b][cols[i], :]
                    cur_pred = torch.softmax(cur_logits, dim=-1)[1]
                    res.append({
                        "predictions": cur_pred.item(),
                        "labels": cur_label.item(),
                        "metadata": inputs['metadata'][b][i]
                    })
                raw = self.accelerator.gather_for_metrics(res)
            if self.accelerator.is_main_process:
                eval_result.extend(raw)             
        if self.accelerator.is_main_process:
            overall_result = defaultdict(list)
            overall_label = defaultdict(list)
            for itm in eval_result:
                aid = itm['metadata']
                overall_result[aid].append(itm['predictions'])
                overall_label[aid].append(itm['labels'])
                if self.args.predict_saved_path is not None:
                    with open(self.args.predict_saved_path,'w') as f:
                        json.dump({"overall_result":overall_result,"overall_label":overall_label},f)
                
            AUCs,MAPs  = cal_auc_map(overall_result, overall_label)
            print(f" AUC and MAP before training is {AUCs}, {MAPs}")
            with open(os.path.join(args.output_dir,f"result.txt"), 'a') as f:
                f.write(f"step:0,epoch:0,AUC:{AUCs},MAP:{MAPs}\n") 
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()
        args = self.args
        dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        eval_result = []
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                if batch_size is None:
                    batch_size = observed_batch_size
            
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys)
            for b in range(labels.shape[0]):
                mask = labels[b] != -100
                cols = torch.where(mask)[0]
                res = []
                for i in range(len(cols)):

                    cur_label = labels[b][cols[i]]
                    cur_pred = logits[b][i]
                    # cur_pred = torch.softmax(cur_logits, dim=-1)[1]
                    res.append({
                        "predictions": cur_pred.item(),
                        "labels": cur_label.item(),
                        "metadata": inputs['metadata'][b][i]
                    })
                raw = self.accelerator.gather_for_metrics(res)
            if self.accelerator.is_main_process:
                eval_result.extend(raw)

        if self.accelerator.is_main_process:
            overall_result = defaultdict(list)
            overall_label = defaultdict(list)
            for itm in eval_result:
                aid = itm['metadata']['aid1']
                overall_result[aid].append(itm['predictions'])
                overall_label[aid].append(itm['labels'])
            AUCs,MAPs = cal_auc_map(overall_result, overall_label)

            #update best metric
            if self.state.best_metric is None:
                self.state.best_metric = AUCs
                self.state.best_model_checkpoint = os.path.join(self.args.output_dir,f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            elif AUCs > self.state.best_metric:
                self.state.best_metric = AUCs
                self.state.best_model_checkpoint = os.path.join(self.args.output_dir,f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            
            output = {'eval_AUC':AUCs,'eval_MAP':MAPs,'step':self.state.global_step,'epoch':self.state.epoch}
            self.log(output)
            with open(os.path.join(args.output_dir,f"result.txt"), 'a') as f:
                f.write(f"step:{self.state.global_step},epoch:{self.state.epoch},AUC:{AUCs},MAP:{MAPs}\n")       
            return output

    def get_logits(self, logits, inputs): #only for IND task
        labels_pos = torch.masked_select(torch.arange(inputs['input_ids'].shape[-1], device = self.model.device), inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        labels_pos -= 1
        
        if "glm4" in self.tokenizer.name_or_path:# for glm4 tokenizer
            YES_TOKEN_IDS, NO_TOKEN_IDS = self.tokenizer.encode('Yes',add_special_tokens = False)[0],self.tokenizer.encode('No',add_special_tokens = False)[0]
        else:# for llama tokenizer
            YES_TOKEN_IDS, NO_TOKEN_IDS = self.tokenizer.convert_tokens_to_ids(['Yes','No'])
        
        yes_logit,no_logit= logits[:,labels_pos,YES_TOKEN_IDS],logits[:,labels_pos,NO_TOKEN_IDS]
        logit = F.softmax(torch.concat([yes_logit,no_logit],dim=0),dim=0)[0]
        return logit

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        if self.args.remove_unused_columns:
            # 保留metadata列，即使它不是模型输入
            if hasattr(dataset, "column_names"):
                columns = dataset.column_names
            else:
                columns = list(dataset.features.keys())
            
            # 确保metadata列不会被移除
            if "metadata" in columns:
                self.args.include_for_metrics = list(set(self.args.include_for_metrics + ["metadata"]))
                
        return super()._remove_unused_columns(dataset, description)

class CrossNDTrainer_v2(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def save_model(self, output_dir=None, _internal_call=False):
    #     # if output_dir is None:
    #     #     output_dir = self.args.output_dir
    #     # self.model.save_pretrained(output_dir)
    #     # if self.tokenizer is not None:
    #     #     self.tokenizer.save_pretrained(output_dir)
        
    #     output_dir = output_dir if output_dir is not None else self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     model_to_save = unwrap_model(self.model)
        
    #     #for debug
    #     state_dict = {}
    #     for k,v in model_to_save.named_parameters():
    #         for module_to_save in self.modules_to_save:
    #             if module_to_save in k:
    #                 state_dict[k] = v.to("cpu")
    #     torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)
    #     torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))

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
    use_similarity: bool = field(
        default=True,
        metadata={"help": "是否使用相似性"}
    )
    use_clean_data: bool = field(
        default=True,
        metadata={"help": "是否使用clean data"}
    )
    base_model_save_path : str = field(
        default=None,
        metadata={"help": "base model save path"}
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

