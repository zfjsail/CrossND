from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
import sys
sys.path.append('.')
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.utils import (
    LossKwargs,
    can_return_tuple,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from utils import LABEL_TOKEN,EMBED_TOKEN,GRAPH_TOKEN,TRAINABLE_SPECIAL_TOKENS
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import GenerationMixin, Qwen3PreTrainedModel, Qwen3Model, LlamaPreTrainedModel, LlamaModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Cache
import logging
logger = logging.getLogger(__name__)
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

@dataclass
class INDModelWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    score: torch.FloatTensor = None

class MSELossWithIgnoreIdx(nn.Module):
    def __init__(self, ignore_index=-100):
        super(MSELossWithIgnoreIdx, self).__init__()
        self.ignore_index = ignore_index

        self.focal_alpha =0.25
        self.focal_gamma =2.0

    def forward(self, logits, labels):
        """
        Args:
            logits: [batch, seq_len, num_classes] 模型输出
            labels: [batch, seq_len, num_classes] 软标签
        """
        # 展平处理
        logits_flat = logits.view(-1, logits.size(-1))  # [batch * seq_len, num_classes]
        labels_flat = labels.view(-1, labels.size(-1))  # [batch * seq_len, num_classes]
        
        # 创建掩码
        mask = (labels_flat[:, 0] != self.ignore_index)
        
        # 如果没有有效位置，返回0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 只保留有效位置
        logits_valid = logits_flat[mask]
        labels_valid = labels_flat[mask]
        
        # 计算MSE损失
        return F.mse_loss(logits_valid, labels_valid, reduction='mean')


class Qwen3ForCrossND(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.is_binary_head = False
        self.use_hybrid_head = False
        self.loss_type = "ce"
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def kl_divergence_loss(self, logits, labels, ignore_value=-100.0, **kwargs):
        """
        计算带有ignore_value功能的KL散度损失，适用于浮点数的soft_labels
        
        Args:
            logits: [batch, seq_len, num_classes] 模型输出的原始logits
            labels: [batch, seq_len, num_classes] 软标签概率分布（包含ignore_value）
            ignore_value: 需要忽略的浮点数值（如-100.0）
        
        Returns:
            KL散度损失值
        """
        # 保存原始形状
        original_shape = logits.shape
        batch_size, seq_len, num_classes = original_shape
        
        # 展平处理
        logits_flat = logits.view(-1, num_classes)  # [batch * seq_len, num_classes]
        soft_labels_flat = labels.view(-1, num_classes)  # [batch * seq_len, num_classes]
        
        # 创建mask：检查soft_labels中是否包含ignore_value
        ignore_mask = torch.any(soft_labels_flat == ignore_value, dim=-1)  # [batch * seq_len]
        
        # 反转mask，得到有效位置的mask
        valid_mask = ~ignore_mask  # [batch * seq_len]
        
        # 应用mask，只保留需要计算损失的位置
        valid_logits = logits_flat[valid_mask]
        valid_soft_labels = soft_labels_flat[valid_mask]
        
        if valid_logits.size(0) == 0:
            # 如果没有有效样本，返回0损失
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 计算对数概率
        log_probs = F.log_softmax(valid_logits, dim=-1)
        
        # 计算KL散度损失
        kl_loss = F.kl_div(log_probs, valid_soft_labels, reduction='batchmean', log_target=False)
        
        return kl_loss
    
    def soft_ce_loss(self, logits, labels, ignore_value=-100.0, **kwargs):
        """
        二分类头的软标签交叉熵损失
        
        Args:
            logits: [batch_size, 2] - 二分类头输出，[no_logit, yes_logit]
            labels: [batch_size] - 软标签，范围[0,1]，ignore_value表示忽略
            ignore_value: float - 忽略的标签值，默认-100.0
        """
        # 创建有效样本的mask
        valid_mask = (labels != ignore_value)
        
        # 只处理有效样本
        valid_logits = logits[valid_mask]  # [valid_batch_size, 2]
        valid_soft_labels = labels[valid_mask]  # [valid_batch_size]
        
        # 构建软目标分布 [no_prob, yes_prob]
        soft_targets = torch.stack([
            1 - valid_soft_labels,  # no的目标概率
            valid_soft_labels       # yes的目标概率
        ], dim=-1)  # [valid_batch_size, 2]
        
        # 计算log softmax
        log_probs = F.log_softmax(valid_logits, dim=-1)
        
        # 计算交叉熵损失
        loss = -torch.sum(soft_targets * log_probs, dim=-1).mean()
        
        return loss
    
    def cross_entropy_loss(self, logits, labels,  vocab_size=None, **kwargs):
        """标准交叉熵损失"""
        # logits: [batch, seq_len, num_classes]
        # labels: [batch, seq_len]
        # if T is not None:
        #     logits = logits / T
        # 展平处理
        # logits_flat = logits.view(-1, logits.size(-1))  # [batch * seq_len, num_classes]
        # labels_flat = labels.view(-1)   
        # [batch * seq_len]
        # mask = (labels_flat != -100)
        # slct_logits = logits_flat[mask,:]
        # slct_labels = labels_flat[mask]
        # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        # loss = loss_fct(slct_logits, slct_labels.long())

        # 使用CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits.squeeze(0), labels.squeeze(0))
    
    def probabilistic_soft_logic_loss(self, logits, labels, phi, p_ain_sim, p_aout_sim, a1_a2_sim, vocab_size=None, **kwargs):  
        """
        计算概率软逻辑(PSL)交叉纠错损失
        
        参数:
            phi: 灵活的间隔超参数 ψ
            p_ain_sim: P(p, a^{in}) 的相似度，即当前模型预测的匹配概率 [batch_size]
            p_aout_sim: P_old(p, a^{out}) 的相似度，即旧模型预测的匹配概率 [batch_size] 
            a1_a2_sim: SIM(a^{in}, a^{out}) 的相似度 [batch_size]
            kwargs: 其他参数如阈值等
        
        返回:
            psl_loss: PSL损失值
        """
        psl_loss = torch.tensor(0.0, device=logits.device)
        return psl_loss

    def cross_entropy_loss_with_temperature(self, logits, labels, T=None,  vocab_size=None, **kwargs):
        """带Temperature标准交叉熵损失"""
        # logits: [batch, seq_len, num_classes]
        # labels: [batch, seq_len]
        if T is not None:
            T = 2-T
            logits = logits/T

        # 展平处理
        logits_flat = logits.view(-1, logits.size(-1))  # [batch * seq_len, num_classes]
        labels_flat = labels.view(-1)                   # [batch * seq_len]
        
        # 使用CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits_flat, labels_flat.long())    
    def margin_ranking_loss(self, logits, labels, margin=0.5, ignore_index=-100, **kwargs):
        """边际排序损失"""
        batch_size, seq_len, num_classes = logits.shape
        valid_mask = (labels != ignore_index)   
        valid_logits = logits[valid_mask]  # [valid_batch, 2]
        valid_labels = labels[valid_mask]  # [valid_batch]        
    
        valid_labels = torch.where(valid_labels==self.YES_TOKEN_IDS, 1, 0)
        
        if hasattr(self, 'is_binary_head') and self.is_binary_head:
            # 获取异常概率（第0类，因为0=异常）
            normal_probs = torch.softmax(valid_logits, dim=-1)[:, 0]
        else:      
            normal_probs = valid_logits[:, self.YES_TOKEN_IDS]
            abnormal_probs = valid_logits[:, self.NO_TOKEN_IDS]
            normalized_probs = torch.softmax(torch.stack([normal_probs,abnormal_probs]),dim=0)[0]
        
        # 分离异常和正常样本
        anomaly_mask = (valid_labels == 0)  # 0=异常
        normal_mask = (valid_labels == 1)   # 1=正常

        if not torch.any(anomaly_mask) or not torch.any(normal_mask):
            return torch.tensor(0.0, device=logits.device)       
         
        anomaly_scores = normalized_probs[anomaly_mask]
        normal_scores = normalized_probs[normal_mask]
        
        # 创建成对比较
        anomaly_scores_exp = anomaly_scores.unsqueeze(1)
        normal_scores_exp = normal_scores.unsqueeze(0)
        
        pairwise_diff = anomaly_scores_exp - normal_scores_exp + margin
        rank_loss = torch.clamp(pairwise_diff, min=0).mean()
        return rank_loss
    
    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0, ignore_index=-100, **kwargs):
        """
        Focal Loss实现，用于处理类别不平衡问题
        
        Args:
            logits: [batch * seq_len, num_classes] 模型输出的原始logits
            labels: [batch * seq_len] 目标标签
            alpha: 类别平衡参数，可以是标量或tensor
            gamma: focusing参数，控制难易样本的权重
            ignore_index: 忽略的标签索引
        
        Returns:
            focal loss值
        """
            
        # 展平处理
        logits_flat = logits.view(-1, logits.size(-1))  # [batch * seq_len, num_classes]
        labels_flat = labels.view(-1).long()            # [batch * seq_len]
        
        # 创建有效样本mask
        valid_mask = (labels_flat != ignore_index)
        
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 只处理有效样本
        valid_logits = logits_flat[valid_mask]  # [valid_samples, num_classes]
        valid_labels = labels_flat[valid_mask]  # [valid_samples]
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
        
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    # def cross_entropy_focal_loss(self, logits, labels, alpha=None, gamma=None, **kwargs):
    #     """
    #     结合交叉熵和focal loss的损失函数
    #     当alpha=1, gamma=0时退化为标准交叉熵损失
    #     """
    #     return self.focal_loss(logits, labels, alpha=alpha, gamma=gamma, **kwargs)

    def psl_loss(self, logits, labels, **kwargs):
        celoss = self.cross_entropy_loss(logits, labels, **kwargs)
        if 'p_in_sim' not in kwargs['metadata'][0][0] or 'p_out_sim' not in kwargs['metadata'][0][0] or 'author_sim' not in kwargs['metadata'][0][0]:
            print("PSL metadata missing, using CE loss only.")
            return celoss
        
        p_in_sim = [i['p_in_sim'] for i in kwargs['metadata'][0]]
        p_out_sim = [i['p_out_sim'] for i in kwargs['metadata'][0]]
        a1_a2_sim = [i['author_sim'] for i in kwargs['metadata'][0]]
        p_out_sim = torch.tensor(p_out_sim,dtype = logits.dtype).to(self.device)
        a1_a2_sim = torch.tensor(a1_a2_sim,dtype = logits.dtype).to(self.device)
        if self.is_binary_head:
            p_yes = torch.softmax(logits,dim=-1 )[0,labels[0]!=-100,1]
            labels = labels[labels!=-100]

        else:
            yes_logits = logits[:,(labels!=-100)[-1], self.YES_TOKEN_IDS]
            no_logits = logits[:,(labels!=-100)[-1], self.NO_TOKEN_IDS]

            labels = labels[labels!=-100]
            labels = (labels==self.YES_TOKEN_IDS).to(torch.long)
            probs = torch.softmax(torch.stack([no_logits, yes_logits], dim=-1), dim=-1)
            p_no, p_yes = probs[:,:,0 ], probs[:,:,1]

        PSI = 1
        eta1 = p_out_sim + a1_a2_sim - p_yes - PSI
        eta2 = torch.max(p_out_sim, 1-p_out_sim) -a1_a2_sim + p_yes - PSI

        eta_cs = torch.minimum(eta1, torch.zeros_like(eta1))
        eta_ics = torch.minimum(eta2, torch.zeros_like(eta2))

        loss = labels * eta_cs + (1-labels) * eta_ics
        return 0.5 * loss.mean() + 0.5 * celoss
    
    def psl_loss_v2(self, logits, labels, **kwargs):
        
        celoss = self.cross_entropy_loss(logits, labels, **kwargs)
        if 'p_in_sim' not in kwargs['metadata'][0][0] or 'p_out_sim' not in kwargs['metadata'][0][0] or 'author_sim' not in kwargs['metadata'][0][0]:
            print("PSL metadata missing, using CE loss only.")
            return celoss
        
        # p_in_sim = [i['p_in_sim'] for i in kwargs['metadata'][0]]
        p_out_sim = [i['p_out_sim'] for i in kwargs['metadata'][0]]
        a1_a2_sim = [i['author_sim'] for i in kwargs['metadata'][0]]
        p_out_sim = torch.tensor(p_out_sim,dtype = logits.dtype).to(self.device)
        a1_a2_sim = torch.tensor(a1_a2_sim,dtype = logits.dtype).to(self.device)
        if self.is_binary_head:
            p_yes = torch.softmax(logits,dim=-1 )[0,labels[0]!=-100,1]
            labels = labels[labels!=-100]
        else:
            yes_logits = logits[:,(labels!=-100)[-1], self.YES_TOKEN_IDS]
            no_logits = logits[:,(labels!=-100)[-1], self.NO_TOKEN_IDS]

            labels = labels[labels!=-100]
            labels = (labels==self.YES_TOKEN_IDS).to(torch.long)
            probs = torch.softmax(torch.stack([no_logits, yes_logits], dim=-1), dim=-1)
            p_no, p_yes = probs[:,:,0 ], probs[:,:,1]

        PSI = 1
        eta1 = p_out_sim + a1_a2_sim - p_yes - PSI
        eta2 = torch.max(p_out_sim, 1-p_out_sim) -a1_a2_sim + p_yes - PSI

        eta_cs = torch.minimum(eta1, torch.zeros_like(eta1))
        eta_ics = torch.minimum(eta2, torch.zeros_like(eta2))
        cs_ics_label = torch.tensor([i>0.5 for i in a1_a2_sim],dtype=torch.long).to(self.device)
        loss = cs_ics_label * eta_cs + (1-cs_ics_label) * eta_ics
        return 0.5 * loss.mean() + 0.5 * celoss
    def regression_mse_loss(logits, labels, mask_value=-100.0):
        """
        回归头 MSE loss
        logits: [B, 1] 或 [B, seq_len, 1]，原始未限制 logit
        labels: [B] 或 [B, seq_len]，范围 [0,1]
        """
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1).float()

        mask = (labels_flat != mask_value)
        if not torch.any(mask):
            return torch.tensor(0.0, device=logits.device)

        return F.mse_loss(logits_flat[mask], labels_flat[mask])
    def binary_cross_entropy_loss(self, logits, labels, ignore_index=-100, **kwargs):
        """
        二分类交叉熵损失
        提取YES_TOKEN_IDS和NO_TOKEN_IDS对应的logit，然后计算BCE loss
        
        logits: [batch, seq_len, vocab_size] - 模型输出logits
        labels: [batch, seq_len] - 标签
        """
        # 展平处理
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)  # [batch * seq_len, vocab_size]
        labels_flat = labels.view(-1)               # [batch * seq_len]
        
        # 创建有效样本的mask
        valid_mask = (labels_flat != ignore_index)
        
        # 只处理有效样本
        valid_logits = logits_flat[valid_mask]  # [valid_count, vocab_size]
        valid_labels = labels_flat[valid_mask]  # [valid_count]
        
        # 提取YES和NO对应的logit值
        yes_logits = valid_logits[:, self.YES_TOKEN_IDS]
        no_logits = valid_logits[:, self.NO_TOKEN_IDS]
        
        # 将两个logit堆叠，构成二分类logits [valid_count, 2]
        binary_logits = torch.stack([no_logits, yes_logits], dim=-1)
        
        # 将标签转换为二分类标签（YES_TOKEN_IDS对应1，其他对应0）
        binary_labels = (valid_labels == self.YES_TOKEN_IDS).long()
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(binary_logits, binary_labels)
        
        return loss

    def compute_loss(self, logits, labels, similarity = None, **kwargs):
        """根据loss_type计算相应的损失"""
        if self.loss_type == 'ls':
            return self.soft_ce_loss(logits, labels, **kwargs)
        elif self.loss_type == 'kl':
            return self.kl_divergence_loss(logits, labels, **kwargs)
        elif self.loss_type == 'ce':
            return self.cross_entropy_loss(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
        elif self.loss_type == 'ce_temperature':
            return self.cross_entropy_loss_with_temperature(logits, labels,T=similarity ,vocab_size=self.config.vocab_size, **kwargs)
        elif self.loss_type == 'ce_fl':  # focal loss and ce
            return self.focal_loss(logits, labels, **kwargs)
        elif self.loss_type == 'ls_ranking':
            ls_loss = self.soft_ce_loss(logits, labels, **kwargs)
            ranking_loss = self.margin_ranking_loss(logits, labels, **kwargs)
            return ls_loss + ranking_loss
        elif self.loss_type == 'kl_ranking':
            kl_loss = self.kl_divergence_loss(logits, labels, **kwargs)
            ranking_loss = self.margin_ranking_loss(logits, labels, **kwargs)
            return kl_loss + ranking_loss
        elif self.loss_type == 'ce_ranking':
            ce_loss = self.cross_entropy_loss(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
            ranking_loss = self.margin_ranking_loss(logits, labels, **kwargs)
            return ce_loss + ranking_loss
        elif self.loss_type == 'ranking':
            return self.margin_ranking_loss(logits, labels, **kwargs)
        elif self.loss_type == 'psl':
            return self.psl_loss(logits, labels , **kwargs)
        elif self.loss_type == 'psl_v2':
            return self.psl_loss_v2(logits, labels , **kwargs)
        elif self.loss_type == 'binary_ce':
            return self.binary_cross_entropy_loss(logits, labels, **kwargs)
        else:
            return self.cross_entropy_loss(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.51", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        # print(input_ids.shape)
        metadata = kwargs.get('metadata',[])
        similarity = metadata[0][0].get('author_sim',0)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        loss = None
        score = None
        lm_logits = self.lm_head(hidden_states)
        if self.use_hybrid_head:
            lm_logits = self.hybrid_lm_head(lm_logits)
        if labels is not None:
            if hasattr(self, 'is_binary_head') and self.is_binary_head:
                masked_labels = torch.full_like(input_ids, -100).squeeze().to(labels.dtype)
                indices = (input_ids.squeeze(0) == self.LABEL_TOKEN_IDS).nonzero().squeeze(-1)
                masked_labels[indices] = torch.tensor(labels, device=labels.device, dtype=labels.dtype)
                if masked_labels.dim() == 1:
                    masked_labels = masked_labels.unsqueeze(0)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = masked_labels[:,1:].contiguous() 
                loss = self.compute_loss(logits=shift_logits, labels=shift_labels, similarity=similarity, **kwargs)
            else:
                lm_logits = lm_logits.float()
                indices = (input_ids.squeeze(0) == self.LABEL_TOKEN_IDS).nonzero().squeeze(-1)
                masked_labels = torch.ones_like(input_ids, device=self.device, dtype=torch.long) * -100
                masked_labels[:, indices] = torch.where(labels==1, self.YES_TOKEN_IDS, self.NO_TOKEN_IDS)
                lm_logits = lm_logits.to(torch.float32)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = masked_labels[:, 1:].contiguous()
                loss = self.compute_loss(logits=shift_logits, labels=shift_labels, similarity=similarity, **kwargs)
        
        if hasattr(self, 'is_binary_head') and self.is_binary_head:
            logits = lm_logits[:, indices-1, :].detach()
            probs = F.softmax(logits, dim=-1)
            score = probs[:, :, 1].squeeze(0)
        else:
            logits = lm_logits[:, indices-1, :].detach()
            yes_logit, no_logit = logits[:, :, self.YES_TOKEN_IDS], logits[:, :, self.NO_TOKEN_IDS]
            score = F.softmax(torch.concat([yes_logit, no_logit], dim=0), dim=0)[0]
        
        return INDModelWithPast(
            loss=loss,
            logits=score.unsqueeze(0),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def add_special_tokens(self, tokenizer):
        self.tokenizer = tokenizer
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))

        YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)

        # self.trainable_token_ids = tokenizer.convert_tokens_to_ids(TRAINABLE_SPECIAL_TOKENS)
        # self.sorted_new_vocab_ids, self.indices = torch.sort(self.trainable_token_ids)
        # self.old_to_new_indices = torch.searchsorted(self.sorted_new_vocab_ids, old_vocab_ids)


    def freeze_lora(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
                
    def unfreeze_lora(self): # unfreeze llm lora parameters
        for name, param in self.model.named_parameters(): #匹配并unfreeze所有'lora'参数
            if 'lora' in name:
                param.requires_grad = True
    
    def monkey_patch_cls_head(self):
        """保持分类头的monkey patch不变"""
        # if use_hybrid_head:
        #     self.hybrid_lm_head = nn.Linear(self.config.vocab_size, 1, bias=False)
        # else:
        original_weights = self.lm_head.weight.data
        binary_head = nn.Linear(self.config.hidden_size, 2, bias=False)
        binary_head.weight.data[0] = original_weights[self.NO_TOKEN_IDS]  # 0对应No
        binary_head.weight.data[1] = original_weights[self.YES_TOKEN_IDS]  # 1对应Yes
        self.lm_head = binary_head
        self.is_binary_head = True

    def add_hybrid_head(self):
        """添加混合分类头"""
        self.hybrid_lm_head = nn.Linear(self.config.vocab_size, 1, bias=False)
        self.use_hybrid_head = True

    def set_token(self,tokenizer):
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)
        
class LlamaForCrossND(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        
        loss = None
        score = None
        
        lm_logits = self.lm_head(hidden_states)        

        if labels is not None:
            if hasattr(self, 'is_binary_head') and self.is_binary_head:

                masked_labels = torch.full_like(input_ids, -100).squeeze()  # -100是PyTorch中忽略的标签值
                indices = (input_ids.squeeze(0) == self.LABEL_TOKEN_IDS).nonzero().squeeze(-1)  # 得到 tensor([1, 3, 5])
                # 将 labels 对应位置的值替换为 label_list
                masked_labels[indices] = torch.tensor(labels, device=labels.device, dtype=torch.long)
                loss = self.loss_function(logits=lm_logits, labels=masked_labels, vocab_size=self.config.vocab_size, **kwargs)
            else:
                masked_labels = torch.full_like(input_ids, -100).squeeze()  
                indices = (input_ids.squeeze(0) == self.LABEL_TOKEN_IDS).nonzero().squeeze(-1)
                masked_labels[:,indices] = torch.tensor([self.YES_TOKEN_IDS if l == 1 else self.NO_TOKEN_IDS for l in labels],device = self.device).unsqueeze(0)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = masked_labels[:,1:].contiguous() 
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))
        
        # 处理二分类头和原始词表头的得分计算
        if hasattr(self, 'is_binary_head') and self.is_binary_head:
            # 如果是二分类头，直接取相应位置的logits
            logits = lm_logits[:, indices, :].detach()
            # 应用softmax得到概率
            probs = F.softmax(logits, dim=-1)
            # 直接取第1列（索引为1）作为Yes的概率
            score = probs[:, :, 1].squeeze(0)
        else:
            # 原始处理方式
            logits = lm_logits[:, indices-1, :].detach()
            yes_logit, no_logit = logits[:, :, self.YES_TOKEN_IDS], logits[:, :, self.NO_TOKEN_IDS]
            score = F.softmax(torch.concat([yes_logit, no_logit], dim=0), dim=0)[0]
        
        return INDModelWithPast(
            loss=loss,
            logits=score.unsqueeze(0),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def add_special_tokens(self, tokenizer):
        self.tokenizer = tokenizer
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))

        YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)

    def freeze_lora(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
                
    def unfreeze_lora(self): # unfreeze llm lora parameters
        for name, param in self.model.named_parameters(): #匹配并unfreeze所有'lora'参数
            if 'lora' in name:
                param.requires_grad = True
    
    def monkey_patch_cls_head(self, loss_type='ls'):
        original_weights = self.lm_head.weight.data
        
        binary_head = nn.Linear(self.config.hidden_size, 2, bias=False)
        
        binary_head.weight.data[0] = original_weights[self.NO_TOKEN_IDS]  # 0对应No
        binary_head.weight.data[1] = original_weights[self.YES_TOKEN_IDS]  # 1对应Yes
        
        self.lm_head = binary_head
        self.is_binary_head = True  # 标记已经使用二分类头
        
        # 同时修改损失函数
        self.monkey_patch_loss_function(type=loss_type)

    def monkey_patch_loss_function(self, type = 'ls'):            
        def label_smoothing_loss(logits, labels, vocab_size, epsilon=0.6, **kwargs):
            # 展平 logits 和 labels
            logits_flat = logits.view(-1, logits.size(-1))  # [batch * seq_len, 2]
            labels_flat = labels.view(-1)                  # [batch * seq_len]
            
            # 获取有效位置 (忽略 -100)
            indices = (labels != -100).nonzero().squeeze(-1)
            prev_indices = indices - 1
            
            # 确保prev_indices中的所有索引都是有效的
            valid_shift_mask = prev_indices >= 0
            
            # 获取有效的logits和标签
            valid_logits = logits_flat[prev_indices[valid_shift_mask]]
            valid_labels = labels_flat[indices[valid_shift_mask]]

            
            # 创建平滑目标 (形状: [num_valid, 2])
            num_classes = valid_logits.size(-1)
            smoothed_targets = torch.full_like(valid_logits, epsilon / (num_classes - 1))
            
            # 将正确类别的值设为 1-epsilon
            smoothed_targets.scatter_(
                dim=1,
                index=valid_labels.long().unsqueeze(1),  # [num_valid, 1]
                value=1 - epsilon
            )
            
            # 计算交叉熵
            log_probs = F.log_softmax(valid_logits, dim=-1)
            loss_per_token = - (smoothed_targets * log_probs).sum(dim=-1)  # [num_valid]
            
            return loss_per_token.mean()  # 仅对有效token平均
        
        def cross_entropy_loss(logits, labels, vocab_size, **kwargs):
            # 处理输入，确保labels是一维的
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            # 找出标签位置
            indices = (labels != -100).nonzero().squeeze(-1)
            # 自回归模型需要将logits向左移动一位
            prev_indices = indices - 1
        
            valid_labels = labels[indices]
            valid_logits = logits[:, prev_indices, :]
            
            # 直接使用标签，因为标签已经是0和1了
            binary_labels = valid_labels.long()
            
            # 使用交叉熵损失
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(valid_logits.view(-1, valid_logits.size(-1)), binary_labels.view(-1))
            
            return ce_loss
        
        # 根据类型选择损失函数
        if type == 'ls':
            self.loss_function = label_smoothing_loss
        else:
            self.loss_function = cross_entropy_loss

