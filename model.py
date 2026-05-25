from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import sys
sys.path.append('.')
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.utils import (
    can_return_tuple,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from utils import LABEL_TOKEN
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import GenerationMixin, Qwen3PreTrainedModel, Qwen3Model
import torch.nn.functional as F

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Cache
import logging
logger = logging.getLogger(__name__)
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

class LossKwargs: ...

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

@dataclass
class INDModelWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    score: torch.FloatTensor = None

class Qwen3ForCrossND(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.is_binary_head = False
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
    def cross_entropy_loss(self, logits, labels, **kwargs):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits.squeeze(0), labels.squeeze(0))

    def psl_loss(self, logits, labels, **kwargs):
        celoss = self.cross_entropy_loss(logits, labels, **kwargs)
        metadata = kwargs.get("metadata") or []
        if not metadata or not metadata[0] or not isinstance(metadata[0][0], dict):
            return celoss
        first_item = metadata[0][0]
        if 'p_out_sim' not in first_item or 'author_sim' not in first_item:
            return celoss

        p_out_sim = torch.tensor([i['p_out_sim'] for i in metadata[0]], dtype=logits.dtype, device=logits.device)
        a1_a2_sim = torch.tensor([i['author_sim'] for i in metadata[0]], dtype=logits.dtype, device=logits.device)
        if self.is_binary_head:
            p_yes = torch.softmax(logits, dim=-1)[0, labels[0] != -100, 1]
        else:
            valid_positions = (labels != -100)[0]
            yes_logits = logits[:, valid_positions, self.YES_TOKEN_IDS]
            no_logits = logits[:, valid_positions, self.NO_TOKEN_IDS]

            probs = torch.softmax(torch.stack([no_logits, yes_logits], dim=-1), dim=-1)
            p_yes = probs[:, :, 1]

        PSI = self.model_args.psi
        eta1 = p_out_sim + a1_a2_sim - p_yes - PSI
        eta2 = torch.max(p_out_sim, 1 - p_out_sim) - a1_a2_sim + p_yes - PSI

        eta_cs = torch.minimum(eta1, torch.zeros_like(eta1))
        eta_ics = torch.minimum(eta2, torch.zeros_like(eta2))
        cs_ics_label = (a1_a2_sim > 0.5).long()
        loss = cs_ics_label * eta_cs + (1-cs_ics_label) * eta_ics
        return self.model_args.alpha * loss.mean() + (1-self.model_args.alpha) * celoss

    def compute_loss(self, logits, labels, **kwargs):
        if self.loss_type == 'ce':
            return self.cross_entropy_loss(logits, labels, **kwargs)
        if self.loss_type == 'psl':
            return self.psl_loss(logits, labels , **kwargs)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}. Use 'ce' or 'psl'.")

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
        indices = (input_ids.squeeze(0) == self.LABEL_TOKEN_IDS).nonzero().squeeze(-1)
        if labels is not None:
            if hasattr(self, 'is_binary_head') and self.is_binary_head:
                masked_labels = torch.full_like(input_ids, -100).squeeze().to(labels.dtype)
                masked_labels[indices] = torch.tensor(labels, device=labels.device, dtype=labels.dtype)
                if masked_labels.dim() == 1:
                    masked_labels = masked_labels.unsqueeze(0)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = masked_labels[:,1:].contiguous() 
                loss = self.compute_loss(logits=shift_logits, labels=shift_labels, **kwargs)
            else:
                lm_logits = lm_logits.float()
                masked_labels = torch.ones_like(input_ids, device=self.device, dtype=torch.long) * -100
                masked_labels[:, indices] = torch.where(labels==1, self.YES_TOKEN_IDS, self.NO_TOKEN_IDS)
                lm_logits = lm_logits.to(torch.float32)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = masked_labels[:, 1:].contiguous()
                loss = self.compute_loss(logits=shift_logits, labels=shift_labels, **kwargs)
        
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

    def freeze_lora(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
                
    def unfreeze_lora(self): # unfreeze llm lora parameters
        for name, param in self.model.named_parameters(): #匹配并unfreeze所有'lora'参数
            if 'lora' in name:
                param.requires_grad = True
    
    def monkey_patch_cls_head(self):
        original_weights = self.lm_head.weight.data
        binary_head = nn.Linear(self.config.hidden_size, 2, bias=False)
        binary_head.weight.data[0] = original_weights[self.NO_TOKEN_IDS]
        binary_head.weight.data[1] = original_weights[self.YES_TOKEN_IDS]
        self.lm_head = binary_head
        self.is_binary_head = True

    def set_token(self,tokenizer):
        self.LABEL_TOKEN_IDS = torch.tensor(tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        YES_TOKEN_IDS, NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids(['Yes','No'])
        self.YES_TOKEN_IDS, self.NO_TOKEN_IDS= torch.tensor(YES_TOKEN_IDS), torch.tensor(NO_TOKEN_IDS)
