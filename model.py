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
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from utils import LABEL_TOKEN,EMBED_TOKEN,GRAPH_TOKEN,TRAINABLE_SPECIAL_TOKENS
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import GenerationMixin, Qwen3PreTrainedModel, Qwen3Model
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


class Qwen3ForCrossND(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
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
        # labels_pos = torch.masked_select(torch.arange(input_ids.shape[-1], device = self.device), input_ids == self.LABEL_TOKEN_IDS)
        
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        lm_logits = self.lm_head(hidden_states)        
        
        # logits = lm_logits[:,labels_pos-1].detach()
        # yes_logit,no_logit= logits[:,:,self.YES_TOKEN_IDS],logits[:,:,self.NO_TOKEN_IDS]
        # score = F.softmax(torch.concat([yes_logit,no_logit],dim=0),dim=0)[0]      

        if labels is not None:
            masked_labels = torch.full_like(input_ids, -100).squeeze()  # -100是PyTorch中忽略的标签值
            indices = (input_ids.squeeze(0) == self.LABEL_TOKEN_IDS).nonzero().squeeze(-1)  # 得到 tensor([1, 3, 5])
            # 将 labels 对应位置的值替换为 label_list
            masked_labels[indices] = torch.tensor(labels, device=labels.device, dtype=torch.long)


            # clabels = labels[labels != -100]
            # masked_labels = torch.ones_like(input_ids,device= self.device,dtype = torch.long)*-100
            # masked_labels[:,labels_pos] = torch.tensor([self.YES_TOKEN_IDS if l == 1 else self.NO_TOKEN_IDS for l in clabels],device = self.device).unsqueeze(0)
            # shift_logits = lm_logits[:, :-1, :].contiguous()
            # shift_labels = masked_labels[1:].unsqueeze(0).contiguous() 
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))


            loss = self.loss_function(logits=lm_logits, labels=masked_labels, vocab_size=self.config.vocab_size, **kwargs)
            
            # shift_logits = lm_logits[:, :-1, :].contiguous()
            # shift_labels = masked_labels[:,1:].contiguous()
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))
            # lm_logits = lm_logits.to(hidden_states.dtype)
            # loss = loss.to(hidden_states.dtype)

            # labels = torch.where(labels == 1, self.YES_TOKEN_IDS, labels)
            # labels = torch.where(labels == 0, self.NO_TOKEN_IDS, labels)
            # lm_logits = lm_logits.to(torch.float32)
            # shift_logits = lm_logits[:, :-1, :].contiguous()
            # shift_labels = labels[:,1:].contiguous()
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # # shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size * seq_len, vocab_size)
            # # shift_labels = shift_labels.view(-1)  # (batch_size * seq_len)       
            # loss = loss_fct(shift_logits.squeeze(0),shift_labels.to(self.device).squeeze(0))
            # lm_logits = lm_logits.to(hidden_states.dtype)
            # loss = loss.to(hidden_states.dtype)
        
        logits = lm_logits[:,indices-1].detach()
        yes_logit,no_logit= logits[:,:,self.YES_TOKEN_IDS],logits[:,:,self.NO_TOKEN_IDS]
        score = F.softmax(torch.concat([yes_logit,no_logit],dim=0),dim=0)[0]
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
    
    def modify_cls_head(self):
        original_weights = self.lm_head.weight.data
        
        binary_head = nn.Linear(self.config.hidden_size, 2, bias=False)
        
        binary_head.weight.data[0] = original_weights[self.YES_TOKEN_IDS]
        binary_head.weight.data[1] = original_weights[self.NO_TOKEN_IDS]
        
        self.lm_head = binary_head
