from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

from .util import ROUTER_TYPES, MoRLayerOutputWithPast
from .transformer_block import MoRVitBlock

logger = logging.get_logger(__name__)

class MoRVitExpertChoiceRouter(nn.Module):
    def __init__(self, config, block_list, cfg):
        super().__init__()
        self.config = config
        self.block_list = block_list
        self.cfg = cfg
        self.num_recursion = cfg.recursive.num_recursion
        assert len(block_list) == self.num_recursion, "Number of recursion should match blocks"
        self.router = ROUTER_TYPES[cfg.mor.router_type](config, out_dim=self.num_recursion)
        self.gating = cfg.mor.expert.gating

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bs, seq_len, hidden_dim = hidden_states.shape
        router_weights = self.router(hidden_states)
        router_probs = torch.sigmoid(router_weights) * self.cfg.mor.expert.alpha
        top_k_values, top_k_indices = router_probs.topk(k=1, dim=-1)
        selected_tokens = top_k_indices.squeeze(-1)

        total_x = torch.zeros_like(hidden_states)
        for index, blk in enumerate(self.block_list):
            mask = (selected_tokens == index).unsqueeze(-1).expand(-1, -1, hidden_dim)
            top_k_tokens = hidden_states * mask
            outputs = blk(top_k_tokens, attention_mask)
            if self.gating == "weighted":
                total_x = total_x + (outputs * top_k_values.unsqueeze(-1).expand(-1, -1, hidden_dim))
            else:
                total_x = total_x + outputs

        router_z_loss = None
        if self.training and self.cfg.mor.z_loss:
            router_z_loss = torch.logsumexp(router_weights, dim=-1).square().mean() * self.cfg.mor.z_coeff

        return MoRLayerOutputWithPast(
            hidden_state=total_x,
            router_z_loss=router_z_loss if self.training else None
        )