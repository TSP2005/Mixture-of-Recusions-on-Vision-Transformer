import torch
import torch.nn as nn
from transformers import ViTConfig
from .util import MoRLayerOutputWithPast
from .transformer_block import MoRVitBlock
from .expert_choice_router import MoRVitExpertChoiceRouter

class MoRVitModel(nn.Module):
    def __init__(self, config, cfg):
        super().__init__()
        self.config = config
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, config.hidden_size, kernel_size=14, stride=14)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, config.hidden_size))
        self.blocks = nn.ModuleList([MoRVitBlock(config) for _ in range(cfg.recursive.num_recursion)])
        self.router = MoRVitExpertChoiceRouter(config, self.blocks, cfg)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.head = nn.Linear(config.hidden_size, 1000)  # ImageNet-1K classes

    def forward(self, pixel_values, attention_mask=None):
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.router(x, attention_mask)
        x = self.norm(x.hidden_state)
        logits = self.head(x)
        return {"logits": logits, "router_z_loss": x.router_z_loss}

    def vit_forward(self, pixel_values, attention_mask=None):
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, attention_mask)
        x = self.norm(x)
        logits = self.head(x)
        return {"logits": logits}

def mor_vit_from_pretrained(pretrained_model_name_or_path, cfg):
    config = ViTConfig.from_pretrained(pretrained_model_name_or_path)
    model = MoRVitModel(config, cfg)
    state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    return model

def vit_from_pretrained(pretrained_model_name_or_path):
    config = ViTConfig.from_pretrained(pretrained_model_name_or_path)
    model = MoRVitModel(config, None)  # Disable MoR for baseline ViT
    state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    return model