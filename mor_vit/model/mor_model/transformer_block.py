import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Minimal transformer block with multi-head self-attention and MLP."""

    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out + residual
        x = x + self.mlp(self.ln2(x))
        return x
