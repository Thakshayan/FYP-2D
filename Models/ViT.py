import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class MultiHead3DAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Conv3d(in_channels=embedding_dim, out_channels=embedding_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_attention = nn.Conv3d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h w) -> k b h w t d', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h w t d -> b t (h w d)")
        x = self.out_attention(x)

        return x

                      

class TransformerEncoderBlock3D(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.multi_head_attention = MultiHead3DAttention(embedding_dim, head_num)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout3d(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm(x)
        
        x = self.layer_norm(x)
        return x


class TransformerEncoder3D(nn.Module):
    def __init__(self, embedding_dim, head_num, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock3D(embedding_dim, head_num) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT3D(nn.Module):
    def __init__(self, embedding_dim, head_num, block_num):
        super().__init__()
        
        self.transformer = TransformerEncoder3D(embedding_dim, head_num, block_num)

    def forward(self, x):
        x = self.transformer(x)
        return x

