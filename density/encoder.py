# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from __future__ import annotations
import copy
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange


def _get_clones(module: nn.Module, n: int):
    """Copied from torch's transformer.py"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class DensityEncoder(nn.Module):
    def __init__(self, encoder_layer: DensityEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self, src: Tensor, pos_embed: Tensor = None, key_padding_mask: Tensor = None
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos_embed, key_padding_mask)

        return output


class DensityEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, src: Tensor, pos_embed: Tensor = None, key_padding_mask: Tensor = None
    ) -> Tensor:
        if pos_embed is not None:
            query = src + pos_embed
        else:
            query = src

        # reshape src from batch, feature, height, width -> batch, seq (height*width), feature
        query = rearrange(query, "b c h w -> b (h w) c")
        key = query
        x = rearrange(src, "b c h w -> b (h w) c")

        x = self.norm1(
            x
            + self._sa_block(
                query=query, key=key, value=x, key_padding_mask=key_padding_mask
            )
        )
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor = None
    ) -> Tensor:
        """self-attention block
        Copied from torch's transformer.py"""
        x = self.self_attn(
            query, key, value, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """feed forward block
        Copied from torch's transformer.py"""
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)
