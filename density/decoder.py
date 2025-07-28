# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from torch import nn
from torch import Tensor
from einops import rearrange


class DensityDecoderLayerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_groups: int,
        dropout: float,
        extra_proj: bool = True,
    ):
        super().__init__()
        self.inter_query_attn = InterQuerySelfAttention(
            d_model=d_model,
            nhead=nhead,
            n_groups=n_groups,
            dropout=dropout,
            extra_proj=extra_proj,
        )
        self.density_cross_attn = DensityCrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model), nn.LayerNorm(d_model)])

    def forward(
        self,
        src: Tensor,
        pos_embed: Tensor,
        density_embed: Tensor,
        density_mask: Tensor,
        self_attn_mask: Tensor = None,
    ):
        output = self.density_cross_attn(src, density_embed, density_mask)
        output = self.norms[0](output)
        output = self.inter_query_attn(output, pos_embed, self_attn_mask=self_attn_mask)
        output = self.norms[1](output)
        return output


class InterQuerySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_groups: int,
        dropout: float,
        extra_proj: bool = True,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.extra_proj = extra_proj
        if self.extra_proj:
            self.sa_q_content_proj = nn.Linear(d_model, d_model)
            self.sa_q_pos_proj = nn.Linear(d_model, d_model)
            self.sa_k_content_proj = nn.Linear(d_model, d_model)
            self.sa_k_pos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def shape_group_as_batch(feature: Tensor, n_groups: int) -> Tensor:
        feature_groups = rearrange(
            feature,
            "b (n_groups n_query) c -> (n_groups b) n_query c",
            n_groups=n_groups,
        )
        return feature_groups

    def forward(
        self, src: Tensor, pos_embed: Tensor, self_attn_mask: Tensor = None
    ) -> Tensor:
        query = key = src + pos_embed
        value = src

        if self.extra_proj:
            q_content = self.sa_q_content_proj(query)
            q_pos = self.sa_q_pos_proj(query)
            query = q_content + q_pos
            k_content = self.sa_k_content_proj(key)
            k_pos = self.sa_k_pos_proj(key)
            key = k_content + k_pos
            value = self.sa_v_proj(value)

        n_used_groups = self.n_groups if self.training else 1

        # if only 1 group is used, reshaping is irrelevant respectively it would not change the tensor
        if n_used_groups > 1:
            query = self.shape_group_as_batch(query, n_groups=n_used_groups)
            key = self.shape_group_as_batch(key, n_groups=n_used_groups)
            value = self.shape_group_as_batch(value, n_groups=n_used_groups)

        output = self.self_attn(
            query, key, value, attn_mask=self_attn_mask, need_weights=False
        )[0]

        # same here with the reshaping
        if n_used_groups > 1:
            output = rearrange(
                output,
                "(n_groups b) n_query c -> b (n_groups n_query) c",
                n_groups=n_used_groups,
            )

        output = src + self.dropout(output)
        return output


class DensityCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: Tensor, density_embed: Tensor, density_mask: Tensor
    ) -> Tensor:
        density_embed = rearrange(
            density_embed, "b c h w -> b (h w) c"
        )  # reshape for attention calculation
        output = self.cross_attn(
            query=src,
            key=density_embed,
            value=density_embed,
            need_weights=False,
            key_padding_mask=density_mask,
            attn_mask=None,
        )[0]
        output = src + self.dropout(output)
        return output
