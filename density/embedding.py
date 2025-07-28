# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
import warnings

from torch import nn
from torch import Tensor
import torch
from einops import rearrange


class EmbeddingMapper(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        min_val: float,
        max_val: float,
        mode: str = "uniform",
    ):
        super().__init__()
        self.embedding = embedding
        self.num_bins = embedding.num_embeddings
        self.min_val = min_val
        self.max_val = max_val

        if mode == "uniform":
            self.mapping_func = self.uniform_mapping
        elif mode == "linear":
            self.mapping_func = self.progressive_mapping
        elif mode == "regressive":
            self.mapping_func = self.regressive_mapping
            warnings.warn(
                "The regressive mapping is only working correctly for a min_val of 0."
            )
        else:
            raise ValueError(f"The mode {mode} is not supported.")

    def clamp_mapping(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, self.min_val, self.max_val)
        return x

    def uniform_mapping(self, x: Tensor):
        bin_size = (self.max_val - self.min_val) / (self.num_bins - 1)
        indices = (x - self.min_val) / bin_size
        return indices

    def progressive_mapping(self, x: Tensor):
        bin_size = (
            2 * (self.max_val - self.min_val) / (self.num_bins * (self.num_bins - 1))
        )
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (x - self.min_val) / bin_size)
        return indices

    def regressive_mapping(self, x: Tensor):
        indices = -0.5 + 0.5 * (
            1 + ((2 * self.num_bins - 2) / (self.max_val**2)) * (x**2)
        )
        return indices

    def forward(self, x: Tensor) -> Tensor:
        x = self.clamp_mapping(x)
        indices = self.mapping_func(x)
        delta = indices % 1
        delta = delta[..., None]  # add axis for broadcasting

        lower_bin_idx = torch.floor(indices).to(torch.long)
        upper_bin_idx = torch.ceil(indices).to(torch.long)

        # retrieve upper and lower embedding vals
        lower_bin_embed = self.embedding(lower_bin_idx)
        upper_bin_embed = self.embedding(upper_bin_idx)

        # weight embeddings
        weighted_avg_embed = lower_bin_embed * (1 - delta) + upper_bin_embed * delta

        # if density map embedding (not count)
        if len(weighted_avg_embed.shape) == 4:
            weighted_avg_embed = rearrange(weighted_avg_embed, "b h w c -> b c h w")

        return weighted_avg_embed
