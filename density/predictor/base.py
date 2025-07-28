# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from torch import Tensor
from torch import nn

from .. import embedding


class DensityPredictor(nn.Module):
    def __init__(
        self,
        min_val: float,
        max_val: float,
        d_model: int,
        num_bins: int,
        embed_mode: str,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins

        density_embed = nn.Embedding(num_bins, d_model)
        self.embed_mapper = embedding.EmbeddingMapper(
            density_embed, min_val=self.min_val, max_val=self.max_val, mode=embed_mode
        )

    def build_density_embed(
        self,
        density_maps: Tensor,
        enc_embed: Tensor,
    ) -> Tensor:
        # get embedding from density map
        density_map_embed = self.embed_mapper(density_maps)

        # get final density embedding by adding embeddings
        density_embed = enc_embed + density_map_embed

        return density_embed
