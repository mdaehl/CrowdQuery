# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from typing import List, Tuple, Optional

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch import nn

from .. import encoder
from .base import DensityPredictor


class ConvDensityPredictor(DensityPredictor):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(d_model=d_model, **kwargs)

        self.down_sampler = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model),
        )
        self.projector = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)), nn.GroupNorm(32, d_model)
        )
        self.up_sampler = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)), nn.GroupNorm(32, d_model)
        )

        self.density_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
        )

        encoder_layer = encoder.DensityEncoderLayer(
            d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.density_encoder = encoder.DensityEncoder(encoder_layer, 1)

    def forward(
        self,
        feature_list: List[Tensor],
        pos_embed: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        feat_map_size = feature_list[1].shape[-2:]
        feat_8 = self.down_sampler(feature_list[0])
        feat_16 = self.projector(feature_list[1])
        feat_32 = self.up_sampler(
            F.interpolate(feature_list[2], size=feat_map_size, mode="bilinear")
        )
        feat_combined = (feat_8 + feat_16 + feat_32) / 3

        density_feats = self.density_head(feat_combined)

        enc_embed = self.density_encoder(density_feats, pos_embed, key_padding_mask)
        _, _, h, w = density_feats.shape
        enc_embed = rearrange(enc_embed, "b (h w) c -> b c h w", h=h, w=w)

        return density_feats, enc_embed


class RegressionConvDensityPredictor(ConvDensityPredictor):
    def __init__(
        self,
        d_model: int = 256,
        **kwargs,
    ):
        super().__init__(d_model=d_model, **kwargs)

        self.density_predictor = nn.Conv2d(d_model, 1, kernel_size=(1, 1))

    def forward(
        self,
        feature_list: List[Tensor],
        pos_embed: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        density_feats, enc_embed = super().forward(
            feature_list, pos_embed, key_padding_mask
        )

        density_maps = self.density_predictor(density_feats)

        density_maps = density_maps.squeeze(dim=1)

        density_embed = self.build_density_embed(density_maps, enc_embed)

        return None, density_maps, density_embed
