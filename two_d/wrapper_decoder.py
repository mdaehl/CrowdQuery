# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from torch import Tensor
from mmengine.model import ModuleList
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import build_norm_layer
from mmdet.models.layers import DetrTransformerDecoder
from mmcv.ops import MultiScaleDeformableAttention
from density.decoder import DensityDecoderLayerBlock
import torch
from torch import nn
from typing import Optional, Tuple
from mmdet.models.layers import inverse_sigmoid


class DensityDeformableDetrTransformerDecoder(DetrTransformerDecoder):
    def _init_layers(self) -> None:
        use_densities = self.layer_cfg["use_densities"]
        del self.layer_cfg["use_densities"]
        assert len(use_densities) == self.num_layers

        self.layers = ModuleList(
            [
                DensityDeformableDETRTransformerDecoderLayer(
                    **self.layer_cfg, use_density=use_density
                )
                for use_density in use_densities
            ]
        )
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError(f"There is not post_norm in {self._get_name()}")

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        density_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        density_embed: Tensor,
        reg_branches: Optional[nn.Module] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                density_mask=density_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                density_embed=density_embed,
                **kwargs,
            )

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2
                    ] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DensityDeformableDETRTransformerDecoderLayer(BaseModule):
    def __init__(
        self,
        density_attn_cfg: ConfigType,
        self_attn_cfg: ConfigType,
        vis_cross_attn_cfg: ConfigType,
        ffn_cfg: ConfigType,
        use_density: bool,
        norm_cfg: OptConfigType = dict(type="LN"),
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.density_attn_cfg = density_attn_cfg
        self.self_attn_cfg = self_attn_cfg
        self.vis_cross_attn_cfg = vis_cross_attn_cfg
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.use_density = use_density
        self._init_layers()

    def _init_layers(self) -> None:
        if self.use_density:
            self.density_decoder_block = DensityDecoderLayerBlock(
                **self.density_attn_cfg
            )
        else:
            self.density_decoder_block = None

        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.vis_cross_attn = MultiScaleDeformableAttention(**self.vis_cross_attn_cfg)
        self.embed_dims = self.vis_cross_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        query_pos: Tensor,
        density_embed: Tensor,
        key_padding_mask: Tensor,
        density_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        # already contains norm
        if self.density_decoder_block is not None:
            query = self.density_decoder_block(
                query,
                pos_embed=query_pos,
                density_embed=density_embed,
                density_mask=density_mask,
                # **kwargs
            )
        else:
            query = self.self_attn(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=None,
                **kwargs,
            )
            query = self.norms[0](query)

        query = self.vis_cross_attn(
            query=query,
            value=value,
            query_pos=query_pos,
            key_pos=None,  # uses key_pos
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[-2](query)
        query = self.ffn(query)
        query = self.norms[-1](query)
        return query
