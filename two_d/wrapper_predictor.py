# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from abc import abstractmethod
from typing import Tuple, Dict, List, Optional

from mmdet.structures import SampleList
from mmdet.utils import ConfigType
from torch import nn, Tensor

from density.density_map import DensityMapEncoder
from density.loss import RegressionDensityLoss
from density.predictor.conv import (
    RegressionConvDensityPredictor,
)
from .misc.utils import mask_to_shape


class DensityPredictor(nn.Module):
    def __init__(
        self,
        density_predictor_module_cfg: ConfigType,
        density_loss_cfg: ConfigType,
        density_encoding_cfg: ConfigType,
    ):
        super().__init__()
        self.count_loss_func = nn.L1Loss()

        self.density_predictor_module_cfg = density_predictor_module_cfg
        self.density_loss_cfg = density_loss_cfg
        self.density_encoding_cfg = density_encoding_cfg

        self.loss_weight = self.density_loss_cfg["loss_weight"]
        self.use_count_loss = self.density_loss_cfg["use_count_loss"]
        del self.density_loss_cfg["loss_weight"]
        del self.density_loss_cfg["use_count_loss"]

        self._init_layers()

    @abstractmethod
    def _init_layers(self):
        raise NotImplementedError

    def forward_pass(
        self, img_feats: Tuple[Tensor], encoder_inputs_dict: dict
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pos_embed = encoder_inputs_dict["mlvl_pos_embeds"][1]
        key_padding_mask = encoder_inputs_dict["mlvl_masks"][1]
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.flatten(1)

        # img feats is the output after the neck
        feature_list = self.get_feature_list(img_feats)

        predictor_out = self.predictor(feature_list, pos_embed, key_padding_mask)
        return predictor_out

    def count_loss(
        self, gt_density_maps_norm: Tensor, pred_instance_counts: Tensor
    ) -> Tensor:
        gt_instance_counts = gt_density_maps_norm.sum(dim=[1, 2])
        count_loss = self.count_loss_func(pred_instance_counts, gt_instance_counts)
        return count_loss

    @staticmethod
    def get_scaled_bbox_list(
        batch_data_samples: SampleList,
        density_map_shapes: List[Tuple],
        use_orig_shape: bool = False,
    ) -> List[Tensor]:
        bbox_list = copy.deepcopy(
            [item.gt_instances.bboxes for item in batch_data_samples]
        )

        if use_orig_shape:
            img_shapes = [data_sample.ori_shape for data_sample in batch_data_samples]
        else:
            img_shapes = [data_sample.img_shape for data_sample in batch_data_samples]

        for bbox_batch, (img_height, img_width), (
            density_map_height,
            density_map_width,
        ) in zip(bbox_list, img_shapes, density_map_shapes):
            bbox_batch[:, [0, 2]] *= density_map_width / img_width
            bbox_batch[:, [1, 3]] *= density_map_height / img_height

        return bbox_list

    def get_gt_density_maps(
        self,
        bbox_list: List[Tensor],
        density_map_shape: Tuple,
    ) -> Tuple[Tensor, Tensor]:
        density_map_height, density_map_width = density_map_shape
        gt_density_maps, gt_density_maps_norm = (
            self.density_map_encoder.transform_to_density_map(
                bbox_list,
                img_height=density_map_height,
                img_width=density_map_width,
            )
        )
        return gt_density_maps, gt_density_maps_norm

    def get_gt_regression_density_maps(
        self,
        bbox_list: List[Tensor],
        density_map_shape: Tuple,
    ) -> Tensor:
        density_map_height, density_map_width = density_map_shape
        gt_regression_density_maps, _ = (
            self.density_map_encoder.transform_to_regression_density_map(
                bbox_list,
                img_height=density_map_height,
                img_width=density_map_width,
            )
        )
        return gt_regression_density_maps

    @staticmethod
    def get_feature_list(feats):
        return list(feats)[:3]


class RegressionDensityPredictor(DensityPredictor):
    def _init_layers(self):
        self.predictor = RegressionConvDensityPredictor(
            **self.density_predictor_module_cfg
        )
        self.loss_func = RegressionDensityLoss(**self.density_loss_cfg)
        self.density_map_encoder = DensityMapEncoder(**self.density_encoding_cfg)

    def forward(self, img_feats: Tuple[Tensor], encoder_inputs_dict: dict):
        _, density_maps, density_embed = self.forward_pass(
            img_feats, encoder_inputs_dict
        )
        return density_maps, density_embed

    def loss(
        self,
        batch_data_samples: SampleList,
        pred_density_maps: Tensor,
        mask: Optional[Tensor],
    ) -> Dict:
        loss = {}

        out_pred_density_map_shape = tuple(pred_density_maps[0].shape[:2])
        if mask is not None:
            density_map_shapes = mask_to_shape(mask)
        else:
            density_map_shapes = [tuple(map(int, out_pred_density_map_shape))] * len(
                pred_density_maps
            )
        scaled_bbox_list = self.get_scaled_bbox_list(
            batch_data_samples, density_map_shapes
        )
        gt_density_maps, gt_density_maps_norm = self.get_gt_density_maps(
            scaled_bbox_list, out_pred_density_map_shape
        )

        density_map_loss = self.loss_func(
            pred_density_maps, gt_density_maps, scaled_bbox_list, mask
        )
        loss["loss_density_map"] = density_map_loss

        for key in loss.keys():
            loss[key] *= self.loss_weight

        return loss

    def loss_and_forward(
        self,
        img_feats: Tuple[Tensor],
        encoder_inputs_dict: dict,
        batch_data_samples: SampleList,
    ) -> Tuple[Dict, Tensor]:
        density_maps, density_embed = self.forward(img_feats, encoder_inputs_dict)

        density_mask = encoder_inputs_dict["mlvl_masks"][1]
        if density_mask is not None:
            unpadded_mask = ~density_mask
        else:
            unpadded_mask = None
        loss = self.loss(batch_data_samples, density_maps, unpadded_mask)
        return loss, density_embed
