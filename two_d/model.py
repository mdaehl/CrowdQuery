# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from torch import nn, Tensor
from typing import Tuple, Dict, Union, List
import torch
from torch.nn import functional as F
from torchvision.transforms import Resize
from mmdet.registry import MODELS
from mmdet.models import DeformableDETR
from mmdet.models import (
    DeformableDetrTransformerEncoder,
    SinePositionalEncoding,
)
from mmdet.utils import ConfigType
from mmdet.structures import SampleList, OptSampleList
from .wrapper_decoder import DensityDeformableDetrTransformerDecoder
from .wrapper_predictor import (
    RegressionDensityPredictor,
)
from .misc.utils import mask_to_shape


@MODELS.register_module()
class CQ2D(DeformableDETR):
    def __init__(self, density_predictor_cfg: ConfigType, **kwargs):
        self.density_predictor_cfg = density_predictor_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DensityDeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

        density_predictor_type = self.density_predictor_cfg["type"]
        del self.density_predictor_cfg["type"]
        if density_predictor_type == "regression":
            self.density_predictor = RegressionDensityPredictor(
                **self.density_predictor_cfg
            )
        else:
            raise ValueError(
                f"The type '{self.density_predictor_cfg['type']}' is no valid density predictor."
            )

    def pre_transformer(
        self, mlvl_feats: Tuple[Tensor], batch_data_samples: OptSampleList = None
    ) -> Tuple[Dict, Dict]:
        # mostly taken from DeformableDETR
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all(
            [s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list]
        )
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:])
                    .to(torch.bool)
                    .squeeze(0)
                )
                mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)
        ):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # (num_level)
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1
            )
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats), 2)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            mlvl_pos_embeds=mlvl_pos_embeds,
            mlvl_masks=mlvl_masks,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            mlvl_masks=mlvl_masks,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return encoder_inputs_dict, decoder_inputs_dict

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        img_feats = self.extract_feat(batch_inputs)
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples
        )

        density_loss, density_embed = self.density_predictor.loss_and_forward(
            img_feats, encoder_inputs_dict, batch_data_samples
        )
        # adapt dict for custom usage
        decoder_inputs_dict["density_embed"] = density_embed
        del encoder_inputs_dict["mlvl_pos_embeds"]
        del encoder_inputs_dict["mlvl_masks"]

        head_inputs_dict = self.forward_core_transformer(
            encoder_inputs_dict, decoder_inputs_dict
        )

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )

        losses.update(density_loss)

        return losses

    def forward_decoder(
        self,
        query: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        mlvl_masks: Tensor,
        density_embed: Tensor = None,
    ) -> Dict:
        if mlvl_masks[1] is not None:
            density_mask = mlvl_masks[1].flatten(start_dim=1)
        else:
            density_mask = None
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            density_mask=density_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            density_embed=density_embed,
            reg_branches=self.bbox_head.reg_branches if self.with_box_refine else None,
        )

        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(hidden_states=inter_states, references=references)
        return decoder_outputs_dict

    def forward_core_transformer(
        self, encoder_inputs_dict: dict, decoder_inputs_dict: dict
    ) -> Dict:
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        img_feats = self.extract_feat(batch_inputs)
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples
        )

        pred_density_maps, density_embed = self.density_predictor.forward(
            img_feats, encoder_inputs_dict
        )
        # adapt dict for custom usage
        decoder_inputs_dict["density_embed"] = density_embed

        density_mask = encoder_inputs_dict["mlvl_masks"][1]
        if density_mask is not None:
            unpadded_mask = ~density_mask
        else:
            unpadded_mask = None

        del encoder_inputs_dict["mlvl_pos_embeds"]
        del encoder_inputs_dict["mlvl_masks"]

        head_inputs_dict = self.forward_core_transformer(
            encoder_inputs_dict, decoder_inputs_dict
        )

        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )

        # use first item in batch to calculate original density map shape
        out_pred_density_map_shape = tuple(pred_density_maps[0].shape[:2])

        # mask out padded area in prediction
        if density_mask is not None:
            pred_density_maps[density_mask] = 0

        # get gt density maps
        if unpadded_mask is not None:
            density_map_shapes = mask_to_shape(unpadded_mask)
        else:
            density_map_shapes = [out_pred_density_map_shape] * len(pred_density_maps)
        scaled_bbox_list = self.density_predictor.get_scaled_bbox_list(
            batch_data_samples, density_map_shapes, use_orig_shape=True
        )

        # use regression maps for visualization independent of the model being regression
        gt_density_maps = self.density_predictor.get_gt_regression_density_maps(
            scaled_bbox_list, out_pred_density_map_shape
        )
        scales = torch.tensor(
            [sample.scale_factor[::-1] for sample in batch_data_samples]
        )  # flip to have h, w order

        # process predicted and ground truth density maps
        processed_pred_density_maps = []
        processed_gt_density_maps = []

        # convert to list as the shapes might differ
        pred_density_maps = list(pred_density_maps)
        gt_density_maps = list(gt_density_maps)

        # crop if required
        if unpadded_mask is not None:
            for idx, (pred_density_map, gt_density_map, (h_max, w_max)) in enumerate(
                zip(pred_density_maps, gt_density_maps, density_map_shapes)
            ):
                pred_density_maps[idx] = pred_density_map[:h_max, :w_max]
                gt_density_maps[idx] = gt_density_map[:h_max, :w_max]

        # rescale density map w.r.t. the original input size
        for pred_density_map, gt_density_map, scale in zip(
            pred_density_maps, gt_density_maps, scales
        ):
            density_map_shape = tuple(
                map(int, torch.tensor(out_pred_density_map_shape) / scale)
            )
            resize_transform = Resize(density_map_shape)
            processed_gt_density_maps.append(
                resize_transform(gt_density_map[None, :, :])[0]
            )
            processed_pred_density_maps.append(
                resize_transform(pred_density_map[None, :, :])[0]
            )

        # add pred and gt density maps to data samples
        for data_sample, pred_density_map, gt_density_map in zip(
            batch_data_samples, processed_pred_density_maps, processed_gt_density_maps
        ):
            data_sample.pred_density_map = pred_density_map
            data_sample.gt_density_map = gt_density_map

        return batch_data_samples

    def _forward(
        self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None
    ) -> Tuple[List[Tensor]]:
        img_feats = self.extract_feat(batch_inputs)

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples
        )

        density_embed = self.density_predictor.forward(img_feats, encoder_inputs_dict)
        # adapt dict for custom usage
        decoder_inputs_dict["density_embed"] = density_embed
        del encoder_inputs_dict["mlvl_pos_embeds"]
        del encoder_inputs_dict["mlvl_masks"]

        head_inputs_dict = self.forward_core_transformer(
            encoder_inputs_dict, decoder_inputs_dict
        )

        results = self.bbox_head.forward(**head_inputs_dict)
        return results
