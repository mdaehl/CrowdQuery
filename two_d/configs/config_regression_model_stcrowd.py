from mmdet.models.necks import ChannelMapper
from mmdet.models.backbones import ResNet
from two_d.model import CQ2D
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.losses import FocalLoss, L1Loss, GIoULoss
from mmdet.models.task_modules.assigners import (
    HungarianAssigner,
    FocalLossCost,
    BBoxL1Cost,
    IoUCost,
)
from mmdet.models.dense_heads import DeformableDETRHead


model = dict(
    type=CQ2D,
    num_queries=1000,
    num_feature_levels=4,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type=ChannelMapper,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256, batch_first=True
            ),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=512, ffn_drop=0.1),  # 512
        ),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True
            ),
            vis_cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256, batch_first=True
            ),
            density_attn_cfg=dict(
                d_model=256, nhead=8, dropout=0.1, n_groups=1, extra_proj=True
            ),
            use_densities=[True, True, True, True, True, True],
            ffn_cfg=dict(embed_dims=256, feedforward_channels=512, ffn_drop=0.1),  # 512
        ),
        post_norm_cfg=None,
    ),
    density_predictor_cfg=dict(
        type="regression",
        density_predictor_module_cfg=dict(
            min_val=0,
            max_val=3,
            num_bins=121,
            embed_mode="uniform",
        ),
        density_encoding_cfg=dict(
            rel_std_width=3,
            rel_std_height=3,
            min_val=0,
            max_val=3,
        ),
        density_loss_cfg=dict(use_count_loss=False, loss_weight=1),
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type=DeformableDETRHead,
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=FocalLoss, use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=GIoULoss, loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format="xywh"),
                dict(type=IoUCost, iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(),
)
