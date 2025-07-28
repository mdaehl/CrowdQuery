from two_d.dataset.stcrowd import STCrowdDataset2D
from mmdet.evaluation import CrowdHumanMetric
from mmdet.datasets.transforms import (
    PackDetInputs,
    RandomFlip,
    LoadAnnotations,
    Resize,
)


dataset_type = STCrowdDataset2D

data_root = "data/st_crowd/"

batch_size = 8  # 8 as default

backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=Resize, scale=(1280, 720), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=PackDetInputs,
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "flip",
            "flip_direction",
        ),
    ),
]
val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type=Resize, scale=(1280, 720), keep_ratio=True),
    # avoid bboxes being resized
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,  # The 'batch_sampler' may decrease the precision
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotation_train.odgt",
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotation_val.odgt",
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type=CrowdHumanMetric,
    ann_file=data_root + "annotation_val.odgt",
    metric=["AP", "MR", "JI"],
    backend_args=backend_args,
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator