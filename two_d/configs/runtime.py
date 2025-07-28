from mmengine.hooks import (
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.runner import LogProcessor

from two_d.misc.visualization import DensityDetLocalVisualizer
from mmdet.engine.hooks import DetVisualizationHook
from mmengine.hooks import CheckpointHook


default_scope = None

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=1),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=DetVisualizationHook, draw=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)


log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

log_level = "INFO"
load_from = None
resume = False

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type=DensityDetLocalVisualizer, vis_backends=vis_backends, name="visualizer"
)