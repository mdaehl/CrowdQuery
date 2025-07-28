from typing import Optional

import numpy as np
from mmdet.visualization.local_visualizer import DetLocalVisualizer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from misc.visualization import get_density_img
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample


@VISUALIZERS.register_module()
class DensityDetLocalVisualizer(DetLocalVisualizer):
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Optional[DetDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        pred_score_thr: float = 0.3,
        step: int = 0,
    ) -> None:
        super().add_datasample(
            name=name,
            image=image,
            data_sample=data_sample,
            draw_gt=draw_gt,
            draw_pred=draw_pred,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            pred_score_thr=pred_score_thr,
            step=step,
        )

        # pred density
        pred_density_map = data_sample.pred_density_map

        # gt density
        gt_density_map = data_sample.gt_density_map

        # density figure
        density_fig = get_density_img(gt_density_map, pred_density_map)

        # convert matplotlib plot to rgb array
        canvas = FigureCanvas(density_fig)
        canvas.draw()
        buffer = np.asarray(canvas.buffer_rgba())
        density_img = buffer[:, :, :3]  # format of the img (h, w, c)

        self.add_image("val_density_img", density_img, step)
