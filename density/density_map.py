# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from typing import List, Tuple
from torch import Tensor
import numpy as np
from scipy import signal, stats
import torch


class DensityMapEncoder:
    def __init__(
        self,
        rel_std_width: float,
        rel_std_height: float,
        clip_boxes: bool = True,
        percent_overflow: float = 0.0,
        std_overflow: float = 0.0,
        min_val: float = 0,
        max_val: float = np.inf,
    ):
        # overflow measure overflow per side
        if std_overflow != 0 and percent_overflow != 0:
            raise ValueError(
                f"Only std overflow or percent overflow are allowed not both. Currently set is std_overflow: "
                f"{std_overflow} and percent_overflow: {percent_overflow}."
            )

        assert min_val < max_val

        self.rel_std_width = rel_std_width
        self.rel_std_height = rel_std_height
        self.clip_boxes = clip_boxes
        self.percent_overflow = percent_overflow
        self.std_overflow = std_overflow
        self.min_val = min_val
        self.max_val = max_val

        if self.std_overflow != 0 or self.percent_overflow != 0:
            self.altered_bboxes = True
        else:
            self.altered_bboxes = False

    def transform_to_density_map(
        self, bboxes_2d: List[Tensor], img_height: int, img_width: int
    ) -> Tuple[Tensor, Tensor]:
        density_maps, norm_density_maps = self.transform_to_regression_density_map(
            bboxes_2d, img_height, img_width
        )

        return density_maps, norm_density_maps

    def init_density_map(
        self, bboxes: Tensor, img_height: int, img_width: int
    ) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        # original position indices as the final map should have the same size as the original image
        orig_start_pos = (0, 0)
        orig_end_pos = (img_width, img_height)

        if not self.clip_boxes:
            x_left_offset = int(abs(min(bboxes[:, 0].min(), 0)))
            x_right_offset = int(max(img_width, bboxes[:, 2].max()) - img_width)
            y_top_offset = int(abs(min(bboxes[:, 1].min(), 0)))
            y_bottom_offset = int(max(img_height, bboxes[:, 3].max()) - img_height)

            # adjust img width and height based on overflowing offsets
            img_width += x_left_offset + x_right_offset
            img_height += y_top_offset + y_bottom_offset

            # correct bounding boxes (in case negative values in bbox are present)
            bboxes[:, [0, 2]] += x_left_offset
            bboxes[:, [1, 3]] += y_top_offset

            # adjust position indices
            orig_start_pos = (x_left_offset, y_top_offset)
            orig_end_pos = (img_width - x_right_offset, img_height - y_bottom_offset)

            # make sure img dimensions are int
            img_width = int(img_width)
            img_height = int(img_height)

        density_map = torch.zeros((img_width, img_height), device=bboxes.device)
        return density_map, orig_start_pos, orig_end_pos

    def extend_bboxes(self, bboxes: Tensor) -> Tensor:
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]

        if self.altered_bboxes:
            if self.percent_overflow != 0:
                abs_std_x = widths
                abs_std_y = heights
                overflow_measure = self.percent_overflow
            else:
                abs_std_x = widths / self.rel_std_width
                abs_std_y = heights / self.rel_std_height
                overflow_measure = self.std_overflow

            bboxes[:, 0] -= (overflow_measure * abs_std_x).to(int)
            bboxes[:, 1] -= (overflow_measure * abs_std_y).to(int)
            bboxes[:, 2] += (overflow_measure * abs_std_x).to(int)
            bboxes[:, 3] += (overflow_measure * abs_std_y).to(int)

        return bboxes

    def get_2d_gauss_kernel(self, bbox: Tensor) -> Tuple[Tensor, Tensor]:
        window_size_x = int(bbox[2] - bbox[0])  # equals box widths
        std_x = window_size_x / self.rel_std_width
        window_size_y = int(bbox[3] - bbox[1])  # equals box heights
        std_y = window_size_y / self.rel_std_height

        # older torch versions do not have the gauss window function yet, so use scipy instead
        kernel_x = signal.windows.gaussian(window_size_x, std=std_x)
        kernel_y = signal.windows.gaussian(window_size_y, std=std_y)

        # normalize the kernels
        max_kernel_x = stats.norm(0, std_x).pdf(0)
        max_kernel_y = stats.norm(0, std_y).pdf(0)

        kernel_2d = np.einsum("i,j", kernel_x, kernel_y)
        norm_kernel = kernel_2d * max_kernel_x * max_kernel_y

        # move kernel back to tensor w.r.t. the device
        kernel_2d = torch.tensor(kernel_2d, device=bbox.device)
        norm_kernel = torch.tensor(norm_kernel, device=bbox.device)
        return kernel_2d, norm_kernel

    def create_density_map(
        self, bboxes: Tensor, img_height: int, img_width: int
    ) -> Tuple[Tensor, Tensor]:
        """
        The density map is calculated in img coords not tensor coords -> it is flipped later
        """
        density_map, start_pos, end_pos = self.init_density_map(
            bboxes, img_height, img_width
        )

        if self.clip_boxes:
            bboxes[:, [0, 2]] = torch.clip(bboxes[:, [0, 2]], 0, img_width)
            bboxes[:, [1, 3]] = torch.clip(bboxes[:, [1, 3]], 0, img_height)

        norm_density_map = torch.zeros_like(density_map)
        for bbox in bboxes:
            heatmap_segment, norm_heatmap_segment = self.get_2d_gauss_kernel(bbox)
            density_map[bbox[0] : bbox[2], bbox[1] : bbox[3]] += heatmap_segment
            norm_density_map[bbox[0] : bbox[2], bbox[1] : bbox[3]] += (
                norm_heatmap_segment
            )

        # correct to original size
        if not self.clip_boxes:
            density_map = density_map[
                start_pos[0] : end_pos[0], start_pos[1] : end_pos[1]
            ]
            norm_density_map = norm_density_map[
                start_pos[0] : end_pos[0], start_pos[1] : end_pos[1]
            ]

        return density_map, norm_density_map

    def transform_to_regression_density_map(
        self, bboxes_2d: List[Tensor], img_height: int, img_width: int
    ) -> Tuple[Tensor, Tensor]:
        """
        bboxes_2d: as x1, y1, x2, y2 format
        img_shape: (width, height)
        """
        # convert boxes to int, as we operate on pixel basis
        density_maps = []
        norm_density_maps = []
        for bbox_batch in bboxes_2d:
            if len(bbox_batch) == 0:
                density_map = torch.zeros(
                    (img_height, img_width), device=bbox_batch.device
                )
                norm_density_map = torch.zeros_like(density_map)
                density_maps.append(density_map)
                norm_density_maps.append(norm_density_map)
                continue

            bbox_batch = bbox_batch.to(int)

            if self.percent_overflow != 0 or self.std_overflow != 0:
                bbox_batch = self.extend_bboxes(bbox_batch)

            density_map, norm_density_map = self.create_density_map(
                bbox_batch, img_height, img_width
            )

            # transpose to correct width/height flip into tensor format
            density_maps.append(density_map.T)
            norm_density_maps.append(norm_density_map.T)

        density_maps = torch.stack(density_maps)
        norm_density_maps = torch.stack(norm_density_maps)

        # clip density map in defined boundaries
        return density_maps, norm_density_maps
