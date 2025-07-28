# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from torch import Tensor
import torch
from torch import nn
from typing import List, Optional


class DensityLoss(nn.Module):
    def __init__(self, balanced_loss: bool, balance_factor: float):
        super().__init__()
        self.balanced_loss = balanced_loss
        self.balance_factor = balance_factor

    def balance_loss(
        self, loss: Tensor, pred_densities: Tensor, bboxes_2d: List[Tensor]
    ):
        batch_size, img_height, img_width = pred_densities.shape[:3]
        weights = self.get_balance_loss_weights(
            batch_size, img_height, img_width, bboxes_2d
        )
        loss *= weights
        return loss

    def get_balance_loss_weights(
        self, batch_size: int, img_height: int, img_width: int, bboxes_2d: List[Tensor]
    ) -> Tensor:
        foreground_mask = self.get_foreground_mask(
            batch_size, img_height, img_width, bboxes_2d
        )
        background_mask = ~foreground_mask
        weights = foreground_mask * self.balance_factor + background_mask
        return weights

    @staticmethod
    def get_foreground_mask(
        batch_size: int, img_height: int, img_width: int, bboxes_2d_list: List[Tensor]
    ) -> Tensor:
        # coords in x1, y1, x2, y2 img format (global NOT norm)
        device = bboxes_2d_list[0].device

        foreground_mask = torch.zeros(
            (batch_size, img_height, img_width), dtype=torch.bool, device=device
        )

        for batch_idx, bboxes_2d in enumerate(bboxes_2d_list):
            bboxes_2d = bboxes_2d.to(int)
            for bbox_2d in bboxes_2d:
                x1, y1, x2, y2 = bbox_2d
                # x and y coords are switched in array indexing compared to image axis
                foreground_mask[batch_idx, y1:y2, x1:x2] = True

        return foreground_mask

    @staticmethod
    def normalize_loss(loss: Tensor, mask: Optional[Tensor]) -> Tensor:
        # normalize by the size of the density map
        if mask is not None:
            n_pixels = mask.sum(dim=[1, 2])  # pixel per img
            loss /= n_pixels[:, None, None]
        else:
            n_pixels = torch.prod(torch.tensor(loss.shape[1:3]))
            loss /= n_pixels

        n_img = loss.shape[0]
        loss /= n_img

        return loss


class RegressionDensityLoss(DensityLoss):
    def __init__(self, balanced_loss: bool = True, balance_factor: float = 5):
        super().__init__(balanced_loss=balanced_loss, balance_factor=balance_factor)
        self.base_loss_func = nn.L1Loss(reduction="none")

    def compute_loss(
        self,
        pred_density: Tensor,
        target_density: Tensor,
    ) -> Tensor:
        base_loss = self.base_loss_func(pred_density, target_density)

        loss = base_loss

        return loss

    def forward(
        self,
        pred_densities: Tensor,
        target_densities: Tensor,
        bboxes_2d: List[Tensor],
        mask: Tensor = None,
    ) -> Tensor:
        if mask is not None:
            pred_densities = pred_densities * mask
            target_densities = (
                target_densities * mask
            )  # not needed in theory, as the padding was taken into account while creating

        loss = self.compute_loss(pred_densities, target_densities)
        if self.balanced_loss:
            loss = self.balance_loss(loss, pred_densities, bboxes_2d)

        loss = self.normalize_loss(loss, mask)
        # aggregate loss
        loss = loss.sum()
        return loss
