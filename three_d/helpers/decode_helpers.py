# ------------------------------------------------------------------------------------------------
# Copyright (c) MonoDETR
# ------------------------------------------------------------------------------------------------
# modified by CrowdQuery: increase topk from 50 to 100
# ------------------------------------------------------------------------------------------------
import torch
from ..utils import box_ops


def extract_dets_from_outputs(outputs, K=100, topk=100):
    # get src outputs

    # b, q, c
    out_logits = outputs["pred_logits"]
    out_bbox = outputs["pred_boxes"]

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits.shape[0], -1), topk, dim=1
    )

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]

    heading = outputs["pred_angle"]
    size_3d = outputs["pred_3d_dim"]
    depth = outputs["pred_depth"][:, :, 0:1]
    sigma = outputs["pred_depth"][:, :, 1:2]
    sigma = torch.exp(-sigma)

    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4

    xs3d = boxes[:, :, 0:1]
    ys3d = boxes[:, :, 1:2]

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes)
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2:4]

    xs2d = xywh_2d[:, :, 0:1]
    ys2d = xywh_2d[:, :, 1:2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = torch.cat(
        [
            labels,
            scores,
            xs2d,
            ys2d,
            size_2d,
            depth,
            heading,
            size_3d,
            xs3d,
            ys3d,
            sigma,
        ],
        dim=2,
    )

    return detections
