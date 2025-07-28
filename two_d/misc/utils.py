# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from typing import List, Tuple


def mask_to_shape(mask: torch.Tensor) -> List[Tuple]:
    mask_int = mask.to(torch.int)  # for argmin
    h_origs = mask_int[:, :, 0].argmin(dim=1)
    h_origs[h_origs == 0] = mask_int.shape[1]
    w_origs = mask_int[:, 0].argmin(dim=1)
    w_origs[w_origs == 0] = mask_int.shape[2]
    density_map_shapes = [
        (int(h_orig), int(w_orig)) for h_orig, w_orig in zip(h_origs, w_origs)
    ]
    return density_map_shapes
