# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
import json
from typing import List, Union
import logging
import os

from mmdet.registry import DATASETS
from mmdet.datasets import BaseDetDataset
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from mmengine.fileio import get_text


@DATASETS.register_module()
class STCrowdDataset2D(BaseDetDataset):
    METAINFO = {
        "classes": ("person",),
        # palette is a list of color tuples, which is used for visualization.
        "palette": [(220, 20, 60)],
    }

    def __init__(self, data_root: str, ann_file: str, **kwargs):
        super().__init__(data_root=data_root, ann_file=ann_file, **kwargs)

    def load_data_list(self) -> List[dict]:
        anno_strs = (
            get_text(self.ann_file, backend_args=self.backend_args).strip().split("\n")
        )
        print_log("loading STCrowd annotation...", level=logging.INFO)
        data_list = []
        prog_bar = ProgressBar(len(anno_strs))
        for i, anno_str in enumerate(anno_strs):
            anno_dict = json.loads(anno_str)
            parsed_data_info = self.parse_data_info(anno_dict)
            data_list.append(parsed_data_info)
            prog_bar.update()

        print_log("\nDone", level=logging.INFO)
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        data_info = {}
        img_path = os.path.join(
            self.data_prefix["img_path"], raw_data_info["rel_img_path"]
        )
        data_info["img_path"] = img_path
        data_info["img_id"] = raw_data_info["ID"]
        data_info["width"] = raw_data_info["img_width"]
        data_info["height"] = raw_data_info["img_height"]

        instances = []
        for bbox_ann in raw_data_info["gtboxes"]:
            if len(bbox_ann["fbox"]) == 0:
                continue

            # convert box to x1, y1, x2, y2
            x1, y1, w, h = bbox_ann["fbox"]
            bbox = [x1, y1, x1 + w, y1 + h]

            instance = {
                "bbox": bbox,
                "bbox_label": self.metainfo["classes"].index(bbox_ann["tag"]),
                "hbox": bbox_ann["hbox"],
                "occlusion": bbox_ann["occlusion"],
                "ignore_flag": 0,
            }
            instances.append(instance)

        data_info["instances"] = instances
        return data_info
