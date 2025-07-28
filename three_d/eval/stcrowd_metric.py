# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List, Tuple, Dict
import numpy as np
from abc import abstractmethod
import itertools


class STCrowdMetric:
    def __init__(self):
        # defined by STCrowd
        self.dist_thresholds = [0.25, 0.5, 1]

        # defined by Nuscenes
        self.min_recall = 0.1
        self.min_precision = 0.1

    @staticmethod
    def combine_sorted_preds(preds):
        # sort predictions by detection score desc (highest score first) across all imgs
        combined_preds = list(itertools.chain.from_iterable(preds))
        combined_preds = sorted(combined_preds, key=lambda x: x["score"], reverse=True)
        return combined_preds

    @abstractmethod
    def compute(self, gts: List[Dict], preds: List[Dict]) -> float:
        raise NotImplementedError

    @staticmethod
    def calc_center_dist(gt_box: np.array, pred_box: np.array) -> float:
        # l2 dist in xy coords
        return np.linalg.norm(pred_box[[0, 2]] - gt_box[[0, 2]])

    def match(
        self, gts: List[List[Dict]], preds: List[Dict], dist_threshold: float
    ) -> Tuple[List[int], List[int]]:
        # copy gts and preds to avoid any changes
        gts = copy.deepcopy(gts)
        preds = copy.deepcopy(preds)  # preds already sorted

        # set all gts as unmatched
        for gt_item in gts:
            for gt in gt_item:
                gt["matched"] = False

        tps = []
        fps = []
        for pred in preds:
            pred_box = pred["bbox_3d"]
            min_dist = np.inf
            match_gt_idx = None

            selected_gts = gts[pred["idx"]]

            for gt_idx, gt in enumerate(selected_gts):
                gt_box = gt["bbox_3d"]

                if not gt["matched"]:
                    dist = self.calc_center_dist(gt_box, pred_box)
                    if dist < min_dist:
                        min_dist = dist
                        match_gt_idx = gt_idx

            is_match = min_dist < dist_threshold

            if is_match:
                tps.append(1)
                fps.append(0)
                # set used gt to matched
                selected_gts[match_gt_idx]["matched"] = True
            else:
                tps.append(0)
                fps.append(1)

        return tps, fps


class STCrowdAR(STCrowdMetric):
    def __init__(self):
        super().__init__()
        # defined by STCrowd
        self.occlusions = [0, 1, 2]

    def compute(
        self, gts: List[List[dict]], preds: List[List[dict]]
    ) -> Dict[str, float]:
        recalls = {str(occlusion): [] for occlusion in self.occlusions}

        # append index
        for idx, pred_item in enumerate(preds):
            for entry in pred_item:
                entry["idx"] = idx

        combined_preds = self.combine_sorted_preds(preds)

        for occlusion in self.occlusions:
            # filter gts based on occlusion
            filtered_gts = [
                [gt for gt in gt_item if gt["occlusion"] == occlusion]
                for gt_item in gts
            ]
            n_gts = len(
                list(itertools.chain.from_iterable(filtered_gts))
            )  # length of flattened list

            for dist in self.dist_thresholds:
                tp, _ = self.match(filtered_gts, combined_preds, dist)
                # get total tps
                total_tp = sum(tp)

                recall = total_tp / n_gts
                recalls[str(occlusion)].append(recall)

        avg_recalls = {f"ar_{key}": np.mean(val) for key, val in recalls.items()}
        return avg_recalls


class STCrowdAP(STCrowdMetric):
    def calc_ap(self, precision: np.array) -> float:
        """copied from nuscenes devkit"""
        prec = precision[
            round(100 * self.min_recall) + 1 :
        ]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= self.min_precision  # Clip low precision
        prec[prec < 0] = 0
        return float(np.mean(prec)) / (1.0 - self.min_precision)

    def compute(self, gts: List[List[dict]], preds: List[List[dict]]) -> dict:
        avg_precisions = {}
        n_gts = len(list(itertools.chain.from_iterable(gts)))

        combined_preds = self.combine_sorted_preds(preds)

        # iterate over distance thresholds
        for dist in self.dist_thresholds:
            # match preds to gts and get tps, fps + confs
            tp, fp = self.match(gts, combined_preds, dist)

            if len(tp) == 0:
                tp = np.array([0])
            if len(fp) == 0:
                fp = np.array([0])

            # calculate precision and recall
            tp = np.cumsum(tp).astype(float)
            fp = np.cumsum(fp).astype(float)

            precision = tp / (fp + tp)
            recall = tp / n_gts

            # calculate interpolated value at 101 steps between 0 and 1
            interp_steps = np.linspace(0, 1, 101)
            precision = np.interp(interp_steps, recall, precision, right=0)

            avg_precision = self.calc_ap(precision)
            avg_precisions[f"ap_{str(dist)}"] = avg_precision

        # average over dist thresholds
        avg_precisions["m_ap"] = np.mean(list(avg_precisions.values()))

        return avg_precisions
