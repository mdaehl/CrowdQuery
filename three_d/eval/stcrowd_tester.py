# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from .stcrowd_metric import STCrowdAR, STCrowdAP
import torch
import numpy as np
import tqdm
import os
import pickle
from three_d.dataset.stcrowd import (
    points_img2cam,
    alpha_to_yaw,
    get_heading_angle,
    load_content,
)
from ..helpers.decode_helpers import extract_dets_from_outputs
from ..helpers.save_helper import load_checkpoint


class STCrowdTester:
    def __init__(self, cfg, model, dataloader, ckpt_path=None, logger=None):
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.checkpoint_path = ckpt_path
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "/tmp/"
        self.results_file = f"{self.output_dir}/results.pickle"

        self.ar_metric = STCrowdAR()
        self.ap_metric = STCrowdAP()

        self.max_detections = np.inf
        self.threshold = self.cfg.get("threshold", 0.2)

    def inference(self, *args, **kwargs):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(
            total=len(self.dataloader), leave=True, desc="Evaluation Progress"
        )

        for batch_idx, (inputs, cam2imgs, targets, info) in enumerate(self.dataloader):
            inputs = inputs.to(self.device)
            cam2imgs = cam2imgs.to(self.device)
            info = {
                key: val.to(self.device) if key != "img_id" else val
                for key, val in info.items()
            }

            img_sizes = info["img_size"].to(self.device)
            dataset = self.dataloader.dataset.dataset_name
            outputs = self.model(inputs, cam2imgs, targets, img_sizes, dataset=dataset)

            dets = extract_dets_from_outputs(
                outputs=outputs, topk=self.model.num_queries
            )  # select all queries
            dets = self.decode_detections(
                dets=dets, cam2imgs=cam2imgs, info=info, threshold=self.threshold
            )

            results.update(dets)
            progress_bar.update()

        combined_results = {"data_samples": results}
        self.save_results(combined_results)

    def save_results(self, results: dict):
        with open(self.results_file, "wb") as f:
            pickle.dump(results, f)

    @staticmethod
    def decode_detections(dets, cam2imgs, info, threshold):
        cls_ids = dets[:, :, 0].to(torch.int)
        scores = dets[:, :, 1]
        mask = scores >= threshold

        # scale score with uncertainty
        scores *= dets[:, :, -1]

        # img sizes to tensor
        # process detections
        # get img coords
        x = dets[:, :, 2] * info["img_size"][:, 0, None]
        y = dets[:, :, 3] * info["img_size"][:, 1, None]
        w = dets[:, :, 4] * info["img_size"][:, 0, None]
        h = dets[:, :, 5] * info["img_size"][:, 1, None]
        bboxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=2)

        # get 3d points
        depths = dets[:, :, 6]
        xys_center_3d = dets[:, :, 34:36] * info["img_size"][:, None, :]
        # add depth
        xys_center_3d = torch.cat([xys_center_3d, depths[:, :, None]], dim=2)

        n_batch = dets.shape[0]
        n_pred = dets.shape[1]
        locations = []
        yaws = []
        for i in range(n_batch):
            location_set = points_img2cam(xys_center_3d[i], cam2imgs[i])

            alpha = get_heading_angle(dets[i, :, 7:31])
            yaw = alpha_to_yaw(alpha, x[i], cam2imgs[i])

            locations.append(location_set)
            yaws.append(yaw)

        locations = torch.stack(locations)
        yaws = torch.stack(yaws)

        # 3d boxes
        dimensions = dets[:, :, 31:34]
        bboxes_3d = torch.cat([locations, dimensions, yaws[:, :, None]], dim=2)

        results = {}
        for i in range(n_batch):
            preds = []
            for j in range(n_pred):
                if mask[i, j]:
                    pred = {
                        "bbox_2d": bboxes[i, j],
                        "bbox_3d": bboxes_3d[i, j],
                        "label": cls_ids[i, j],
                        "score": scores[i, j],
                    }
                    preds.append(pred)

            results[info["img_id"][i]] = preds

        return results

    def evaluate(self):
        gt_contents = self.dataloader.dataset.contents

        gt_data = gt_contents["data_samples"]
        pred_data = load_content(self.results_file)["data_samples"]

        gt_indices = [item["images"]["img_id"] for item in gt_data]

        gt_instances = [item["instances"] for item in gt_data]
        pred_instances = [pred_data[gt_idx] for gt_idx in gt_indices]

        # move pred instances to cpu and convert to numpy
        pred_instances = [
            [
                {key: val.cpu().numpy() for key, val in pred.items()}
                for pred in pred_item
            ]
            for pred_item in pred_instances
        ]

        ar = self.ar_metric.compute(gt_instances, pred_instances)
        ap = self.ap_metric.compute(gt_instances, pred_instances)

        metrics = {}
        metrics.update(ap)
        metrics.update(ar)
        print(metrics)

        return ap["m_ap"]

    def test(self):
        assert os.path.exists(self.checkpoint_path)
        load_checkpoint(model=self.model,
                        optimizer=None,
                        filename=self.checkpoint_path,
                        map_location=self.device,
                        logger=self.logger)
        self.model.to(self.device)
        self.inference()
        self.evaluate()