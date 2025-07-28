# ------------------------------------------------------------------------
# CrowdQuery
# Copyright (c) 2025 Marius Dähling. All Rights Reserved.
# ------------------------------------------------------------------------
from torch.utils import data
import pickle
import numpy as np
from PIL import Image
import copy
from typing import List
import torch
from .kitti_utils import (
    affine_transform,
    get_affine_transform,
)
from .pd import PhotometricDistort


def load_content(file_path):
    with open(file_path, "rb") as f:
        contents = pickle.load(f)
    return contents


def yaw_to_alpha(yaw, x_center, cam2img):
    f_u = cam2img[0, 0]  # focal length horizontal
    c_u = cam2img[0, 2]  # principal point
    view_ray_angles = np.arctan2(x_center - c_u, f_u)

    alpha = yaw - view_ray_angles
    alpha[alpha > np.pi] -= 2 * np.pi
    alpha[alpha < -np.pi] += 2 * np.pi
    return alpha


def alpha_to_yaw(alpha, x_center, cam2img):
    f_u = cam2img[0, 0]  # focal length horizontal
    c_u = cam2img[0, 2]  # principal point
    view_ray_angles = torch.atan2(x_center - c_u, f_u)

    yaw = alpha + view_ray_angles
    yaw[yaw > np.pi] -= 2 * np.pi
    yaw[yaw < -np.pi] += 2 * np.pi
    return yaw


def points_img2cam(points, cam2img):
    # Project points in image coordinates to camera coordinates. (from mm points_img_2cam)
    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=points.device)
    pad_cam2img[: cam2img.shape[0], : cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points_3d = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]
    return points_3d


def get_heading_angle(heading):
    # vectorized extension of base method from monodetr
    heading_bin, heading_res = heading[:, :12], heading[:, 12:24]
    cls = torch.argmax(heading_bin, dim=1)
    res = torch.gather(heading_res, 1, cls[:, None]).view(-1)
    return class_to_angle(cls, res, to_label_format=True)


def angle_to_class(angle):
    angle = angle % (2 * np.pi)
    num_heading_bin = 12  # hardcoded
    angle_per_class = 2 * np.pi / num_heading_bin
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = (shifted_angle / angle_per_class).astype(int)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class_to_angle(cls, residual, to_label_format=False):
    num_heading_bin = 12  # hardcoded
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format:
        angle[angle > np.pi] -= 2 * np.pi
    return angle


class STCrowd_Dataset(data.Dataset):
    def __init__(self, cfg: dict, split: str):
        self.dataset_name = "stcrowd"

        ann_file = f"{cfg['root_dir']}/annotation3d_{split}.pickle"
        self.data_dir = cfg["root_dir"]
        self.contents = load_content(ann_file)
        self.split = split

        # self.contents["data_samples"] = self.contents["data_samples"][:100]  # for debugging

        self.invalid_occlusions = cfg.get("invalid_occlusions", [])
        self.delete_sitting = cfg.get("delete_sitting", True)

        # imagenet stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # -------------------  monodetr specific  ----------------
        # general
        self.resolution = np.array([1280, 720])
        self.class_name = ["Pedestrian"]

        # augmentation
        self.data_augmentation = True if split in ["train", "trainval"] else False

        self.pd = PhotometricDistort()

        self.aug_pd = cfg.get("aug_pd", False)
        self.aug_crop = cfg.get("aug_crop", False)
        self.aug_calib = cfg.get("aug_calib", False)

        self.random_flip = cfg.get("random_flip", 0.5)
        self.random_crop = cfg.get("random_crop", 0.5)
        self.scale = cfg.get("scale", 0.4)
        self.shift = cfg.get("shift", 0.1)

    def __len__(self):
        return len(self.contents["data_samples"])

    @staticmethod
    def get_heading_encoding(yaw, bbox_2d, cam2img):
        x_center = (bbox_2d[:, 0] + bbox_2d[:, 2]) / 2
        alpha = yaw_to_alpha(yaw, x_center, cam2img)
        heading_bin, heading_res = angle_to_class(alpha)
        return heading_bin, heading_res

    def forward_pipeline(self, img, instance_data, img_data, cam2img):
        ################# augmentation selection and images augmentation ######################
        img_size = np.array([img_data["width"], img_data["height"]])
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag = False

        if self.data_augmentation:
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    crop_scale = np.clip(
                        np.random.randn() * self.scale + 1,
                        1 - self.scale,
                        1 + self.scale,
                    )
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(
                        np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift
                    )
                    center[1] += img_size[1] * np.clip(
                        np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift
                    )

        trans, trans_inv = get_affine_transform(
            center, crop_size, 0, self.resolution, inv=1
        )
        img = img.transform(
            tuple(self.resolution.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR,
        )

        ############  labels ##################
        for instance in instance_data:
            bbox_2d = instance["bbox_2d"]
            center_3d = instance["proj_3d_center"]
            yaw = instance["bbox_3d"][-1]

            # apply flipping if used
            if random_flip_flag:
                bbox_2d_orig = copy.deepcopy(bbox_2d)
                bbox_2d[0] = img_size[0] - bbox_2d_orig[2]
                bbox_2d[2] = img_size[0] - bbox_2d_orig[0]
                center_3d[0] = img_size[0] - center_3d[0]

                yaw = np.pi - yaw
                if yaw > np.pi:
                    yaw -= 2 * np.pi
                if yaw < -np.pi:
                    yaw += 2 * np.pi

            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # reassign values
            instance["bbox"] = bbox_2d
            instance["proj_3d_center"] = center_3d
            instance["bbox_3d"][-1] = yaw

        if random_flip_flag:
            cam2img[0, 2] = img_data["width"] - cam2img[0, 2]

        return img, instance_data, crop_scale, cam2img

    def prepare_output(self, instance_data, crop_scale, cam2img):
        if len(instance_data) == 0:
            return {
                "labels": np.zeros(shape=(0,), dtype=np.int64),
                "boxes_3d": np.zeros(shape=(0, 6), dtype=np.float32),
                "boxes": np.zeros(shape=(0, 4), dtype=np.float32),
                "depth": np.zeros(shape=(0, 1), dtype=np.float32),
                "size_3d": np.zeros(shape=(0, 3), dtype=np.float32),
                "heading_res": np.zeros(shape=(0, 1), dtype=np.float32),
                "heading_bin": np.zeros(shape=(0, 1), dtype=np.int64),
                "mask_2d": np.zeros(shape=(0,), dtype=bool),
            }

        boxes_2d = np.array([instance["bbox_2d"] for instance in instance_data])
        boxes_3d = np.array([instance["bbox_3d"] for instance in instance_data])
        proj_3d_centers = np.array(
            [instance["proj_3d_center"] for instance in instance_data]
        )
        labels = np.array([instance["label"] for instance in instance_data])
        size_3d = boxes_3d[:, 3:6]

        # heading
        yaw = boxes_3d[:, -1]
        heading_bin, heading_res = self.get_heading_encoding(yaw, boxes_2d, cam2img)

        # encode 3d boxes
        l = proj_3d_centers[:, 0] - boxes_2d[:, 0]
        r = boxes_2d[:, 2] - proj_3d_centers[:, 0]
        t = proj_3d_centers[:, 1] - boxes_2d[:, 1]
        b = boxes_2d[:, 3] - proj_3d_centers[:, 1]
        encoded_boxes_3d = np.stack(
            [proj_3d_centers[:, 0], proj_3d_centers[:, 1], l, r, t, b], axis=1
        )

        # normalize bbox coords
        boxes_2d[:, [0, 2]] /= self.resolution[0]
        boxes_2d[:, [1, 3]] /= self.resolution[1]
        encoded_boxes_3d[:, [0, 2, 3]] /= self.resolution[0]
        encoded_boxes_3d[:, [1, 4, 5]] /= self.resolution[1]

        # convert 2d bboxes from x1,y1,x2,y2 -> x_center, y_center, w, h
        centers = (boxes_2d[:, :2] + boxes_2d[:, 2:]) / 2
        dims_2d = boxes_2d[:, 2:] - boxes_2d[:, :2]
        boxes_2d = np.concatenate([centers, dims_2d], axis=1)

        # depth scaling
        depths = boxes_3d[:, 2]
        depths *= crop_scale

        # extra filter to drop invalid samples
        mask = (encoded_boxes_3d[:, 2:] > 0).all(axis=1)
        # apply mask to all variables
        labels = labels[mask]
        encoded_boxes_3d = encoded_boxes_3d[mask]
        boxes_2d = boxes_2d[mask]
        depths = depths[mask]
        size_3d = size_3d[mask]
        heading_res = heading_res[mask]
        heading_bin = heading_bin[mask]

        targets = {
            "labels": labels,
            "boxes_3d": encoded_boxes_3d.astype(np.float32),
            "boxes": boxes_2d.astype(np.float32),
            "depth": depths.astype(np.float32)[:, None],
            "size_3d": size_3d.astype(np.float32),
            "heading_res": heading_res.astype(np.float32)[:, None],
            "heading_bin": heading_bin[:, None],
            "mask_2d": np.ones_like(labels, dtype=bool),
        }
        return targets

    def filter_instances(self, instance_data):
        instance_data = self.filter_outside(instance_data)
        if len(self.invalid_occlusions) != 0:
            instance_data = self.filter_occlusion(instance_data)
        if self.delete_sitting:
            instance_data = self.filter_sitting_person(instance_data)
        return instance_data

    def filter_outside(self, instance_data):
        filtered_instance_data = []
        for instance in instance_data:
            inside_img = True
            center_3d = instance["proj_3d_center"]
            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                inside_img = False
            elif center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                inside_img = False

            if inside_img:
                filtered_instance_data.append(instance)

        return filtered_instance_data

    @staticmethod
    def filter_sitting_person(instance_data):
        filtered_instance_data = []
        for instance in instance_data:
            if instance["bbox_3d"][4] < 1.2:
                continue
            filtered_instance_data.append(instance)

        return filtered_instance_data

    def filter_occlusion(self, instance_data) -> List:
        filtered_instance_data = []
        for instance in instance_data:
            if instance["occlusion"] not in self.invalid_occlusions:
                filtered_instance_data.append(instance)

        return filtered_instance_data

    def __getitem__(self, item):
        data_item = self.contents["data_samples"][item]

        # img data
        img_path = data_item["images"]["img_path"]
        img = Image.open(img_path)
        imd_id = data_item["images"]["img_id"]

        # label data
        instance_data = copy.deepcopy(data_item["instances"])
        img_data = copy.deepcopy(data_item["images"])

        cam2img = img_data["P"]

        # augment data
        img, instance_data, crop_scale, cam2img = self.forward_pipeline(
            img, instance_data, img_data, cam2img
        )

        # filter data
        instance_data = self.filter_instances(instance_data)

        # prepare output
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        # img info
        img_size = np.array([img_data["width"], img_data["height"]])
        info = {"img_size": img_size, "img_id": imd_id}

        targets = self.prepare_output(instance_data, crop_scale, cam2img)
        targets["img_size"] = img_size

        return img, cam2img, targets, info
