# ------------------------------------------------------------------------------------------------
# Copyright (c) MonoDETR
# ------------------------------------------------------------------------------------------------
# modified by CrowdQuery: add density components
# ------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from ..dataset.stcrowd import STCrowd_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def custom_collate_fn(batch):
    imgs, cam2img, targets, info = zip(*batch)

    # Convert to tensors
    imgs = torch.tensor(imgs)
    cam2img = torch.tensor(cam2img)

    formatted_info = {}
    for key in info[0].keys():
        if key != "img_id":
            formatted_info[key] = torch.tensor([item[key] for item in info])
        else:
            formatted_info[key] = [item[key] for item in info]

    padded_targets = {}
    for key in targets[0].keys():
        key_items = [torch.tensor(item[key]) for item in targets]
        padded_key_items = pad_sequence(key_items, batch_first=True)
        padded_targets[key] = padded_key_items

    return imgs, cam2img, padded_targets, formatted_info


def build_dataloader(cfg, workers=4):
    # perpare dataset
    not_kitti = True
    if cfg["type"] == "stcrowd":
        train_set = STCrowd_Dataset(split=cfg["train_split"], cfg=cfg)
        test_set = STCrowd_Dataset(split=cfg["test_split"], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg["type"])

    # prepare dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg["batch_size"],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        collate_fn=custom_collate_fn if not_kitti else None,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg["batch_size"],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        collate_fn=custom_collate_fn if not_kitti else None,
    )

    return train_loader, test_loader
