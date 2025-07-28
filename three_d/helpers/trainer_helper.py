# ------------------------------------------------------------------------------------------------
# Copyright (c) MonoDETR
# ----------------------------------------------------------------------------------------------
import os
import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np

from .save_helper import get_checkpoint_state
from .save_helper import load_checkpoint

from utils import misc


class Trainer(object):
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        train_loader,
        test_loader,
        lr_scheduler,
        warmup_lr_scheduler,
        logger,
        loss,
        model_name,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.tester = None
        self.clip_grad = False

        # loading pretrain/resume model
        if cfg.get("pretrain_model"):
            assert os.path.exists(cfg["pretrain_model"])
            load_checkpoint(
                model=self.model,
                optimizer=None,
                filename=cfg["pretrain_model"],
                map_location=self.device,
                logger=self.logger,
            )

        if cfg.get("resume_model", None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger,
            )
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info(
                "Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(
                    self.best_result, self.best_epoch
                )
            )

    def train(self):
        from torch.utils.tensorboard import SummaryWriter
        from time import time

        output_dir = "./logs"
        writer_path = os.path.join(output_dir, ("timestamp_%s" % time()))
        writer = SummaryWriter(log_dir=writer_path)
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(
            range(start_epoch, self.cfg["max_epoch"]),
            dynamic_ncols=True,
            leave=True,
            desc="epochs",
        )
        best_result = self.best_result
        best_epoch = self.best_epoch

        for epoch in range(start_epoch, self.cfg["max_epoch"]):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch, writer)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg["save_frequency"]) == 0:
                os.makedirs(self.output_dir, exist_ok=True)

                checkpoint_state = get_checkpoint_state(
                    self.model, self.optimizer, self.epoch, best_result, best_epoch
                )
                torch.save(checkpoint_state, f"{self.output_dir}/model_{self.epoch}.pth")

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference(epoch, writer)
                    cur_result = self.tester.evaluate(epoch, writer)
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        checkpoint_state = get_checkpoint_state(
                            self.model,
                            self.optimizer,
                            self.epoch,
                            best_result,
                            best_epoch,
                        )
                        torch.save(
                            checkpoint_state, f"{self.output_dir}/best_model.pth"
                        )

                    self.logger.info(
                        "Best Result:{}, epoch:{}".format(best_result, best_epoch)
                    )

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        writer.close()

        return None

    def train_one_epoch(self, epoch, writer):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(
            total=len(self.train_loader),
            leave=(self.epoch + 1 == self.cfg["max_epoch"]),
            desc="iters",
        )
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            img_sizes = targets["img_size"]
            targets = self.prepare_targets(targets, inputs.shape[0])
            ##dn
            dn_args = None
            if self.cfg["use_dn"]:
                dn_args = (
                    targets,
                    self.cfg["scalar"],
                    self.cfg["label_noise_scale"],
                    self.cfg["box_noise_scale"],
                    self.cfg["num_patterns"],
                )
            ###
            # train one batch
            self.optimizer.zero_grad()
            dataset = self.train_loader.dataset.dataset_name
            outputs = self.model(
                inputs, calibs, targets, img_sizes, dn_args=dn_args, dataset=dataset
            )
            mask_dict = None
            # ipdb.set_trace()
            detr_losses_dict = self.detr_loss(outputs, targets, mask_dict, info)

            weight_dict = self.detr_loss.weight_dict
            detr_losses_dict_weighted = [
                detr_losses_dict[k] * weight_dict[k]
                for k in detr_losses_dict.keys()
                if k in weight_dict
            ]
            detr_losses = sum(detr_losses_dict_weighted)

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (
                        detr_losses_dict[k] * weight_dict[k]
                    ).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            log_epoch = batch_idx + len(self.train_loader) * epoch
            for key, val in detr_losses_dict_log.items():
                writer.add_scalar(key, val, log_epoch)
            lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
            writer.add_scalar("bias_lr", lrs[0], log_epoch)
            writer.add_scalar("weight_lr", lrs[1], log_epoch)

            flags = [True] * 5
            if batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " % ("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if (
                        "0" in key
                        or "1" in key
                        or "2" in key
                        or "3" in key
                        or "4" in key
                        or "5" in key
                    ):
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " % (key, val), end="")
                print("")
                print("")

            detr_losses.backward()

            if self.clip_grad:
                params = []
                for param_group in self.optimizer.param_groups:
                    params.extend(param_group["params"])

                params = list(
                    filter(lambda p: p.requires_grad and p.grad is not None, params)
                )
                norm = clip_grad_norm_(params, max_norm=0.1)

            else:
                grads = [
                    param.grad.detach().flatten()
                    for param in self.model.parameters()
                    if param.grad is not None
                ]
                norm = torch.cat(grads).norm()

            self.optimizer.step()

            progress_bar.update()
        progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets["mask_2d"]

        key_list = [
            "labels",
            "boxes",
            "calibs",
            "depth",
            "size_3d",
            "heading_bin",
            "heading_res",
            "boxes_3d",
        ]
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list
