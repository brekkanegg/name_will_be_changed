import pytorch_lightning as pl

import torch
import numpy as np
import pandas as pd
import os, sys
import timm
import torchmetrics

import segmentation_models_pytorch as smp


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_ddp = len(self.cfg.gpus) > 1
        self.save_hyperparameters()

        # self.model = timm.create_model(
        #     cfg.model,
        #     pretrained=cfg.pretrained,
        #     in_chans=cfg.in_channels,
        #     num_classes=4,
        # )

        aux_params = {
            "classes": 1,
            "pooling": "avg",
            "dropout": self.cfg.drop_rate,
            "activation": None,
        }
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b7",
            encoder_weights="noisy-student",
            in_channels=3,
            classes=1,
            activation=None,
            aux_params=aux_params,
        )

        self.seg_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()
        self.cls_criterion = torch.nn.BCEWithLogitsLoss()

        self.train_ap = torchmetrics.AveragePrecision(num_classes=1, pos_label=0)
        self.val_ap = torchmetrics.AveragePrecision(num_classes=1, pos_label=0)

        self.test_preds = []

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def configure_optimizers(self):
        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        elif self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.cfg.lr, momentum=0.9
            )

        if self.cfg.scheduler == "cosineWR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, T_0=5, T_mult=1
            )
        elif self.cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.cfg.max_epochs
            )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "strict": True,
            "name": "learning_rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        img, label, mask, img_path = batch

        seg_logit, cls_logit = self.model(img)

        seg_loss = self.seg_criterion(seg_logit[:, 0, :, :], mask)
        cls_loss = self.cls_criterion(cls_logit, label)

        ap = self.train_ap(torch.sigmoid(cls_logit), label)
        # For Callback Drawing later
        # self.last_train_batchs = img, label, img_path
        # self.last_train_logits = logit

        loss = self.cfg.aux_weight * seg_loss + cls_loss

        log_dict = {
            "loss/train/seg": seg_loss,
            "loss/train/cls": cls_loss,
            "ap/train": ap,
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.use_ddp,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        img, label, mask, img_path = batch
        # img = img.permute(0, 3, 1, 2)

        seg_logit, cls_logit = self.model(img)

        seg_loss = self.seg_criterion(seg_logit[:, 0, :, :], mask)
        cls_loss = self.cls_criterion(cls_logit, label)

        # ap = self.val_ap(torch.softmax(cls_logit, axis=1), torch.argmax(label, axis=1))
        ap = self.val_ap(torch.sigmoid(cls_logit), label)
        # _map = np.nanmean([i.detach().cpu().item() for i in ap])

        # For CallBack Drawing
        # self.last_val_batchs = img, label, img_path
        # self.last_val_logits = logit

        log_dict = {"loss/valid/seg": seg_loss, "loss/valid/cls": cls_loss}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.use_ddp,
        )

        return {"logit": (cls_logit), "batch": ((img, label, mask, img_path))}

    def validation_epoch_end(self, validation_step_outputs):

        # FIXME: custom_metrics not working for DDP case
        # if self.use_ddp:
        #     return

        # acc_result = self.val_acc.compute()
        # ap_result = self.val_ap.compute()
        ap_result = self.val_ap.compute().detach().cpu().item()
        # map_result = np.mean(ap_result)

        log_dict = {
            # "vacc": acc_result,
            "ap/valid": ap_result,
            # "ap/valid/0": ap_result[0],
            # "ap/valid/1": ap_result[1],
            # "ap/valid/2": ap_result[2],
            # "ap/valid/3": ap_result[3],
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.use_ddp,
        )

    # FIXME:
    def test_step(self, batch, batch_idx):
        img, img_path = batch
        logit = self.model(img)
        pred = torch.sigmoid(logit).detach().cpu().numpy().tolist()
        self.test_preds.append((img_path, pred))

    # def test_epoch_end(self, test_step_outputs):
    #     return self.test_preds
