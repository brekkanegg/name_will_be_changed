import pytorch_lightning as pl

import torch
import numpy as np
import pandas as pd
import os, sys
import timm
import torchmetrics

# import segmentation_models_pytorch as smp


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_ddp = len(self.cfg.gpus) > 1
        self.save_hyperparameters()

        self.model = timm.create_model(
            cfg.model, pretrained=cfg.pretrained, in_chans=1, num_classes=4
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()
        self.train_metric = torchmetrics.Accuracy()
        self.val_metric = torchmetrics.Accuracy()

        self.test_preds = []

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=5, T_mult=1
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
        img, label, img_path = batch
        # img = img.permute(0, 3, 1, 2)

        logit = self.model(img)

        acc = self.train_metric(torch.sigmoid(logit), label.int())

        # For Callback Drawing later
        # self.last_train_batchs = img, label, img_path
        # self.last_train_logits = logit

        loss = self.criterion(logit, label)

        log_dict = {"tloss": loss, "tacc": acc}
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
        img, label, img_path = batch
        # img = img.permute(0, 3, 1, 2)

        logit = self.model(img)

        # For CallBack Drawing
        # self.last_val_batchs = img, label, img_path
        # self.last_val_logits = logit

        # Metric

        loss = self.criterion(logit, label)
        acc = self.val_metric(torch.sigmoid(logit), label.int())

        log_dict = {"vloss": loss}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.use_ddp,
        )

        return {"logit": (logit), "batch": ((img, label, img_path))}

    # def validation_step_end(self, outputs):
    #     """
    #     If using metrics in data parallel mode (dp),
    #     the metric update/logging should be done in the <mode>_step_end method
    #     (where <mode> is either training, validation or test).
    #     This is due to metric states else being destroyed after each forward pass,
    #     leading to wrong accumulation.
    #     """
    #     logit = outputs["logit"]
    #     _, label, _ = outputs["batch"]

    #     pred = torch.sigmoid(logit)
    #     self.val_metric(pred, label.int())

    def validation_epoch_end(self, validation_step_outputs):

        # FIXME: custom_metrics not working for DDP case
        # if self.use_ddp:
        #     return

        metric_result = self.val_metric.compute()
        self.log(
            "vacc",
            metric_result,
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
