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
            cfg.model,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            num_classes=4,
        )
        if self.cfg.dropblock:
            print('Dropping last block to prevent overfitting')
            #self.model.blocks[-1] = nn.Identity()
            #in_features = self.model.blocks[-2][-1].bn3.num_features
            self.model.blocks[6] = torch.nn.Identity()
            in_features = 384  # this works only for tf_efficientnetv2_l_in21ft1k maybe
            self.model.conv_head = torch.nn.Conv2d(
                in_features, in_features * 2,
                kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.model.bn2 = torch.nn.BatchNorm2d(
                in_features * 2, eps=0.001, momentum=0.1, affine=True,
                track_running_stats=True)
            
            self.model.classifier = torch.nn.Linear(
                in_features * 2, 4)
        else:
            in_features = self.model.classifier.in_features

        if self.cfg.drop_rate > 0: # This only works for efficientnet in timm maybe
            self.model.drop_rate = self.cfg.drop_rate

        if self.cfg.loss == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss()  
            if self.cfg.pos_weight:
                pw = torch.FloatTensor([float(i) for i in self.cfg.pos_weight]).cuda()
                self.cls_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        elif self.cfg.loss == "ce":
            self.criterion = torch.nn.CrossEntropyLoss()
            if self.cfg.pos_weight:
                pw = torch.FloatTensor([float(i) for i in self.cfg.pos_weight]).cuda()
                self.cls_criterion = torch.nn.CrossEntropyLoss(weight=pw)

        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()

        self.train_ap = torchmetrics.AveragePrecision(num_classes=4)
        self.val_ap = torchmetrics.AveragePrecision(num_classes=4)

        self.test_preds = []

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def configure_optimizers(self):
        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay
            )

        if self.cfg.scheduler == "cosineWR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, T_0=5, T_mult=1
            )
        elif self.cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.cfg.max_epochs
            )
        # elif self.cfg.scheduler == "plateau":
        #     scheduler = {
        #         "optimizer": optimizer,
        #         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        #             optimizer=optimizer, mode="min", factor=0.1
        #         ),
        #         "monitor": "vloss",
        #     }

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

        if self.cfg.loss == "bce":
            # acc = self.train_acc(torch.sigmoid(logit), label.int())
            ap = self.train_ap(torch.sigmoid(logit), torch.argmax(label, axis=1))
            loss = self.criterion(logit, label)

        elif self.cfg.loss == "ce":
            # acc = self.train_acc(torch.softmax(logit, axis=1), label.int())
            ap = self.train_ap(
                torch.softmax(logit, axis=1), torch.argmax(label, axis=1)
            )
            loss = self.criterion(logit, torch.argmax(label, axis=1))

        _map = np.nanmean([i.detach().cpu().item() for i in ap])
        # For Callback Drawing later
        # self.last_train_batchs = img, label, img_path
        # self.last_train_logits = logit

        log_dict = {"loss/train": loss, "map/train": _map}
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
        if self.cfg.loss == "bce":
            loss = self.criterion(logit, label)
            # acc = self.val_acc(torch.sigmoid(logit), label.int())
            ap = self.val_ap(torch.sigmoid(logit), torch.argmax(label, axis=1))
        elif self.cfg.loss == "ce":
            # acc = self.val_acc(torch.softmax(logit, axis=1), label.int())
            ap = self.val_ap(torch.softmax(logit, axis=1), torch.argmax(label, axis=1))
            loss = self.criterion(logit, torch.argmax(label, axis=1))
        else:
            raise NotImplementedError("loss should be either bce or ce")

        _map = np.nanmean([i.detach().cpu().item() for i in ap])


        log_dict = {"loss/valid": loss}
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
    #     self.val_acc(pred, label.int())

    def validation_epoch_end(self, validation_step_outputs):

        # FIXME: custom_metrics not working for DDP case
        # if self.use_ddp:
        #     return

        # acc_result = self.val_acc.compute()
        # ap_result = self.val_ap.compute()
        ap_result = [i.detach().cpu().item() for i in self.val_ap.compute()]
        map_result = np.mean(ap_result)

        log_dict = {
            # "vacc": acc_result,
            "map/valid": map_result,
            "ap/valid/0": ap_result[0],
            "ap/valid/1": ap_result[1],
            "ap/valid/2": ap_result[2],
            "ap/valid/3": ap_result[3],
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
        if self.cfg.loss == "bce":
            pred = torch.sigmoid(logit).detach().cpu().numpy().tolist()
            self.test_preds.append((img_path, pred))
        elif self.cfg.loss == "ce":
            pred = torch.softmax(logit, axis=1).detach().cpu().numpy().tolist()
            self.test_preds.append((img_path, pred))

    # def test_epoch_end(self, test_step_outputs):
    #     return self.test_preds
