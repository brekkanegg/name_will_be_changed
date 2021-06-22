# For better usage on ddp

import torch
from pytorch_lightning.metrics import Metric
import cv2
import numpy as np
from sklearn import metrics


class ClsMetric(Metric):
    def __init__(self, prob_thr, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        self.prob_thr = prob_thr
        self.use_ddp = dist_sync_on_step
        self.add_state("gt_cls", default=[], dist_reduce_fx="cat")
        self.add_state("pred_cls_s", default=[], dist_reduce_fx="cat")
        self.add_state("pred_cls_c", default=[], dist_reduce_fx="cat")
        self.add_state("csv_files", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        logit_seg, logit_cls = preds
        _, _, mask_cls, _, _, _ = target

        temp = logit_seg.flatten().reshape(logit_seg.shape[0], -1)
        logit_cls_fromseg, _ = torch.max(temp, dim=1)

        # image-basis
        val_gt_cls = mask_cls.detach().cpu().numpy()[:, 0].tolist()
        val_pred_cls_s = (
            torch.sigmoid(logit_cls_fromseg).detach().cpu().numpy().tolist()
        )
        val_pred_cls_c = torch.sigmoid(logit_cls).detach().cpu().numpy()[:, 0].tolist()

        self.gt_cls += val_gt_cls
        self.pred_cls_s += val_pred_cls_s
        self.pred_cls_c += val_pred_cls_c

    def compute(self):
        pred_cls_s = np.array(self.pred_cls_s)
        pred_cls_c = np.array(self.pred_cls_c)
        gt_cls = np.array(self.gt_cls)

        # seg-based-cls
        fpr, tpr, _ = metrics.roc_curve(gt_cls, pred_cls_s)
        val_auc_seg = metrics.auc(fpr, tpr)
        result_cls_s = metrics.classification_report(
            gt_cls, pred_cls_s > self.prob_thr, output_dict=True, zero_division="warn"
        )
        cls_s_report = result_cls_s["weighted avg"]

        # cls-based-cls
        fpr, tpr, _ = metrics.roc_curve(gt_cls, pred_cls_c)
        val_auc_cls = metrics.auc(fpr, tpr)
        result_cls_c = metrics.classification_report(
            gt_cls, pred_cls_c > self.prob_thr, output_dict=True, zero_division="warn"
        )
        cls_c_report = result_cls_c["weighted avg"]

        image_metric_dict = {
            "auc_s": val_auc_seg,
            "auc_c": val_auc_cls,
        }
        for k, v in cls_s_report.items():
            image_metric_dict[f"{k[:3]}_s"] = v
        for k, v in cls_c_report.items():
            image_metric_dict[f"{k[:3]}_c"] = v

        if self.use_ddp:
            image_metric_dict = torch.FloatTensor([val_auc_seg], device=self.device)

        return image_metric_dict

    # For testing each csv
    def update_each(self, preds: torch.Tensor, target: torch.Tensor):
        logit_seg, logit_cls = preds
        _, _, mask_cls, _, img_path, _ = target

        temp = logit_seg.flatten().reshape(logit_seg.shape[0], -1)
        logit_cls_fromseg, _ = torch.max(temp, dim=1)

        # image-basis
        val_gt_cls = mask_cls.detach().cpu().numpy()[:, 0].tolist()
        val_pred_cls_s = (
            torch.sigmoid(logit_cls_fromseg).detach().cpu().numpy().tolist()
        )
        val_pred_cls_c = torch.sigmoid(logit_cls).detach().cpu().numpy()[:, 0].tolist()

        for idx, file_path in enumerate(img_path):
            csv_gt_cls = val_gt_cls[idx]
            csv_pred_cls_s = val_pred_cls_s[idx]
            csv_pred_cls_c = val_pred_cls_c[idx]
            # csv = file_path.split("/")[5]
            csv = file_path.split("png_1024/")[1].split("/")[0]
            if not hasattr(self, f"{csv}_gt_cls"):
                self.csv_files += [csv]
                self.add_state(f"{csv}_gt_cls", default=[], dist_reduce_fx="cat")
                self.add_state(f"{csv}_pred_cls_s", default=[], dist_reduce_fx="cat")
                self.add_state(f"{csv}_pred_cls_c", default=[], dist_reduce_fx="cat")

            setattr(
                self, f"{csv}_gt_cls", getattr(self, f"{csv}_gt_cls") + [csv_gt_cls]
            )
            setattr(
                self,
                f"{csv}_pred_cls_s",
                getattr(self, f"{csv}_pred_cls_s") + [csv_pred_cls_s],
            )
            setattr(
                self,
                f"{csv}_pred_cls_c",
                getattr(self, f"{csv}_pred_cls_c") + [csv_pred_cls_c],
            )

    # For testing each csv
    def compute_each(self):
        metric_dict_each_csv = {}
        for csv in self.csv_files:
            csv_gt_cls = np.array(getattr(self, f"{csv}_gt_cls"))
            csv_pred_cls_s = np.array(getattr(self, f"{csv}_pred_cls_s"))
            # csv_pred_cls_c = np.array(getattr(self, f"{csv}_pred_cls_c"))

            # seg-based-cls
            fpr, tpr, _ = metrics.roc_curve(csv_gt_cls, csv_pred_cls_s)
            val_auc_seg = metrics.auc(fpr, tpr)

            try:
                tn, fn, fp, tp = metrics.confusion_matrix(
                    csv_pred_cls_s > self.prob_thr, csv_gt_cls
                ).ravel()
            except:
                tn = metrics.confusion_matrix(
                    csv_pred_cls_s > self.prob_thr, csv_gt_cls
                ).ravel()[0]
                fn = 0
                fp = 0
                tp = 0

            image_metric_dict = {
                "auc": val_auc_seg,
                "acc": (tp + tn) / (tp + tn + fp + fn + 1e-5),
                "sen": tp / (tp + fn + 1e-5),
                "spe": tn / (tn + fp + 1e-5),
            }
            metric_dict_each_csv[csv] = image_metric_dict

        return metric_dict_each_csv
