# NOTE: Deprecated
print(
    "This Callback is Deprecated: Use metrics/seg_metric.py and metrics/cls_metric instead!"
)
# Move to metrics

from typing import Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

# from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from torch import nn, Tensor
import numpy as np
import skimage
from sklearn import metrics
import cv2


class FROCMetricCallback(Callback):  # pragma: no cover
    """
    FROC Metric Calculation
    Example::
        trainer = Trainer(callbacks=[FROCMetricCallback()])
    .. note:: Whenever called, this model will look for ``self.last_{train/val}_batch`` and ``self.last_{train/val}_logits``
              in the LightningModule.

    Authored by:
        - Brekkanegg
    """

    def __init__(
        self, prob_thr: float = 0.5, iou_thr: float = 0.1, img_size: int = 512
    ):
        """
        Args:
            draw_num: How many images we should plot
        """
        super().__init__()
        self.prob_thr = prob_thr
        self.iou_thr = iou_thr
        self.img_size = img_size

    def on_validation_epoch_start(self, trainer, pl_module):
        # Initialize
        self.froc_result = {
            "gt_cls": [],
            "pred_cls_s": [],
            "pred_cls_c": [],
            "gt_seg": 0,
            "pred_seg": 0,
            "tp_seg": 0,
            "fp_seg": 0,
        }
        for c in pl_module.csv_files:
            self.froc_result[c[:-4]] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        try:
            logits = pl_module.last_val_logits
            batchs = pl_module.last_val_batchs
        except AttributeError as err:
            m = """please track the last_logits in the training_step like so:
                def training_step(...):
                    self.last_logits = your_logits
            """
            raise AttributeError(m) from err

        # See pl_module.dataset to check below return
        logit_seg, logit_cls = logits
        img, mask, mask_cls, _, img_path, disease = batchs

        assert logit_seg.shape == mask.shape  # shape: (N, C, H, W)

        temp = logit_seg.flatten().reshape(logit_seg.shape[0], -1)
        logit_cls_fromseg, _ = torch.max(temp, dim=1)

        # image-basis
        val_gt_cls = mask_cls.detach().cpu().numpy()[:, 0].tolist()
        val_pred_cls_s = (
            torch.sigmoid(logit_cls_fromseg).detach().cpu().numpy().tolist()
        )
        val_pred_cls_c = torch.sigmoid(logit_cls).detach().cpu().numpy()[:, 0].tolist()

        self.froc_result["gt_cls"] += val_gt_cls
        self.froc_result["pred_cls_s"] += val_pred_cls_s
        self.froc_result["pred_cls_c"] += val_pred_cls_c

        # lesion-basis
        pred_seg = torch.sigmoid(logit_seg).detach().cpu().numpy()
        gt_seg = mask.detach().cpu().numpy()

        pred_seg = pred_seg.astype("float32")
        for idx, file_path in enumerate(img_path):
            pred = cv2.resize(pred_seg[idx][0], (self.img_size, self.img_size))
            pred = np.expand_dims(pred, 0)
            gt = cv2.resize(
                gt_seg[idx][0],
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            gt = np.expand_dims(gt, 0)

            gt_nums_, pred_nums_, tp_nums_, fp_nums_ = evaluation(
                pred, gt, iou_th=self.iou_thr
            )

            csv_dir = file_path.split("/")[5]
            # TODO: Need to be change if num_class > 1
            self.froc_result[csv_dir]["gt"] += gt_nums_[0]
            self.froc_result[csv_dir]["pred"] += pred_nums_[0, 0]
            self.froc_result[csv_dir]["tp"] += tp_nums_[0, 0]
            self.froc_result[csv_dir]["fp"] += fp_nums_[0, 0]

    # BUG: documentation 과 달리 outputs 없어야함! callback_hook.py
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:

        # image-basis
        pred_cls_s = np.array(self.froc_result["pred_cls_s"])
        pred_cls_c = np.array(self.froc_result["pred_cls_c"])
        gt_cls = np.array(self.froc_result["gt_cls"])

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

        pl_module.log_dict(
            image_metric_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # lesion-basis
        val_pos_num = gt_cls.sum()
        val_neg_num = len(gt_cls) - gt_cls.sum()

        gt = 0
        tp = 0
        fp = 0
        for c in pl_module.csv_files:
            gt += self.froc_result[c[:-4]]["gt"]
            tp += self.froc_result[c[:-4]]["tp"]
            fp += self.froc_result[c[:-4]]["fp"]
        pre = tp / (tp + fp * (val_pos_num / (val_neg_num + 1e-5)) + 1e-5)
        rec = tp / (gt + 1e-5)
        f1 = 2 * (pre * rec) / (pre + rec + 1e-5)
        myf1 = (pre + rec) / 2.0

        lesion_metric_dict = {
            "pre": pre,
            "rec": rec,
            "f1": f1,
            "myf1": myf1,
        }

        pl_module.log_dict(
            lesion_metric_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


# Helper functions
def calc_iou(bbox_a, bbox_b):
    """
    :param a: bbox list [min_y, min_x, max_y, max_x]
    :param b: bbox list [min_y, min_x, max_y, max_x]
    :return:
    """
    size_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    size_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    min_ab_y = max(bbox_a[0], bbox_b[0])
    min_ab_x = max(bbox_a[1], bbox_b[1])
    max_ab_y = min(bbox_a[2], bbox_b[2])
    max_ab_x = min(bbox_a[3], bbox_b[3])

    inter_ab = max(0, max_ab_y - min_ab_y) * max(0, max_ab_x - min_ab_x)

    return inter_ab / (size_a + size_b - inter_ab)


def evaluation(pred, gt, iou_th=0.15, prob_ths=[0.5]):
    """
    :param pred: Prediction Seg Map, shape = (1, num_classes, height, width)
    :param gt: Ground-truth Seg Map, shape = (1, num_classes, height, width)
    :param iou_th: Threshold for prediction and gt matching
    :return:
        gt_nums: Ground-truth region numbers
        pred_nums: Prediction region numbers
        tp_nums: True Positive region numbers
        fp_nums: False Positive region numbers
    # 필수 가정: batch_size=1 (regionprops 함수가 2차원 행렬에만 적용 가능함)
    # Region을 고려에서 제외하는 경우(2048x2048 이미지 기반, pixel spacing=0.2mm)
    # i) Region bbox 크기 < 400 pixels
    # ii) (현재 사용x) Region bbox 장축<4mm(20pixels), 단축<2mm(10 pixels)
    # issue:  # 3. 영상사이즈는 디텍터 크기에 따라 달라질 수 있습니다. 완벽히 하기 위해선 pixel spacing 정보를 받아야 합니다.
    #         # 따라서 영상 크기에 대해 기준이 변경되는 것은 현단계에서는 적용할 필요가 없어 보입니다.
    """

    if len(pred.shape) > 3:
        pred = pred[0]
        gt = gt[0]

    num_classes = pred.shape[0]
    image_size = gt.shape[2]

    gt_regions = [
        skimage.measure.regionprops(skimage.measure.label(gt[c, :, :]))
        for c in range(num_classes)
    ]
    for c in range(num_classes):
        gt_regions[c] = [
            r for r in gt_regions[c] if r.area > (20 * (image_size / 2048)) ** 2
        ]

    pred_regions = [
        [
            skimage.measure.regionprops(skimage.measure.label(pred[c, :, :] > th))
            for c in range(num_classes)
        ]
        for th in prob_ths
    ]  # shape - len(prob_th), num_classes

    # 초기화
    gt_nums = np.array([len(gt_regions[c]) for c in range(num_classes)])
    pred_nums = np.array(
        [
            [len(pred_regions[thi][c]) for c in range(num_classes)]
            for thi in range(len(prob_ths))
        ]
    )
    tp_nums = np.zeros((len(prob_ths), num_classes))
    fp_nums = pred_nums.copy()  # .copy() 없으면 포인터가 같아짐

    # Gt-Pred Bbox Iou Matrix
    for c in range(num_classes):
        for thi in range(len(prob_ths)):
            if (gt_nums[c] == 0) or (pred_nums[thi][c] == 0):  # np array 이상함;
                continue

            iou_matrix = np.zeros((gt_nums[c], pred_nums[thi][c]))
            for gi, gr in enumerate(gt_regions[c]):
                for pi, pr in enumerate(pred_regions[thi][c]):
                    iou_matrix[gi, pi] = calc_iou(gr.bbox, pr.bbox)

            tp_nums[thi][c] = np.sum(np.any((iou_matrix >= iou_th), axis=1))
            fp_nums[thi][c] -= np.sum(np.any((iou_matrix > iou_th), axis=0))

    return gt_nums, pred_nums, tp_nums, fp_nums