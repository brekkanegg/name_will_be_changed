# For better usage on ddp

import torch
from pytorch_lightning.metrics import Metric
import cv2
import numpy as np
import skimage
import torch.tensor as Tensor


class SegMetric(Metric):
    def __init__(self, iou_thr, prob_thr, img_size, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        self.iou_thr = iou_thr
        self.prob_thr = prob_thr
        self.img_size = img_size
        self.use_ddp = dist_sync_on_step
        self.add_state("csv_files", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        logit_seg, _ = preds
        _, mask, mask_cls, _, img_path, _ = target

        assert logit_seg.shape == mask.shape

        pred_seg = torch.sigmoid(logit_seg).detach().cpu().numpy()
        gt_seg = mask.detach().cpu().numpy()
        gt_cls = mask_cls.detach().cpu().numpy()[:, 0].tolist()

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

            gt_c = gt_cls[idx]
            is_p = int(gt_c == 1.0)
            is_n = 1 - is_p

            gt_nums_, pred_nums_, tp_nums_, fp_nums_ = evaluation(
                pred, gt, iou_th=self.iou_thr, prob_ths=[self.prob_thr]
            )

            # csv = file_path.split("/")[5]
            csv = file_path.split("png_1024/")[1].split("/")[0]
            if not hasattr(self, f"{csv}_gt"):
                self.csv_files += [csv]
                self.add_state(f"{csv}_gt", default=Tensor(0), dist_reduce_fx="sum")
                self.add_state(f"{csv}_pred", default=Tensor(0), dist_reduce_fx="sum")
                self.add_state(f"{csv}_tp", default=Tensor(0), dist_reduce_fx="sum")
                self.add_state(f"{csv}_fp", default=Tensor(0), dist_reduce_fx="sum")
                self.add_state(f"{csv}_pos", default=Tensor(0), dist_reduce_fx="sum")
                self.add_state(
                    f"{csv}_neg", default=torch.tensor(0), dist_reduce_fx="sum"
                )

            # TODO: Need to be change if num_class > 1
            # FIXME: 몬 생긴 포맷..
            setattr(self, f"{csv}_gt", getattr(self, f"{csv}_gt") + gt_nums_[0])
            setattr(
                self, f"{csv}_pred", getattr(self, f"{csv}_pred") + pred_nums_[0, 0]
            )
            setattr(self, f"{csv}_tp", getattr(self, f"{csv}_tp") + tp_nums_[0, 0])
            setattr(self, f"{csv}_fp", getattr(self, f"{csv}_fp") + fp_nums_[0, 0])
            setattr(self, f"{csv}_pos", getattr(self, f"{csv}_pos") + is_p)
            setattr(self, f"{csv}_neg", getattr(self, f"{csv}_neg") + is_n)

    def update_each(self, preds: torch.Tensor, target: torch.Tensor):
        self.update(preds, target)

    def compute(self):
        gt = 0
        tp = 0
        fp = 0
        pos = 0
        neg = 0
        for csv in self.csv_files:
            gt += getattr(self, f"{csv}_gt").item()
            tp += getattr(self, f"{csv}_tp").item()
            fp += getattr(self, f"{csv}_fp").item()
            pos += getattr(self, f"{csv}_pos").item()
            neg += getattr(self, f"{csv}_neg").item()

        pre = tp / (tp + fp * (pos / (neg + 1e-5)) + 1e-5)
        rec = tp / (gt + 1e-5)
        f1 = 2 * (pre * rec) / (pre + rec + 1e-5)
        myf1 = (pre + rec) / 2.0

        lesion_metric_dict = {
            "pre": pre,
            "rec": rec,
            "f1": f1,
            "myf1": myf1,
        }

        # FIXME: DDP Error: https://github.com/PyTorchLightning/pytorch-lightning/discussions/2529
        # Tensors must be CUDA and dense
        # if self.use_ddp:
        #     lesion_metric_dict = torch.FloatTensor([myf1], device=self.device)

        return lesion_metric_dict

    def compute_each(self):
        metric_dict_each_csv = {}
        for csv in self.csv_files:
            gt = getattr(self, f"{csv}_gt").item()
            tp = getattr(self, f"{csv}_tp").item()
            fp = getattr(self, f"{csv}_fp").item()
            pos = getattr(self, f"{csv}_pos").item()
            neg = getattr(self, f"{csv}_neg").item()

            pre = tp / (tp + fp * (pos / (neg + 1e-5)) + 1e-5)
            rec = tp / (gt + 1e-5)
            f1 = 2 * (pre * rec) / (pre + rec + 1e-5)
            fppi = fp / (pos + neg + 1e-5)
            # myf1 = (pre + rec) / 2.0

            lesion_metric_dict = {
                "gt": gt,
                "pos": pos,
                "neg": neg,
                "pre": pre,
                "rec": rec,
                "f1": f1,
                "fppi": fppi
                # "myf1": myf1,
            }

            metric_dict_each_csv[csv] = lesion_metric_dict

        return metric_dict_each_csv


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