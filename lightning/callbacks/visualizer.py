from typing import Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn, Tensor
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

# TODO: Drawing for MlFlow


class PredandGTPlotCallback(Callback):  # pragma: no cover
    """
    Prediction and GT drawing callback
    Example::
        trainer = Trainer(callbacks=[PredandGTPlotCallback()])
    .. note:: Whenever called, this model will look for ``self.last_{train/val}_batch`` and ``self.last_{train/val}_logits``
              in the LightningModule.
    .. note:: This callback supports tensorboard only right now.
    Authored by:
        - Brekkanegg
    """

    def __init__(
        self,
        logging_batch_interval: int = 100,
        draw_num: int = 5,
        prob_thr: float = 0.5,
    ):
        """
        Args:
            draw_num: How many images we should plot
        """
        super().__init__()
        self.draw_num = draw_num
        self.logging_batch_interval = logging_batch_interval
        self.prob_thr = prob_thr

    # FIXME: on_epoch_end 로 바꿀까?
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:

        if (batch_idx + 1) % self.logging_batch_interval != 0:  # type: ignore[attr-defined]
            return

        # pick the last batch and logits
        try:
            logits = pl_module.last_train_logits
            batchs = pl_module.last_train_batchs
        except AttributeError as err:
            m = """please track the last_logits in the training_step like so:
                def training_step(...):
                    self.last_logits = your_logits
            """
            raise AttributeError(m) from err

        # See pl_module.dataset to check below return
        logit_seg, _ = logits
        pred = torch.sigmoid(logit_seg) > self.prob_thr
        img, mask, _, _, img_path, disease = batchs

        assert logit_seg.shape == mask.shape  # shape: (N, C, H, W)

        self._plot(trainer, img, pred, mask, "train")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:

        if (batch_idx + 1) % self.logging_batch_interval != 0:  # type: ignore[attr-defined]
            return

        # pick the last batch and logits
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
        logit_seg, _ = logits  # shape: (N, C, H, W)
        pred = torch.sigmoid(logit_seg) > self.prob_thr  # (N, C, H, W)
        img, mask, _, _, img_path, disease = batchs

        self._plot(trainer, img, pred, mask, "val")

    def _plot(
        self, trainer: Trainer, img: np.array, pred: np.array, gt: np.array, mode: str
    ) -> None:

        img = img[: self.draw_num]
        pred = pred[: self.draw_num]
        gt = gt[: self.draw_num]

        fig, axarr = plt.subplots(
            nrows=len(img), ncols=3, figsize=(4 * 3, 4 * len(img))
        )
        for i in range(len(img)):
            x = img[i].squeeze(0).detach().cpu().numpy()
            p = pred[i].squeeze(0).detach().cpu().numpy()
            y = gt[i].squeeze(0).detach().cpu().numpy()

            self.__draw_sample(axarr, i, 0, x, f"img")
            self.__draw_sample(axarr, i, 1, [x, y], f"gt")
            self.__draw_sample(axarr, i, 2, [x, p], f"pred")

        trainer.logger.experiment.add_figure(
            f"{mode}_imgs", fig, global_step=trainer.global_step
        )

    @staticmethod
    def __draw_sample(
        axarr: Axes,
        row_idx: int,
        col_idx: int,
        img: list,
        title: str,
    ) -> None:
        # axarr[row_idx, col_idx]
        if type(img) == list:
            x = img[0]
            m = img[1]
            m = np.ma.masked_where(m == 0, m)
            axarr[row_idx, col_idx].imshow(x, cmap="gray")
            axarr[row_idx, col_idx].imshow(m, cmap="spring", alpha=0.3)
        else:
            x = img
            axarr[row_idx, col_idx].imshow(x, cmap="gray")

        # fig.colorbar(im, ax=axarr[row_idx, col_idx])
        if row_idx == 0:
            axarr[row_idx, col_idx].set_title(title, fontsize=20)
