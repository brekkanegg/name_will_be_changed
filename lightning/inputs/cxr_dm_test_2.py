import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
from albumentations import CLAHE
import glob
from .augmentation import get_augmentation_v2


class CXRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        size=1024,
        mode="test",
        transform=None,
        # apply_windowing=True,
    ):
        self.data_dir = data_dir

        self.source = glob.glob(self.data_dir + "/*.png")

        self.size = size
        self.mode = mode
        self.training = self.mode == "train"
        self.transform = transform
        # self.apply_windowing = apply_windowing

        # self.clahe = CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=1.0)

    def __len__(self):

        return len(self.source)

    def __getitem__(self, index):
        # img_id = self.df.loc[index, "id"].split("_image")[0]
        img_path = self.source[index]

        img = cv2.imread(img_path, -1).astype("float32")
        img = self.transform(image=img)["image"]

        return img, img_path


class CXRDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size  # For auto_scale_batch_size
        self.setup()

    def setup(self, stage=None):
        _, val_aug = get_augmentation_v2(self.cfg.image_size)

        self.test_dataset = CXRDataset(
            data_dir=self.cfg.data_dir,
            size=self.cfg.image_size,
            mode="test",
            transform=val_aug,
        )

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )

        return test_dataloader
