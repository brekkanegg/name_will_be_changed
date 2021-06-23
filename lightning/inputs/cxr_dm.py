import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
from albumentations import CLAHE

from .augmentation import get_augmentation


class CXRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        df,
        size=1024,
        mode="train",
        transform=None,
        apply_windowing=True,
    ):
        self.data_dir = data_dir
        self.df = df

        self.size = size
        self.mode = mode
        self.training = self.mode == "train"
        self.transform = transform
        self.apply_windowing = apply_windowing

        self.clahe = CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.df.loc[index, "id"].split("_image")[0]
        img_path = f"{self.data_dir}/train/{img_id}.png"

        label = np.array(
            list(
                self.df.loc[
                    index,
                    [
                        "Negative for Pneumonia",
                        "Typical Appearance",
                        "Indeterminate Appearance",
                        "Atypical Appearance",
                    ],
                ]
            )
        )

        label = label.astype("float32")

        img = cv2.imread(img_path, -1).astype("float32")

        img = cv2.resize(img, (self.size, self.size))
        if self.apply_windowing:
            img = self.windowing(img, training=self.training)
        img = (img - img.min()) / (img.max() - img.min())

        if self.transform or self.training:

            if np.random.random() < 0.5:
                img *= 255.0
                img = img.astype(np.uint8)
                clahe = self.clahe(image=img)
                img = clahe["image"]
                img = img.astype("float32")
                img /= 255.0

            img = self.random_transform(img, self.transform)
            # blurring and sharpening
            prob = np.random.random()
            if prob < 0.333:
                min_val = np.min(img)
                max_val = np.max(img)
                blurred_f = ndimage.gaussian_filter(img, 2)
                filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
                alpha = 30
                img = blurred_f + alpha * (blurred_f - filter_blurred_f)
                img = np.clip(img, min_val, max_val)

        # img = self.normalization(img)
        img = self.standardization(img)

        img = np.expand_dims(img, 0)

        return img, label, img_path

    def windowing(self, img, training=False):
        center = np.mean(img)
        if training:
            width_param = 4.5 + np.random.random()
        else:
            width_param = 5.0
        width = np.std(img) * width_param
        low = center - width / 2
        high = center + width / 2
        img[img < low] = low
        img[img > high] = high
        return img

    def normalization(self, img, eps=1e-5):
        img = (img - img.mean()) / (img.std() + eps)
        return img

    def standardization(self, img, mean=0.51718974, std=0.21841954):
        img = (img - mean) / std
        return img

    def random_transform(self, img, transform):
        augment = transform(image=img)
        img = augment["image"]
        return img


class CXRDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size  # For auto_scale_batch_size
        self.setup()

    def setup(self, stage=None):

        train_study_level = pd.read_csv(self.cfg.data_dir + "/train_study_level.csv")
        train_image_level = pd.read_csv(self.cfg.data_dir + "/train_image_level.csv")

        more_than_2_ids = []
        for i in range(len(train_image_level)):
            row = train_image_level.iloc[i]
            sid = row["StudyInstanceUID"]
            sid_df = train_image_level[train_image_level["StudyInstanceUID"] == sid]
            if len(sid_df) >= 2:
                more_than_2_ids.append(sid)

        # Cleansing
        train_image_level = train_image_level[
            ~train_image_level["StudyInstanceUID"].isin(more_than_2_ids)
        ]
        train_image_level.reset_index(inplace=True)

        train_study_level["StudyInstanceUID"] = train_study_level["id"].apply(
            lambda x: x.replace("_study", "")
        )
        del train_study_level["id"]
        df = train_image_level.merge(train_study_level, on="StudyInstanceUID")

        # Apply fold
        df = df.sample(frac=1).reset_index(drop=True)

        df["fold"] = df.index % 7

        df_train = df[(df["fold"] != self.cfg.fold_index)].reset_index(drop=True)
        df_valid = df[(df["fold"] == self.cfg.fold_index)].reset_index(drop=True)
        df_test = df[(df["fold"] == self.cfg.fold_index)].reset_index(drop=True)

        print("Training :: ", len(df_train))
        print("Validation :: ", len(df_valid))

        transform = get_augmentation(self.cfg.image_size, self.cfg.image_size, p=0.5)

        self.train_dataset = CXRDataset(
            data_dir=self.cfg.data_dir,
            df=df_train,
            size=self.cfg.image_size,
            mode="train",
            transform=transform,
            apply_windowing=True,
        )

        self.val_dataset = CXRDataset(
            data_dir=self.cfg.data_dir,
            df=df_valid,
            size=self.cfg.image_size,
            mode="val",
            transform=None,
        )

        self.test_dataset = CXRDataset(
            data_dir=self.cfg.data_dir,
            df=df_test,
            size=self.cfg.image_size,
            mode="test",
            transform=None,
        )

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        return train_dataloader

    # FIXME: shuffle=True: for various viz, doesn't matter at performance right?
    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        return val_dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )

        return test_dataloader
