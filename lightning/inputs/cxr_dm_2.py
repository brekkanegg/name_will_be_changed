import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import cv2

# from scipy import ndimage
# from albumentations import CLAHE

from .augmentation import get_augmentation_v2


class CXRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        df,
        size=1024,
        mode="train",
        transform=None,
        smooth=0,
    ):
        self.data_dir = data_dir
        self.df = df
        self.size = size
        self.mode = mode
        self.training = self.mode == "train"
        self.transform = transform
        self.smooth = smooth

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if np.isnan(self.df.loc[index, 'index']): # This is from pseudo_label
            img_id = self.df.loc[index, "id"].split("_study")[0]
            img_path = f"{self.data_dir}/test/{img_id}.png"
        else:
            img_id = self.df.loc[index, "id"].split("_image")[0]
            img_path = f"{self.data_dir}/train/{img_id}_image.png"

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
        
        if self.smooth:
            label = np.clip(label, self.smooth, 1 - self.smooth)

        try:
            img = cv2.imread(img_path, -1).astype("float32")
        except:
            print(img_path)
        img = np.concatenate((img[:, :, np.newaxis],) * 3, axis=-1)
        img = self.transform(image=img)["image"]

        return img, label, img_path


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

        df["fold"] = df.index % 5

        df_train = df[(df["fold"] != self.cfg.fold_index)].reset_index(drop=True)
        df_valid = df[(df["fold"] == self.cfg.fold_index)].reset_index(drop=True)
        # df_test = df[(df["fold"] == self.cfg.fold_index)].reset_index(drop=True)

        # TODO: add pseudo-label
        if self.cfg.use_pseudo_label:
            df_test_pseudo = pd.read_csv('./pseudo_test_study_level.csv')
            # remove if std is too big
            df_test_pseudo = df_test_pseudo[df_test_pseudo['std'] < 0.2].iloc[:, :5]
            # Temporary fix:
            # if 'id' in df_test_pseudo.columns:
            #     df_test_pseudo.rename(columns={'id': 'StudyInstanceUID'}, inplace=True)
            #     df_test_pseudo['StudyInstanceUID'] = df_test_pseudo.apply(lambda row: row['StudyInstanceUID'].split('_study')[0], axis=1)

            print('Pseudo Test as Train :: ', len(df_test_pseudo))
            df_train = pd.concat([df_train, df_test_pseudo], axis=0).reset_index(drop=True)
            df_train = df_train.sample(frac=1).reset_index(drop=True)


        print("Training :: ", len(df_train))
        print("Validation :: ", len(df_valid))

        train_aug, val_aug = get_augmentation_v2(self.cfg.image_size)

        self.train_dataset = CXRDataset(
            data_dir=self.cfg.data_dir,
            df=df_train,
            size=self.cfg.image_size,
            mode="train",
            transform=train_aug,
            smooth=self.cfg.label_smoothing,
        )

        self.val_dataset = CXRDataset(
            data_dir=self.cfg.data_dir,
            df=df_valid,
            size=self.cfg.image_size,
            mode="val",
            transform=val_aug,
        )

        # self.test_dataset = CXRDataset(
        #     data_dir=self.cfg.data_dir,
        #     df=df_test,
        #     size=self.cfg.image_size,
        #     mode="test",
        #     transform=va,
        # )

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
