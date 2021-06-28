from albumentations.augmentations.transforms import HorizontalFlip
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation(h, w, p=0.5):
    transform = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=cv2.BORDER_REPLICATE,
                p=p,
            ),
            A.RandomSizedCrop(
                min_max_height=(h - int(h * 0.15), h), height=h, width=w, p=p
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
            A.GaussNoise(var_limit=0.001, mean=0.0, p=p),
        ],
        p=1,
    )
    return transform


def get_augmentation_v2(cfg):
    h, w = cfg.image_size, cfg.image_size
    train_aug = A.Compose(
        [
            A.Resize(cfg.image_size, cfg.image_size),
            A.RandomSizedCrop(
                min_max_height=(h - int(h * 0.15), h), height=h, width=w, p=0.5
            ),
            A.HorizontalFlip(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            ),
            A.CoarseDropout(p=0.5),  # cutout
            # A.OneOf([A.Cutout(), A.ElasticTransform()], p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )

    val_aug = A.Compose(
        [
            A.Resize(cfg.image_size, cfg.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return train_aug, val_aug
