import cv2
import albumentations as A


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
