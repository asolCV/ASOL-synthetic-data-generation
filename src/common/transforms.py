# common/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import config using absolute path from project root
from src.common.config import cfg


def get_train_transforms():
    """Augmentations for training tiles"""
    return A.Compose(
        [
            A.Rotate(limit=35, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ],
                p=0.4,
            ),
            A.HueSaturationValue(p=0.4),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def get_minimal_transforms():
    """Minimal transforms for validation/prediction tiles"""
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
