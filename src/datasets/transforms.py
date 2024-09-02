import random
import numpy as np


# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2


def define_transforms(cfg, is_training: bool = True):
    data_transforms = {
        # ref: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412
        "train": A.Compose(
            [
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.75),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=(3, 5), p=0.25),
                        A.MedianBlur(blur_limit=3, p=0.25),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.25),
                        A.GaussNoise(var_limit=(5.0, 30.0), p=0.25),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.0),
                        A.ElasticTransform(alpha=3),
                    ],
                    p=0.7,
                ),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(cfg.img_size, cfg.img_size),
                CustomCutout(
                    num_holes=1, max_h_size=int(cfg.img_size * 0.375), max_w_size=int(cfg.img_size * 0.375), p=0.7
                ),
                # Microscope(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ],
        ),
        "valid": A.Compose(
            [
                A.Resize(cfg.img_size, cfg.img_size),
                A.Normalize(),
                ToTensorV2(),
            ],
        ),
    }

    return data_transforms["train"] if is_training else data_transforms["valid"]


class Microscope(ImageOnlyTransform):
    """
    Simulate the effect of viewing through a microscope by cutting out edges around the center circle of the image.

    Args:
        p (float): probability of applying the augmentation.
    """

    def __init__(self, p=0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        """
        Apply the transformation.

        Args:
            img (numpy.ndarray): Image to apply transformation to.

        Returns:
            numpy.ndarray: Image with transformation applied.
        """
        circle = cv2.circle(
            (np.ones(img.shape) * 255).astype(np.uint8),
            (img.shape[1] // 2, img.shape[0] // 2),  # center point of circle
            random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),  # radius
            (0, 0, 0),  # color
            -1,
        )

        mask = circle - 255
        img = np.multiply(img, mask.astype(np.uint8))
        return img

    def get_transform_init_args_names(self):
        return ("p",)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class CustomCutout(ImageOnlyTransform):
    def __init__(self, num_holes, max_h_size, max_w_size, always_apply=False, p=0.5):
        super(CustomCutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def apply(self, image, **params):
        h, w = image.shape[:2]
        mask = np.ones((h, w), np.float32)
        for n in range(self.num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.max_h_size // 2, 0, h)
            y2 = np.clip(y + self.max_h_size // 2, 0, h)
            x1 = np.clip(x - self.max_w_size // 2, 0, w)
            x2 = np.clip(x + self.max_w_size // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        image = image * mask[:, :, np.newaxis]
        return image
