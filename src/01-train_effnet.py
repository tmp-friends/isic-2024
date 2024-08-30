from pathlib import Path
import logging
import os
import gc

import random
import math
import copy
import time
import glob
from collections import defaultdict
import joblib
from tqdm import tqdm

import hydra

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import cv2

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

# Sklearn Imports
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

from utils.utils import set_seed, save_processed_img
from utils.metrics import score_p_auc_with_torch
from conf.type import TrainEffnetConfig
from datasets.dataset import ISICDataset
from models.efficientnet import EfficientNet


def load_data(cfg):
    train_df = pd.read_csv(cfg.dir.train_meta_csv)

    # # 2020 data (external data)
    # train_df_ext1 = pd.read_csv(cfg.dir.train_meta_csv_2020)
    # train_df_ext1 = train_df_ext1[train_df_ext1["tfrecord"] != -1].reset_index(drop=True)
    # train_df_ext1["filepath"] = train_df_ext1["image_name"].apply(
    #     lambda v: os.path.join(cfg.dir.train_image_dir_2020, f"{v}.jpg")
    # )

    # # 2018, 2019 data (external data)
    # train_df_ext2 = pd.read_csv(cfg.dir.train_meta_csv_2019)
    # train_df_ext2 = train_df_ext2[train_df_ext2["tfrecord"] != -1].reset_index(drop=True)
    # train_df_ext2["filepath"] = train_df_ext2["image_name"].apply(
    #     lambda v: os.path.join(cfg.dir.train_image_dir_2019, f"{v}.jpg")
    # )

    return train_df


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


def get_sampler(df: pd.DataFrame):
    """ref: https://www.kaggle.com/code/syzygyfy/addressing-the-class-imbalance-in-pytorch"""
    u, c = np.unique(df["target"], return_counts=True)
    class_weights = [1.0 / v for v in c]
    sample_weights = [class_weights[v] for v in df["target"]]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return sampler


def prepare_loaders(cfg: TrainEffnetConfig, df: pd.DataFrame, sampler=None) -> tuple[DataLoader, DataLoader]:
    train_df = df[df.kfold != cfg.fold].reset_index(drop=True)
    valid_df = df[df.kfold == cfg.fold].reset_index(drop=True)

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
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
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

    train_dataset = ISICDataset(
        train_df,
        file_path=cfg.dir.train_image_hdf,
        transforms=data_transforms["train"],
        is_training=True,
    )
    valid_dataset = ISICDataset(
        valid_df,
        file_path=cfg.dir.train_image_hdf,
        transforms=data_transforms["valid"],
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=4,
        sampler=sampler,
        # shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=4,
        # sampler=sampler,
        shuffle=False,
        pin_memory=True,
    )

    save_processed_img(train_dataset)

    return train_loader, valid_loader


def fetch_scheduler(cfg: TrainEffnetConfig, optimizer: optim) -> lr_scheduler:
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr,
        )
    elif cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.T_0,
            eta_min=cfg.min_lr,
        )
    elif cfg.scheduler == None:
        return None

    return scheduler


def criterion(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.BCELoss()(outputs, targets)


def run_training(
    cfg: TrainEffnetConfig,
    model: nn.Module,
    optimizer: optim,
    scheduler: lr_scheduler,
    train_loader: DataLoader,
    valid_loader: DataLoader,
) -> nn.Module | dict:
    if torch.cuda.is_available():
        LOGGER.info("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_pauc = -np.inf
    history = defaultdict(list)

    for epoch in range(1, cfg.n_epochs + 1):
        train_epoch_loss, train_epoch_pauc = train_one_epoch(
            cfg,
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            epoch=epoch,
        )
        valid_epoch_loss, valid_epoch_pauc = valid_one_epoch(
            model,
            optimizer,
            valid_loader,
            epoch=epoch,
        )
        LOGGER.info(
            f"Epoch {epoch}: Train pAUC: {train_epoch_pauc:.6f} - Valid pAUC: {valid_epoch_pauc:.6f} | Train Loss: {train_epoch_loss:.6f} - Valid Loss: {valid_epoch_loss:.6f}\n"
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(valid_epoch_loss)
        history["Train pAUC"].append(train_epoch_pauc)
        history["Valid pAUC"].append(valid_epoch_pauc)
        history["lr"].append(scheduler.get_last_lr()[0])

        # deep copy the model
        if best_epoch_pauc <= valid_epoch_pauc:
            LOGGER.info(f"Val pAUC Improved ({best_epoch_pauc} ---> {valid_epoch_pauc})")

            best_epoch_pauc = valid_epoch_pauc
            best_model_wts = copy.deepcopy(model.state_dict())

            PATH = "pAUC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(valid_epoch_pauc, valid_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)

            # Save a model file from the current directory
            LOGGER.info("Model saved")

    end = time.time()
    time_elapsed = end - start
    LOGGER.info(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60
        )
    )
    LOGGER.info("Best pAUC: {:.4f}".format(best_epoch_pauc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def train_one_epoch(
    cfg: TrainEffnetConfig,
    model: nn.Module,
    optimizer: optim,
    scheduler: lr_scheduler,
    dataloader: DataLoader,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()

    dataset_size = 0
    running_loss = 0.0
    running_pauc = 0.0

    y_true = []
    y_preds = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        targets = data["target"].to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images).squeeze()

        loss = criterion(outputs, targets)
        loss /= cfg.n_accumulates

        loss.backward()

        if (step + 1) % cfg.n_accumulates == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        y_true.extend(targets.detach().cpu().numpy())
        y_preds.extend(outputs.detach().cpu().numpy())

        bar.set_postfix(Epoch=epoch, LR=optimizer.param_groups[0]["lr"])

    epoch_loss = running_loss / dataset_size
    epoch_pauc = score_p_auc_with_torch(y_true=y_true, y_preds=y_preds)

    return epoch_loss, epoch_pauc


@torch.inference_mode()
def valid_one_epoch(
    model: nn.Module,
    optimizer: optim,
    dataloader: DataLoader,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    y_true = []
    y_preds = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        targets = data["target"].to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        y_true.extend(targets.detach().cpu().numpy())
        y_preds.extend(outputs.detach().cpu().numpy())

        bar.set_postfix(Epoch=epoch, LR=optimizer.param_groups[0]["lr"])

    epoch_loss = running_loss / dataset_size
    epoch_pauc = score_p_auc_with_torch(y_true=y_true, y_preds=y_preds)

    return epoch_loss, epoch_pauc


@hydra.main(config_path="conf", config_name="train_effnet", version_base="1.1")
def main(cfg: TrainEffnetConfig):
    # Read meta
    # df = pd.read_csv(cfg.dir.train_meta_csv)

    # df_positive = df[df["target"] == 1].reset_index(drop=True)
    # df_negative = df[df["target"] == 0].reset_index(drop=True)

    # # positive:negative=1:20になるようDown sampling
    # # positiveが少ない不均衡データなので学習がうまくいくようにする意図
    # df = pd.concat([df_positive, df_negative.iloc[: df_positive.shape[0] * 20, :]]).reset_index(drop=True)
    df = load_data(cfg)
    LOGGER.info(df)

    cfg.T_max = df.shape[0] * (cfg.n_folds - 1) * cfg.n_epochs // cfg.train_batch_size // cfg.n_folds

    # Create fold
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds)
    for fold, (_, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
        df.loc[val_, "kfold"] = int(fold)

    # Create dataloader
    sampler = get_sampler(df=df)
    train_loader, valid_loader = prepare_loaders(cfg=cfg, df=df, sampler=sampler)

    # Def model
    # https://www.kaggle.com/models/timm/tf-efficientnet/pyTorch/tf-efficientnet-b0/1
    model = EfficientNet(model_name=cfg.model_name, checkpoint_path=cfg.checkpoint_path)
    model.to(device)
    LOGGER.info(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = fetch_scheduler(cfg=cfg, optimizer=optimizer)

    # Train
    model, history = run_training(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    # Monitor
    history = pd.DataFrame.from_dict(history)
    history.to_csv("history.csv", index=False)

    plt.plot(range(history.shape[0]), history["Train Loss"].values, label="Train Loss")
    plt.plot(range(history.shape[0]), history["Valid Loss"].values, label="Valid Loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig("plt-loss.png")
    plt.clf()

    plt.plot(range(history.shape[0]), history["Train pAUC"].values, label="Train pAUC")
    plt.plot(range(history.shape[0]), history["Valid pAUC"].values, label="Valid pAUC")
    plt.xlabel("epochs")
    plt.ylabel("pAUC")
    plt.grid()
    plt.legend()
    plt.savefig("plt-p-auc.png")
    plt.clf()

    plt.plot(range(history.shape[0]), history["lr"].values, label="lr")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.grid()
    plt.legend()
    plt.savefig("plt-lr.png")
    plt.clf()


if __name__ == "__main__":
    # Logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Set GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed()

    main()
