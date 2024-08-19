from pathlib import Path
import logging
import os
import gc

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

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda import amp
import torchvision
from torcheval.metrics.functional import binary_auroc

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import set_seed
from conf.type import TrainTimmModelConfig
from datasets.dataset import ISICDataset_for_Train, ISICDataset
from models.net import Net


def prepare_loaders(cfg: TrainTimmModelConfig, df: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
    df_train = df[df.kfold != cfg.fold].reset_index(drop=True)
    df_valid = df[df.kfold == cfg.fold].reset_index(drop=True)

    data_transforms = {
        "train": A.Compose(
            [
                A.Resize(cfg.img_size, cfg.img_size),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Downscale(p=0.25),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=60,
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(-0.1, 0.1),
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "valid": A.Compose(
            [
                A.Resize(cfg.img_size, cfg.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    }

    train_dataset = ISICDataset(df_train, transforms=data_transforms["train"])
    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader


def fetch_scheduler(cfg: TrainTimmModelConfig, optimizer: optim) -> lr_scheduler:
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
    cfg: TrainTimmModelConfig,
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
    best_epoch_auroc = -np.inf
    history = defaultdict(list)

    for epoch in range(1, cfg.n_epochs + 1):
        # gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(
            cfg,
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            epoch=epoch,
        )

        val_epoch_loss, val_epoch_auroc = valid_one_epoch(
            model,
            optimizer,
            valid_loader,
            epoch=epoch,
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)
        history["Train AUROC"].append(train_epoch_auroc)
        history["Valid AUROC"].append(val_epoch_auroc)
        history["lr"].append(scheduler.get_lr()[0])

        # deep copy the model
        if best_epoch_auroc <= val_epoch_auroc:
            LOGGER.info(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")

            best_epoch_auroc = val_epoch_auroc
            best_model_wts = copy.deepcopy(model.state_dict())

            PATH = "AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
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
    LOGGER.info("Best AUROC: {:.4f}".format(best_epoch_auroc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def train_one_epoch(
    cfg: TrainTimmModelConfig,
    model: nn.Module,
    optimizer: optim,
    scheduler: lr_scheduler,
    dataloader: DataLoader,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()

    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0

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

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()

        running_loss += loss.item() * batch_size
        running_auroc += auroc * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc, LR=optimizer.param_groups[0]["lr"])

    return epoch_loss, epoch_auroc


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
    running_auroc = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        targets = data["target"].to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        running_loss += loss.item() * batch_size
        running_auroc += auroc * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc, LR=optimizer.param_groups[0]["lr"])

    return epoch_loss, epoch_auroc


@hydra.main(config_path="conf", config_name="train_swin_large", version_base="1.1")
def main(cfg: TrainTimmModelConfig):
    """
    ref:
        train: https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-eva02
        infer: https://www.kaggle.com/code/motono0223/isic-inference-eva02-for-training-data
    """
    # Read meta
    df = pd.read_csv(cfg.dir.train_meta_csv)
    LOGGER.info(df)

    # Read image
    train_images = sorted(glob.glob(f"{cfg.dir.train_image_dir}/*.jpg"))

    def _get_train_file_path(image_id):
        return f"{cfg.dir.train_image_dir}/{image_id}.jpg"

    df["file_path"] = df["isic_id"].apply(_get_train_file_path)
    df = df[df["file_path"].isin(train_images)].reset_index(drop=True)
    LOGGER.info(df)

    cfg.T_max = df.shape[0] * (cfg.n_folds - 1) * cfg.n_epochs // cfg.train_batch_size // cfg.n_folds

    # Create fold
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds)
    for fold, (_, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
        df.loc[val_, "kfold"] = int(fold)

    # Create dataloader
    train_loader, valid_loader = prepare_loaders(cfg=cfg, df=df)

    # Def model
    model = Net(model_name=cfg.model_name)
    model.to(device)

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

    plt.plot(range(history.shape[0]), history["Train AUROC"].values, label="Train AUROC")
    plt.plot(range(history.shape[0]), history["Valid AUROC"].values, label="Valid AUROC")
    plt.xlabel("epochs")
    plt.ylabel("AUROC")
    plt.grid()
    plt.legend()
    plt.savefig("plt-auroc.png")
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
