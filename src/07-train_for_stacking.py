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

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

# Sklearn Imports
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from utils.utils import set_seed, save_processed_img
from utils.metrics import score_p_auc
from conf.type import TrainConfig
from datasets.dataset import load_data, balance_train_set, ISICDataset
from models.common import get_model


def get_sampler(df: pd.DataFrame):
    """ref: https://www.kaggle.com/code/syzygyfy/addressing-the-class-imbalance-in-pytorch"""
    u, c = np.unique(df["target"], return_counts=True)
    class_weights = [1.0 / v for v in c]
    sample_weights = [class_weights[v] for v in df["target"]]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return sampler


def prepare_loaders(
    cfg: TrainConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    meta_features: list = None,
    sampler=None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ISICDataset(
        cfg,
        train_df,
        file_path=os.path.join(cfg.dir.data_dir, "train-image.hdf5"),
        meta_features=meta_features,
        is_training=True,
    )
    valid_dataset = ISICDataset(
        cfg,
        valid_df,
        file_path=os.path.join(cfg.dir.data_dir, "train-image.hdf5"),
        meta_features=meta_features,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=4,
        sampler=sampler,
        shuffle=True,
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


def fetch_scheduler(cfg: TrainConfig, optimizer: optim) -> lr_scheduler:
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


def train_one_epoch(
    cfg: TrainConfig,
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
        x = data["image"].to(device, dtype=torch.float)
        x_meta = data["meta"].to(device, dtype=torch.float)
        y = data["target"].to(device, dtype=torch.float)

        batch_size = x.size(0)

        outputs = model(x, x_meta).squeeze()

        loss = criterion(outputs, y)
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

        y_true.extend(y.detach().cpu().numpy())
        y_preds.extend(outputs.detach().cpu().numpy())

        bar.set_postfix(Epoch=epoch, LR=optimizer.param_groups[0]["lr"])

    epoch_loss = running_loss / dataset_size

    solution_df = pd.DataFrame({"target": y_true})
    submission_df = pd.DataFrame({"prediction": y_preds})
    epoch_pauc = score_p_auc(solution_df, submission_df)

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
        x = data["image"].to(device, dtype=torch.float)
        x_meta = data["meta"].to(device, dtype=torch.float)
        y = data["target"].to(device, dtype=torch.float)

        batch_size = x.size(0)

        outputs = model(x, x_meta).squeeze()
        loss = criterion(outputs, y)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        y_true.extend(y.detach().cpu().numpy())
        y_preds.extend(outputs.detach().cpu().numpy())

        bar.set_postfix(Epoch=epoch, LR=optimizer.param_groups[0]["lr"])

    epoch_loss = running_loss / dataset_size

    solution_df = pd.DataFrame({"target": y_true})
    submission_df = pd.DataFrame({"prediction": y_preds})
    epoch_pauc = score_p_auc(solution_df, submission_df)

    return epoch_loss, epoch_pauc


@torch.inference_mode()
def run_inference(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    model.eval()

    preds = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        x = data["image"].to(device, dtype=torch.float)
        x_meta = data["meta"].to(device, dtype=torch.float)

        batch_size = x.size(0)

        outputs = model(x, x_meta)
        preds.append(outputs.detach().cpu().numpy())

    return np.concatenate(preds).flatten()


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg: TrainConfig):
    # Read meta
    df, test_df, meta_features, n_meta_features = load_data(cfg)
    LOGGER.info(df)
    LOGGER.info(f"{meta_features=}")

    cfg.T_max = df.shape[0] * (cfg.n_folds - 1) * cfg.n_epochs // cfg.train_batch_size // cfg.n_folds

    # oof preds
    oof_preds = np.zeros(len(df))
    model_filenames = [""] * len(df)
    # Cross validation train
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds)
    for i, (train_ix, valid_ix) in enumerate(sgkf.split(df, df["target"], df["patient_id"])):
        df.loc[valid_ix, "fold"] = int(i)

        LOGGER.info(f"Fold {i}")

        # Create dataloader
        train_df = balance_train_set(df.iloc[train_ix])
        valid_df = df.iloc[valid_ix]
        # sampler = get_sampler(df=df)
        train_loader, valid_loader = prepare_loaders(
            cfg=cfg,
            train_df=train_df,
            valid_df=valid_df,
            meta_features=meta_features,
            # sampler=sampler,
        )

        # Initialize model
        model = get_model(cfg=cfg.model, is_pretrained=True, n_meta_features=n_meta_features)
        model.to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = fetch_scheduler(cfg=cfg, optimizer=optimizer)

        start = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch_loss = -np.inf
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
                dataloader=valid_loader,
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

                best_epoch_loss = valid_epoch_loss
                best_epoch_pauc = valid_epoch_pauc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Save a model file from the current directory
        model_filename = "fold{:.0f}_pAUC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(
            i, best_epoch_pauc, best_epoch_loss, epoch
        )
        torch.save(best_model_wts, model_filename)
        LOGGER.info("Model saved")

        end = time.time()
        time_elapsed = end - start

        LOGGER.info(
            "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60
            )
        )
        LOGGER.info("Best pAUC: {:.4f}".format(best_epoch_pauc))

        # oof preds
        model.load_state_dict(best_model_wts)
        oof_preds[valid_df.index] = run_inference(model, valid_loader)
        for ix in valid_df.index:
            model_filenames[ix] = model_filename

        # Monitor
        history = pd.DataFrame.from_dict(history)
        history.to_csv(f"history_fold{i}.csv", index=False)

        plt.plot(range(history.shape[0]), history["Train Loss"].values, label="Train Loss")
        plt.plot(range(history.shape[0]), history["Valid Loss"].values, label="Valid Loss")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.savefig(f"plt-loss-fold{i}.png")
        plt.clf()

        plt.plot(range(history.shape[0]), history["Train pAUC"].values, label="Train pAUC")
        plt.plot(range(history.shape[0]), history["Valid pAUC"].values, label="Valid pAUC")
        plt.xlabel("epochs")
        plt.ylabel("pAUC")
        plt.grid()
        plt.legend()
        plt.savefig(f"plt-pauc-fold{i}.png")
        plt.clf()

        plt.plot(range(history.shape[0]), history["lr"].values, label="lr")
        plt.xlabel("epochs")
        plt.ylabel("lr")
        plt.grid()
        plt.legend()
        plt.savefig("plt-lr.png")
        plt.clf()

    oof_preds_df = pd.DataFrame(
        {
            "isic_id": df["isic_id"],
            "target": df["target"],
            "fold": df["fold"],
            "oof_preds": oof_preds,
            "model_filename": model_filenames,
        }
    )
    oof_preds_df.to_csv("oof_preds.csv", index=False)


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
