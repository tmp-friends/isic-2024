from pathlib import Path
import logging
import os
import sys
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
from torch.utils.data import DataLoader


# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import set_seed
from conf.type import InferTimmModelConfig
from datasets.dataset import ISICDataset_for_Test
from models.eva import EVA


def prepare_loaders(cfg: InferTimmModelConfig, df: pd.DataFrame) -> DataLoader:
    data_transforms = {
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
    test_dataset = ISICDataset_for_Test(
        df=df,
        file_hdf=cfg.dir.test_image_hdf,
        transforms=data_transforms["valid"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return test_loader


@torch.inference_mode()
def run_inference(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    model.eval()

    preds = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images)
        preds.append(outputs.detach().cpu().numpy())

    return np.concatenate(preds).flatten()


@hydra.main(config_path="conf", config_name="infer_eva", version_base="1.1")
def main(cfg: InferTimmModelConfig):
    """
    ref:
        train: https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-eva02
        infer: https://www.kaggle.com/code/motono0223/isic-inference-eva02-for-training-data
    """
    # Read meta
    df = pd.read_csv(cfg.dir.test_meta_csv)
    df["target"] = 0  # dummy
    LOGGER.info(df)

    df_sub = pd.read_csv(cfg.dir.sample_csv)
    LOGGER.info(df_sub)

    # Create dataloader
    test_loader = prepare_loaders(cfg=cfg, df=df)

    # Def model
    # https://www.kaggle.com/models/timm/tf-efficientnet/pyTorch/tf-efficientnet-b0/1
    model = EVA(model_name=cfg.model_name, pretrained=False)
    model.load_state_dict(torch.load(cfg.best_model_bin))
    model.to(device)

    # Infer
    preds = run_inference(model=model, dataloader=test_loader)

    df_sub["target"] = preds
    df_sub.to_csv("submission.csv", index=False)


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
