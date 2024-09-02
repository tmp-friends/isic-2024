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


from utils.utils import set_seed
from conf.type import InferConfig
from datasets.dataset import ISICDataset
from models.common import get_model


def prepare_loaders(cfg: InferConfig, df: pd.DataFrame) -> DataLoader:
    test_dataset = ISICDataset(
        cfg=cfg,
        df=df,
        file_path=os.path.join(cfg.dir.data_dir, "test-image.hdf5"),
        is_training=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=4,
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


@hydra.main(config_path="conf", config_name="infer", version_base="1.2")
def main(cfg: InferConfig):
    """ref: https://www.kaggle.com/code/motono0223/isic-script-inference-effnetv1b0-f313ae/notebook"""
    # Read meta
    df = pd.read_csv(os.path.join(cfg.dir.data_dir, "test-metadata.csv"))
    df["target"] = 0  # dummy
    LOGGER.info(df)

    df_sub = pd.read_csv(os.path.join(cfg.dir.data_dir, "sample_submission.csv"))

    # Create dataloader
    test_loader = prepare_loaders(cfg=cfg, df=df)

    # Def model
    model = get_model(cfg=cfg.model, is_pretrained=False)
    model.load_state_dict(torch.load(cfg.best_model_bin))
    model.to(device)

    # Infer
    preds = run_inference(model=model, dataloader=test_loader)

    df_sub["target"] = preds
    df_sub.to_csv("submission.csv", index=False)
    LOGGER.info(df_sub)


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
