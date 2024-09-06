from pathlib import Path
import logging
import os
import re
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
from datasets.dataset import load_data, ISICDataset
from models.common import get_model


def prepare_loaders(cfg: InferConfig, df: pd.DataFrame, meta_features=None) -> DataLoader:
    test_dataset = ISICDataset(
        cfg=cfg,
        df=df,
        file_path=os.path.join(cfg.dir.data_dir, "test-image.hdf5"),
        meta_features=meta_features,
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


def load_models(cfg: InferConfig):
    models = []
    # ファイル名のパターン定義
    pattern = "fold(\d+)_pAUC([\d\.]+)_Loss([\d\.]+)_epoch(\d+).bin"

    # ディレクトリ内のファイルを事前にフィルタリング
    files = [f for f in os.listdir(cfg.model_dir) if re.match(pattern, f)]

    for fold in range(cfg.n_folds):
        # フォルダ内のファイルに対して正規表現マッチング
        for file in files:
            match = re.match(pattern, file)
            if match:
                fold_number, pAUC, loss, epoch = match.groups()
                # フォールド番号が現在のフォールドと一致するかチェック
                if int(fold_number) == fold:
                    print(f"Loading model for fold {fold}: pAUC={pAUC}, Loss={loss}, Epoch={epoch}")
                    model_path = os.path.join(model_directory, file)
                    model = get_model(cfg=cfg.model, is_pretrained=False, n_meta_features=n_meta_features)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    models.append(model)
                    break
        else:
            print(f"No model found for fold {fold}.")

    return models


@torch.inference_mode()
def run_ensemble_inference(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
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


@hydra.main(config_path="conf", config_name="infer", version_base="1.3")
def main(cfg: InferConfig):
    """ref: https://www.kaggle.com/code/motono0223/isic-script-inference-effnetv1b0-f313ae/notebook"""
    # Read meta
    _, test_df, meta_features, n_meta_features = load_data(cfg)
    test_df["target"] = 0  # dummy
    LOGGER.info(test_df)

    df_sub = pd.read_csv(os.path.join(cfg.dir.data_dir, "sample_submission.csv"))

    # Create dataloader
    test_loader = prepare_loaders(cfg=cfg, df=test_df, meta_features=meta_features)

    # Def model
    models = load_models()

    # Infer
    preds = run_ensemble_inference(models=models, dataloader=test_loader)

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
