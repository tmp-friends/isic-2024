import random
import h5py
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2

from datasets.transforms import define_transforms


class ISICDataset(Dataset):
    def __init__(self, cfg, df, file_path, is_training=True):
        self.df = df
        self.file_path = h5py.File(file_path, mode="r")

        if is_training:
            self.positive_df = df[df["target"] == 1].reset_index()
            self.negative_df = df[df["target"] == 0].reset_index()

            self.positive_isic_ids = self.positive_df["isic_id"].values
            self.negative_isic_ids = self.negative_df["isic_id"].values
            self.positive_targets = self.positive_df["target"].values
            self.negative_targets = self.negative_df["target"].values

        self.transforms = define_transforms(cfg=cfg, is_training=is_training)
        self.is_training = is_training

    def __len__(self):
        return len(self.positive_df) * 2 if self.is_training else len(self.df)

    def __getitem__(self, ix):
        if self.is_training:
            is_positive = random.random() >= 0.5

            subset_df = self.positive_df if is_positive else self.negative_df
            isic_ids = self.positive_isic_ids if is_positive else self.negative_isic_ids
            targets = self.positive_targets if is_positive else self.negative_targets

            ix %= len(subset_df)

            isic_id = isic_ids[ix]
            target = targets[ix]
        else:
            row = self.df.iloc[ix]
            isic_id = row["isic_id"]
            target = row["target"]

        img = np.array(Image.open(BytesIO(self.file_path[isic_id][()])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "target": target,
            # "meta": meta, # TODO
        }


class PseudoISICDataset(Dataset):
    def __init__(
        self,
        cfg,
        df: pd.DataFrame,
        file_path: str,
        pseudo_df: pd.DataFrame,
        pseudo_file_path: str,
        pseudo_threshold: float,
        is_training=True,
    ):
        self.df = df
        self.file_path = h5py.File(file_path, mode="r")
        self.pseudo_file_path = h5py.File(pseudo_file_path, mode="r")

        if is_training:
            self.positive_df = pd.concat(
                [
                    df[df["target"] == 1],
                    pseudo_df[pseudo_df["target"] >= pseudo_threshold],
                ]
            ).reset_index()
            self.negative_df = pd.concat(
                [
                    df[df["target"] == 0],
                    pseudo_df[pseudo_df["target"] < pseudo_threshold],
                ]
            ).reset_index()

            self.positive_isic_ids = self.positive_df["isic_id"].values
            self.negative_isic_ids = self.negative_df["isic_id"].values
            self.positive_targets = self.positive_df["target"].values
            self.negative_targets = self.negative_df["target"].values

        self.transforms = define_transforms(cfg=cfg, is_training=is_training)
        self.is_training = is_training

    def __len__(self):
        return len(self.positive_df) * 2 if self.is_training else len(self.df)

    def __getitem__(self, ix):
        if self.is_training:
            is_positive = random.random() >= 0.5

            subset_df = self.positive_df if is_positive else self.negative_df
            isic_ids = self.positive_isic_ids if is_positive else self.negative_isic_ids
            targets = self.positive_targets if is_positive else self.negative_targets

            ix %= len(subset_df)

            isic_id = isic_ids[ix]
            target = targets[ix]
        else:
            row = self.df.iloc[ix]
            isic_id = row["isic_id"]
            target = row["target"]

        if isic_id in self.file_path.keys():
            file_path = self.file_path
        else:
            file_path = self.pseudo_file_path

        img = np.array(Image.open(BytesIO(file_path[isic_id][()])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "target": target,
            # "meta": meta, # TODO
        }
