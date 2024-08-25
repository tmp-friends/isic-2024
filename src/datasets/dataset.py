import random
import h5py
from PIL import Image
from io import BytesIO
import numpy as np
from torch.utils.data import Dataset
import cv2


class ISICDataset(Dataset):
    def __init__(self, df, file_path, transforms=None, is_training=True):
        self.df = df
        self.file_path = h5py.File(file_path, mode="r")

        if is_training:
            self.positive_df = df[df["target"] == 1].reset_index()
            self.negative_df = df[df["target"] == 0].reset_index()

            self.positive_isic_ids = self.positive_df["isic_id"].values
            self.negative_isic_ids = self.negative_df["isic_id"].values
            self.positive_targets = self.positive_df["target"].values
            self.negative_targets = self.negative_df["target"].values

        self.transforms = transforms
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
        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "target": target,
            # "meta": meta, # TODO
        }
