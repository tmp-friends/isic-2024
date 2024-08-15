import random
import h5py
from PIL import Image
from io import BytesIO
import numpy as np
from torch.utils.data import Dataset
import cv2


class ISICDataset_for_Train(Dataset):
    def __init__(self, df, transforms=None):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()

        self.file_names_positive = self.df_positive["file_path"].values
        self.file_names_negative = self.df_negative["file_path"].values
        self.targets_positive = self.df_positive["target"].values
        self.targets_negative = self.df_negative["target"].values

        self.transforms = transforms

    def __len__(self):
        return len(self.df_positive) * 2

    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative

        index %= df.shape[0]

        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = targets[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "target": target,
        }


class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.targets = df["target"].values

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.targets[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "target": target,
        }


class ISICDataset_for_Test(Dataset):
    def __init__(self, df, file_hdf, transforms=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df["isic_id"].values
        self.targets = df["target"].values

        self.transforms = transforms

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))

        target = self.targets[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "target": target,
        }
