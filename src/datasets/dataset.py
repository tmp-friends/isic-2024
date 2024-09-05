import os
import random
import h5py
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2

from datasets.transforms import define_transforms
from conf.type import TrainConfig


class ISICDataset(Dataset):
    def __init__(self, cfg, df: pd.DataFrame, file_path: str, meta_features: list = None, is_training=True):
        self.df = df
        self.file_path = h5py.File(file_path, mode="r")
        self.uses_meta = meta_features is not None
        self.meta_features = meta_features
        self.transforms = define_transforms(cfg=cfg, is_training=is_training)
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        row = self.df.iloc[ix]

        isic_id = row["isic_id"]
        target = row["target"]

        img = np.array(Image.open(BytesIO(self.file_path[isic_id][()])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(image=img)["image"]

        if self.uses_meta:
            meta = row[self.meta_features]
            if meta.isna().any():  # NaNをチェック
                LOGGER.info(f"NaN detected in metadata for index {ix}, ISIC ID {isic_id}")
            meta = np.array(meta, dtype=np.float32)
        else:
            meta = np.array([])

        return {
            "image": img,
            "meta": meta,
            "target": target,
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


def get_meta(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    disc_cols = ['patient_id', 'age_approx', 'sex', 'anatom_site_general', 'tbp_tile_type',  'tbp_lv_location', 'tbp_lv_location_simple']
    """

    """
    対数変換を行うメリット

    - 偏りの軽減：一部の大きな値によって生じるデータの偏りを軽減できる
    - 外れ値の影響緩和：データの外れ値による影響を緩和できる
    - 正規分布への近似：多くの機械学習モデルが正規分布するデータを前提としているため、非正規分布データを正規分布に近づけることで、予測精度が向上する可能性がある

    ref: https://note.com/sasayaka360/n/n5d166a796d66
    """
    # age
    train_df["age_approx"] /= 90
    test_df["age_approx"] /= 90
    train_df["age_approx"] = train_df["age_approx"].fillna(0)
    test_df["age_approx"] = test_df["age_approx"].fillna(0)

    # sex
    train_df["sex"] = train_df["sex"].map({"male": 1, "female": 0})
    test_df["sex"] = test_df["sex"].map({"male": 1, "female": 0})
    train_df["sex"] = train_df["sex"].fillna(-1)
    test_df["sex"] = test_df["sex"].fillna(-1)

    # anatom_site_general
    con = pd.concat([train_df["anatom_site_general"], test_df["anatom_site_general"]], ignore_index=True)
    dummies = pd.get_dummies(con, dummy_na=True, dtype=np.uint8, prefix="site")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1)

    # tbp_tile_type
    con = pd.concat([train_df["tbp_tile_type"], test_df["tbp_tile_type"]], ignore_index=True)
    dummies = pd.get_dummies(con, dummy_na=True, dtype=np.uint8, prefix="tile_type")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1)

    # tbp_lv_location
    con = pd.concat([train_df["tbp_lv_location"], test_df["tbp_lv_location"]], ignore_index=True)
    dummies = pd.get_dummies(con, dummy_na=True, dtype=np.uint8, prefix="lv_location")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1)

    # tbp_lv_location_simple
    con = pd.concat([train_df["tbp_lv_location_simple"], test_df["tbp_lv_location_simple"]], ignore_index=True)
    dummies = pd.get_dummies(con, dummy_na=True, dtype=np.uint8, prefix="lv_location_simple")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1)

    # n_images per user
    train_df["n_images"] = train_df["patient_id"].map(train_df.groupby(["patient_id"])["isic_id"].count())
    test_df["n_images"] = test_df["patient_id"].map(test_df.groupby(["patient_id"])["isic_id"].count())
    train_df["n_images"] = np.log1p(train_df["n_images"].values)
    test_df["n_images"] = np.log1p(test_df["n_images"].values)

    columns_to_apply = [
        "clin_size_long_diam_mm",
        "tbp_lv_areaMM2",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_minorAxisMM",
        "tbp_lv_nevi_confidence",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max",
        "tbp_lv_stdL",
        "tbp_lv_symm_2axis",
        "tbp_lv_symm_2axis_angle",
    ]
    train_df[columns_to_apply] = train_df[columns_to_apply].apply(np.log1p)
    test_df[columns_to_apply] = test_df[columns_to_apply].apply(np.log1p)

    meta_features = [
        "age_approx",
        "sex",
        "n_images",
        "clin_size_long_diam_mm",
        "tbp_lv_A",
        "tbp_lv_Aext",
        "tbp_lv_B",
        "tbp_lv_Bext",
        "tbp_lv_C",
        "tbp_lv_Cext",
        "tbp_lv_H",
        "tbp_lv_Hext",
        "tbp_lv_L",
        "tbp_lv_Lext",
        "tbp_lv_areaMM2",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaA",
        "tbp_lv_deltaB",
        "tbp_lv_deltaL",
        "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_minorAxisMM",
        "tbp_lv_nevi_confidence",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max",
        "tbp_lv_stdL",
        "tbp_lv_stdLExt",
        "tbp_lv_symm_2axis",
        "tbp_lv_symm_2axis_angle",
        "tbp_lv_x",
        "tbp_lv_y",
        "tbp_lv_z",
    ] + [
        col
        for col in train_df.columns
        if col.startswith(("site_", "tile_type_", "lv_location_", "lv_location_simple_"))
    ]
    n_meta_features = len(meta_features)

    return train_df, test_df, meta_features, n_meta_features


def load_data(cfg: TrainConfig):
    train_df = pd.read_csv(os.path.join(cfg.dir.data_dir, "train-metadata.csv"))
    test_df = pd.read_csv(os.path.join(cfg.dir.data_dir, "test-metadata.csv"))

    train_df, test_df, meta_features, n_meta_features = get_meta(train_df, test_df)

    pos_train_df = train_df[train_df["target"] == 1].reset_index(drop=True)
    neg_train_df = train_df[train_df["target"] == 0].reset_index(drop=True)

    # positive:negative=1:20になるようDown sampling
    # positiveが少ない不均衡データなので学習がうまくいくようにする意図
    train_df = pd.concat([pos_train_df, neg_train_df.iloc[: pos_train_df.shape[0] * 20, :]]).reset_index(drop=True)

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

    # train_df["benign_malignant"].fillna("unknown", inplace=True)

    # # class mapping diagnosis2idx = {v: i for i, v in enumerate(sorted(train_df["benign_malignant"].unique()))}
    # train_df["target"] = train_df["benign_malignant"].map(diagnosis2idx)

    return train_df, test_df, meta_features, n_meta_features
