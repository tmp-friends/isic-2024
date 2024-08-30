import os
import random

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve, auc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_processed_img(dataset: Dataset):
    for i in range(2):
        # figsizeをsubplotsに直接指定
        f, axarr = plt.subplots(1, 5, figsize=(20, 10))
        for p in range(5):
            ix = np.random.randint(0, len(dataset))
            data = dataset[ix]

            image = data["image"]
            target = data["target"]

            if image.shape[0] == 3:  # チャンネルが3つある場合、RGB画像と仮定
                image = image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

            # 画像の表示
            axarr[p].imshow(image)
            axarr[p].set_title(str(target))
            axarr[p].axis("off")  # 軸の表示をオフ

        plt.savefig(f"plt-processed-img{i}.png")
        plt.clf()
