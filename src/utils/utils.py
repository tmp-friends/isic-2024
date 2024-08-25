import os

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve, auc


def set_seed(seed=42):
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


def score(y_true, y_preds, min_tpr: float = 0.80):
    v_gt = abs(np.asarray(y_true) - 1)
    v_pred = -1.0 * np.asarray(y_preds)
    max_fpr = abs(1 - min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return partial_auc
