import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc


def score_p_auc(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float = 0.80):
    """ホストから提供されているスコア計算処理"""
    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray((solution["target"] >= 1.0).astype(int).values) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(submission["prediction"].values)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return partial_auc


# def score_p_auc(solution, submission, min_tpr=0.80):
#     v_gt = (solution["target"] >= 1.0).astype(int).values
#     # v_gt = solution["target"].values
#     v_pred = submission["prediction"].values
#     max_fpr = 1 - min_tpr

#     fpr, tpr, _ = roc_curve(v_gt, v_pred)
#     if max_fpr is None or max_fpr == 1:
#         return auc(fpr, tpr)
#     if max_fpr <= 0 or max_fpr > 1:
#         raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

#     stop = np.searchsorted(fpr, max_fpr, "right")
#     if stop >= len(fpr):
#         return auc(fpr, tpr)  # Handle case where max_fpr is beyond the maximum fpr
#     x_interp = [fpr[stop - 1], fpr[stop]]
#     y_interp = [tpr[stop - 1], tpr[stop]]
#     tpr_at_max_fpr = np.interp(max_fpr, x_interp, y_interp)

#     return tpr_at_max_fpr


def score_p_auc_with_torch(y_true, y_preds, min_tpr: float = 0.80):
    """ホストから提供されているスコア計算処理の入力が違うパターン"""
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
