import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc


def score_p_auc(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float = 0.80):
    """ホストから提供されているスコア計算処理"""
    v_gt = abs(np.asarray(solution.values) - 1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc


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
