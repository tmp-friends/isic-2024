from pathlib import Path
import logging
import os
import sys
import gc

import hydra
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from src.types.config import TrainConfig
from src.utils.metrics import score_p_auc


def feature_enginnering(df: pd.DataFrame):
    # New features to try...
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2
    )
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]

    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["area_to_perimeter_ratio"] = df["tbp_lv_areaMM2"] / df["tbp_lv_perimeterMM"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["consistency_symmetry_border"] = (
        df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"] / (df["tbp_lv_symm_2axis"] + df["tbp_lv_norm_border"])
    )

    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    df["consistency_color"] = df["tbp_lv_stdL"] * df["tbp_lv_Lext"] / (df["tbp_lv_stdL"] + df["tbp_lv_Lext"])
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = (
        df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    )

    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt(
        (df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3
    )
    df["color_shape_composite_index"] = (
        df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]
    ) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3

    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (
        df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]
    ) / 4
    df["color_variance_ratio"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    df["border_color_interaction"] = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]
    df["size_color_contrast_ratio"] = df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]
    df["age_normalized_nevi_confidence"] = df["tbp_lv_nevi_confidence"] / df["age_approx"]
    df["color_asymmetry_index"] = df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]

    df["3d_volume_approximation"] = df["tbp_lv_areaMM2"] * np.sqrt(
        df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2
    )
    df["color_range"] = (
        (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
        + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs()
        + (df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()
    )
    df["shape_color_consistency"] = df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]
    df["border_length_ratio"] = df["tbp_lv_perimeterMM"] / (2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi))
    df["age_size_symmetry_index"] = df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]
    df["index_age_size_symmetry"] = df["age_approx"] * df["tbp_lv_areaMM2"] * df["tbp_lv_symm_2axis"]
    # Until here..
    # df['np1']                           = np.sqrt(df["tbp_lv_deltaB"]**2 + df["tbp_lv_deltaL"]**2 + df["tbp_lv_deltaLB"]**2) / (df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLB"])
    # df['np2']                           = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaLB"]) / np.sqrt(df["tbp_lv_deltaA"]**2 + df["tbp_lv_deltaLB"]**2)
    # df['np3']                           = ?
    # ...
    # df['npn']                           = ?

    new_num_cols = [
        "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
        "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
        "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
        "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
        "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
        "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
        "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
        "3d_position_distance",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
        "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
        "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
        "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
        # "combined_anatomical_site"      # anatom_site_general     + "_" + tbp_lv_location ! categorical feature
        "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
        "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
        "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
        "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
        "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
        "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
        "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
        "shape_complexity_index",  # border_complexity       + lesion_shape_index
        "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
        "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
        "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
        "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
        "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
        "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
        "3d_lesion_orientation",  # tbp_lv_y                , tbp_lv_x  np.arctan2
        "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
        "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
        "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
        "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
        "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
        "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
        "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
        "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
        "3d_volume_approximation",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
        "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
        "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
        "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
        "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
        # "index_age_size_symmetry",      # age_approx              * sqrt(tbp_lv_areaMM2 * tbp_lv_symm_2axis)
        "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
        # Until here..
        # 'np1',                         # in case of a positive manifestation
        # 'np2',                         # in case of a positive manifestation
        # 'np3'                          # = ?
        # ...
        # 'npn'                          # = ?
    ]

    new_cat_cols = ["combined_anatomical_site"]

    return df, new_num_cols, new_cat_cols


def objective(trial):
    param = {
        "objective": "binary",
        # "metric":           "custom",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "device": "gpu",
    }

    scores = []
    for fold in range(cfg.n_splits):
        _train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        _valid_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
        dtrain = lgb.Dataset(_train_df[train_cols], label=_train_df["target"])

        model = lgb.train(param, dtrain)

        preds = model.predict(_valid_df[train_cols])

        score = score_p_auc(_valid_df[["target"]], pd.DataFrame(preds, columns=["prediction"]), "")
        scores.append(score)

    return np.mean(scores)


@hydra.main(config_path="conf", config_name="train_gbdt", version_base="1.1")
def main(cfg: TrainConfig):
    train_df = pd.read_csv(f"{cfg.dir}/train-metadata.csv")
    test_df = pd.read_csv(f"{cfg.dir}/test-metadata.csv")

    num_cols = [
        "age_approx",  # Approximate age of patient at time of imaging.
        "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
        "tbp_lv_A",  # A inside  lesion.+
        "tbp_lv_Aext",  # A outside lesion.+
        "tbp_lv_B",  # B inside  lesion.+
        "tbp_lv_Bext",  # B outside lesion.+
        "tbp_lv_C",  # Chroma inside  lesion.+
        "tbp_lv_Cext",  # Chroma outside lesion.+
        "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
        "tbp_lv_Hext",  # Hue outside lesion.+
        "tbp_lv_L",  # L inside lesion.+
        "tbp_lv_Lext",  # L outside lesion.+
        "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
        "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
        "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
        "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
        "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
        "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
        "tbp_lv_deltaLB",  #
        "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
        "tbp_lv_eccentricity",  # Eccentricity.+
        "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
        "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
        "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
        "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
        "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
        "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
        "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
        "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
        "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
        "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
        "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
        "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
        "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
    ]

    train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
    test_df[num_cols] = test_df[num_cols].fillna(train_df[num_cols].median())

    train_df, new_num_cols, new_cat_cols = feature_engineering(train_df.copy())
    test_df, _, _ = feature_engineering(test_df.copy())

    num_cols += new_num_cols

    # anatom_site_general
    cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"] + new_cat_cols
    train_cols = num_cols + cat_cols

    # TODO: 画像モデルのtrain特徴量 or preds

    category_encoder = OrdinalEncoder(
        categories="auto",
        dtype=int,
        handle_unknown="use_encoded_value",
        unknown_value=-2,
        encoded_missing_value=-1,
    )

    X_cat = category_encoder.fit_transform(train_df[cat_cols])
    for c, cat_col in enumerate(cat_cols):
        train_df[cat_col] = X_cat[:, c]

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True)
    if cfg.is_debug:
        positive_df = train_df[train_df["target"] == 1]
        negative_df = train_df[train_df["target"] == 0]
        negative_df = negative_df.sample(frac=cfg.subsample_ratio)
        train_df = pd.concat([positive_df, negative_df]).sample(frac=1.0).reset_index(drop=True)

    train_df["fold"] = -1
    for idx, (train_idx, val_idx) in enumerate(
        sgkf.split(
            train_df,
            train_df["target"],
            groups=train_df["patient_id"],
        )
    ):
        train_df.loc[val_idx, "fold"] = idx

    if cfg.optimizes_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=cfg.n_trials)
        LOGGER.info(f"Number of finished trials: {len(study.trials)}")

        trial = study.best_trial
        LOGGER.info(f"Best trial value: {trial.value}")
        for k, v in trial.params.items():
            LOGGER.info(f"params: {k}, {v}")

    """
    Number of finished trials: 21
    Best trial:
    Value: 0.18667512366806385
    Params:
        lambda_l1: 1.928905024777545e-07
        lambda_l2: 7.067500111852746
        num_leaves: 120
        feature_fraction: 0.8378182236514783
        bagging_fraction: 0.9358083467981043
        bagging_freq: 3
        min_child_samples: 89
    """
    lgb_params = {
        "objective": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "lambda_l1": 1.928905024777545e-07,
        "lambda_l2": 7.067500111852746,
        "num_leaves": 120,
        "feature_fraction": 0.8378182236514783,
        "bagging_fraction": 0.9358083467981043,
        "bagging_freq": 3,
        "min_child_samples": 89,
        "device": "gpu",
    }

    lgb_scores = []
    lgb_models = []
    oof_df = pd.DataFrame()
    for fold in range(cfg.n_folds):
        _train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        _valid_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        model = lgb.LGBMClassifier(**lgb_params)
        # model = VotingClassifier([(f"lgb_{i}", lgb.LGBMClassifier(random_state=i, **lgb_params)) for i in range(7)], voting="soft")

        model.fit(_train_df[train_cols], _train_df["target"])

        preds = model.predict_proba(_valid_df[train_cols])[:, 1]
        score = comp_score(_valid_df[["target"]], pd.DataFrame(preds, columns=["prediction"]), "")
        LOGGER.info(f"fold: {fold} - Partial AUC Score: {score:.5f}")

        lgb_models.append(model)

        oof_single = _valid_df[["isic_id", "target"]].copy()
        oof_single["pred"] = preds
        oof_df = pd.concat([oof_df, oof_single])

    lgbm_score = score_p_auc(oof_df["target"], oof_df["pred"], "")
    LOGGER.info(f"LGBM Score: {lgbm_score:.4f}")

    importances = np.mean([model.feature_importances_ for model in lgb_models], 0)
    df_imp = (
        pd.DataFrame({"feature": model.feature_name_, "importance": importances})
        .sort_values("importance")
        .reset_index(drop=True)
    )

    plt.figure(figsize=(16, 12))
    plt.barh(df_imp["feature"], df_imp["importance"])


if __name__ == "__main__":
    # Logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
    LOGGER = logging.getLogger(Path(__file__).name)

    seed()

    main()
