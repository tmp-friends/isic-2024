## Overview

### Overview
#### Description

- Background Infomation
  - 皮膚がん3大タイプ
    - Basal Cell Carcinoma(BCC): 基底細胞がん
    - Squamous Cell Carcinoma(SCC): 扁平上皮がん
    - Melanoma
  - BCCとSCCは一般的
    - アメリカでは500万人/年が罹患
  - Melanomaは最も致命的
    - アメリカでは9000人/年が死亡
  - 早期発見できれば軽い手術で治せる
- Clinical Context
  - 皮膚科医は非典型的な病変を記録するために、デジタル皮膚鏡検査を使う
  - 6大陸に渡る数千人の患者のすべての病変を含むデータセットのため、病変選択バイアスを回避するのに役立ち、一般的な良性例が過小評価される傾向にある
  - 携帯電話での撮影のため、クリニックで撮影するより質が悪い
- Image Modality
  - Whole Body Photography
    - 皮膚科用に特別設計されたVECTRA WB360全身3D画像システムは92のカメラ画像をショリすることで、1回でマクロ画質の解像度で皮膚表面全体を撮影可能
  - Tiles
    - 各病変は15x15mmの切り取られた画像として自動検出される
    - test setとtraining setはこのtilesで構成されている
    - 他のpublic datasetを使っても良い
  - Dermoscopy
    - Dermoscopy: 皮膚表面顕微鏡を用いた皮膚の検査
    - 高品質の拡大レンズと強力な照明システムが必要で、肉眼では見えない形態学的特徴を照らす
    - 大規模なdermoscopy datasetsはISICで入手可能
      - https://www.isic-archive.com/

#### Evaluation

- pAUC
  - （テストデータにおける）真陽性率80%以上の範囲のROC
  - とりうる値としては、[0.0, 0.2]
  - 真陽性率とは、実際にpositive（例えば、病気である場合）であるすべてのケースの中で、正しくpositiveと識別されたケースの割合
    - positiveをいかに当てられるか
    - =Recall
    ```
    例
    健康診断で乳がんのスクリーニングを行う場合を考えます。以下のような結果が得られたとします。

    実際に乳がんである女性の数: 100人
    このうち、検査で乳がんと正しく診断された女性の数: 90人
    乳がんであるにもかかわらず、検査で見逃された女性の数: 10人
    ```

  - 実装

    ```py
    """
    2024 ISIC Challenge primary prize scoring metric

    Given a list of binary labels, an associated list of prediction
    scores ranging from [0,1], this function produces, as a single value,
    the partial area under the receiver operating characteristic (pAUC)
    above a given true positive rate (TPR).
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

    (c) 2024 Nicholas R Kurtansky, MSKCC
    """

    import numpy as np
    import pandas as pd
    import pandas.api.types
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    class ParticipantVisibleError(Exception):
        pass


    def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80) -> float:
        '''
        2024 ISIC Challenge metric: pAUC

        Given a solution file and submission file, this function returns the
        the partial area under the receiver operating characteristic (pAUC)
        above a given true positive rate (TPR) = 0.80.
        https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

        (c) 2024 Nicholas R Kurtansky, MSKCC

        Args:
            solution: ground truth pd.DataFrame of 1s and 0s
            submission: solution dataframe of predictions of scores ranging [0, 1]

        Returns:
            Float value range [0, max_fpr]
        '''

        del solution[row_id_column_name]
        del submission[row_id_column_name]

        # check submission is numeric
        if not pandas.api.types.is_numeric_dtype(submission.values):
            raise ParticipantVisibleError('Submission target column must be numeric')

        # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
        v_gt = abs(np.asarray(solution.values)-1)

        # flip the submissions to their compliments
        v_pred = -1.0*np.asarray(submission.values)

        max_fpr = abs(1-min_tpr)

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

    #     # Equivalent code that uses sklearn's roc_auc_score
    #     v_gt = abs(np.asarray(solution.values)-1)
    #     v_pred = np.array([1.0 - x for x in submission.values])
    #     max_fpr = abs(1-min_tpr)
    #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

        return(partial_auc)
    ```

---

### Data

- Strongly-labelled tiles: 病理組織学的評価によってラベリングされたtile
- Weak-lablelled tiles: 医師がラベリングしたtile

#### train only metadata

- target: {0: benign, 1: malignant}
- lesion_id: 病変id, 注目病変として手動でタグ付けされた病変に存在
- iddx_full: 病変の診断(完全に分類されている)
- iddx_1: First level 病変の診断
- iddx_2: Second level 病変の診断
- iddx_3: Third level 病変の診断
- iddx_4: Fourth level 病変の診断
- iddx_5: Fifth level 病変の診断
- mel_mitotic_index: 浸潤性悪性黒色腫の分裂指数
- mel_thick_mm: melanomaの浸潤における暑さ
- tbp_lv_dnn_lesion_confidence: 病変の確信度score[0, 100]

#### metadata

- isic_id: id
- patient_id: 患者id
- age_approx: 撮影時のおおよその年齢
- sex: 性別
- anatom_site_general: 病変のある体の箇所
- clin_size_long_diam_mm: 病変の最大径 mm
- image_type: 画像タイプを表すISICアーカイブの構造体
- tbp_tile_type: 3DTBP画像の照明モダリティ
- tbp_lv_A: A inside lesion
- tbp_lv_Aex: A outside lesion
- tbp_lv_B: B inside lesion
- tbp_lv_Bext: B outside lesion
- tbp_lv_C: Chroma inside lesion
- tbp_lv_Cext: Chroma outside lesion
- tbp_lv_H: Hue inside lesion
- tbp_lv_Hext: Hue outside lesion
- tbp_lv_L: L inside lesion
- tbp_lv_Lext: L outside lesion
- tbp_lv_areaMM2: 病変の面積 (mm^2)
- tbp_lv_area_perim_ratio: 境界のギザギザ、病変の周囲と面積の比率。円に近いほど低い値。不規則な形状ほど高い値。
- tbp_lv_color_std_mean: 病変境界内の色のばらつきとして計算される色ムラ
- tbp_lv_deltaA: A (insideとoutsideの)コントラスト平均
- tbp_lv_deltaB: B (insideとoutsideの)コントラスト平均
- tbp_lv_deltaL: L (insideとoutsideの)コントラスト平均
- tbp_lv_deltaLBnorm: 病変とその周囲の皮膚とのコントラスト。低コントラストの病変はそばかすのようにかすかに見える傾向があり、高コントラストの病変は色素が濃い傾向がある。LAB*色空間における背景に対する病変の平均デルタLBとして計算される。典型的な値は5.5から25の範囲である。
- tbp_lv_eccentricity: 返信
- tbp_lv_location: 解剖学的位置の分類、腕と脚を上部と下部に分け、胴体を3分の1に分ける。
- tbp_lv_location_simple: 単純な解剖学的位置の分類
- tbp_lv_minorAxisMM: 最も小さい病変の径 (mm)
- tbp_lv_nevi_confidence: 母斑信頼度スコア（0-100スケール）は、畳み込みニューラルネットワーク分類器によって推定された、病変が母斑である確率である。 ニューラルネットワークは、皮膚科医によって分類されラベリングされた約57,000の病変でトレーニングされた。
- tbp_lv_norm_border: 境界の不規則性(0-10スケール)。境界のギザギザと非対称性の正規化平均値。
- tbp_lv_norm_color: 色のバリエーション(0-10スケール)。色の非対称性と不規則性の正規化平均値。
- tbp_lv_perimeterMM: 病変の周径 (mm)
- tbp_lv_radial_color_std_max: 色の非対称性：病変内の色の空間分布の非対称性の尺度。 このスコアは、病変中心を起点とする同心円内のLAB*色空間における平均標準偏差を見て計算される。 [0, 10]
- tbp_lv_stdL: 病変内のLの標準偏差
- tbp_lv_stdLExt: 病変外のLの標準偏差
- tbp_lv_symm_2axis: 境界の非対称性；病変の最も対称的な軸に垂直な軸に関する病変の輪郭の非対称性の尺度。 従って、対称軸が2つある病変はスコアが低く（より対称的）、対称軸が1つしかないか0つの病変はスコアが高く（より対称的でない）なる。 このスコアは、病変の輪郭の反対側の半分を何度も回転させて比較することにより算出される。 半分が最も似ている角度が対称性の主軸であり、対称性の第2軸は主軸に垂直である。 ボーダーの非対称性は、この第2軸に関する非対称性の値として報告される。 [0, 10]
- tbp_lv_symm_2axis_angle: 病変境界の非対称角度
- tbp_lv_x: 3DTBP上のX座標
- tbp_lv_y: 3DTBP上のY座標
- tbp_lv_z: 3DTBP上のZ座標
- attribution: 画像のattribution(image sourceと同義)
- copyright_license

#### 備考

- 浸潤性悪性腫瘍
  - 悪性腫瘍（がん）が周囲の組織や臓器に広がり、破壊する能力を持つ腫瘍
  - がん細胞は小さな塊として存在しますが、時間とともに大きくなり、隣接する組織に広がります
  - この過程を「浸潤」といい、がんは周囲の組織や臓器に影響を与え、健康を脅かすようになります
