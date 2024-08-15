## Doc
### EXP

#### 2024/07/31

- `ISIC2024 - Baseline` のパラメータをいじってみる
  - Optunaによる最適化
    - LGBM Score: 0.18445 -> 0.18537
    - CatBoost Score: 0.18467 -> 0.18421 (採用せず)
    - LB:
  - VotingClassifier
    - LGBM Score:
      - あり: 0.18525
      - なし: 0.18537
    - CatBoost Score:
      - あり: 0.18467
      - なし: 0.18414
- xgb
- AutoNN

- 画像特徴量を得る公開notebookを見る
  - https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-image-only/notebook
    - backbone: tf_efficientnet_b0_ns
    - CosineAnnealingLR
  - 画像特徴量の重要性をFeatureImportanceで見る
    -


#### 2024/07/29

- `ISIC2024 - Baseline` を作ってひとまずsub提出
  - copy元: https://www.kaggle.com/code/vyacheslavbolotin/lightgbm-catboost-with-new-features?scriptVersionId=190166764
    - LGBM Score: 0.18445
    - CatBoost Score: 0.18467
    - LB: 0.176
  - Ver1: https://www.kaggle.com/code/komekami/isic2024-baseline?scriptVersionId=190276791
    - LB: 0.176
  - Ver2: https://www.kaggle.com/code/komekami/isic2024-baseline?scriptVersionId=190283639
    - VotingClassifierによるEnsemble
    - LB: 0.175
  - Ver5:
    - lgb: cat = 5:5
    - LB: 0.175
  - Ver6:
    - lgb: cat = 5:5
    - LB: 0.174

- 案
  - ImageFeature
    - MetricLearning
    - Mamba
  - Stucking
    - 以下のモデルもサクッと試してみる
      - xgb
      - AutoNN
  - データのかさ増し
    - 皮膚がんのDatasetは他にもありそう
      - ISICの過去コンペ
      - https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
      - https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset

---

### Survey

#### ISIC - Detect Skin Cancer - Let's Learn Together

https://www.kaggle.com/code/dschettler8845/isic-detect-skin-cancer-let-s-learn-together

- EDA

---

### Overview

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

---

### Data
