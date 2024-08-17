## Doc

### 2024/08/17

- Only Tabular data LB:0.181
    - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/528582
    - LB0.181を達成したが、CVは低い
    - LBはあまり信用ならないという意見
    - 実装: https://www.kaggle.com/code/yunsuxiaozi/isic-2024-starter
        - 7.find best fusion weightの実装が参考になる

- [Solved] 2018-2020 datasets worsen LB. Why?
    - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/528453
    - 過去のデータセットを加えるとCVは良くなるが、LBは悪化する
    - 過去のデータセットに今回と同じデータが含まれており、LEAKしている説

- ISIC 2024 | Parallel image lines + 1 new line
    - LB0.183のPublicNotebook
    - https://www.kaggle.com/code/vyacheslavbolotin/isic-2024-parallel-image-lines-1-new-line?scriptVersionId=192839706

- LB 0.170 Image + Tabular data single fold
    - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/528032
    - LB 0.170を出したモデル情報
    - efficientnet b1のため試してみる

### 2024/08/15

- Benchmarking Image Models for ISIC 2024
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/527023
  - timmで扱える画像モデルの性能比較
    - 単体性能
    - モデルの推論の相関
    - Ensembleの結果
  - 実験notebook例: https://www.kaggle.com/code/inoueu1/isic2024-baseline-image-model?scriptVersionId=191789087
  - 単体ではswin_largeが強そう
  - Emsembleで相関の小さいmaxvitを入れたりすると良さげ
  - CV, LBの相関が小さい
    - positive labelが少ないのでfold１つに対してのCVだと安定しない
    - コメントで前回大会で提案されたtriple stratifiedにするといいとのこと
  - Augmentationでより頑健にする
  - マルチモーダルなモデル
    - tableとimageを同時に扱えるアーキテクチャのモデルを試してみる

- Sci Data article published describing the dataset
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/528081
  - 今回のDatasetについてホスト側が説明している論文
  - https://www.nature.com/articles/s41597-024-03743-w

- ISIC - Detect Skin Cancer - Let's Learn Together
  - https://www.kaggle.com/code/dschettler8845/isic-detect-skin-cancer-let-s-learn-together
  - EDA

- TODO:
  - pAUCをCVとして使えるようにする
  - swin_largeのみでsub
  - stacking
    - swin_largeの推論結果も加えて
    - AutoNNもstackingに加える

### 2024/08/15

- effnet_b0を使った手法を実装&sub
  - AUROC: 0.5177, LB: 0.132
  - 参考コードではLB0.144出ているのに低い
  - seedは合わせているので再現はできているはず
  - GPUマシンの違い？
  - 参考
    - train: https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-image-only
    - infer: https://www.kaggle.com/code/motono0223/isic-pytorch-inference-baseline-image-only

### 2024/07/31

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

### 2024/07/29

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
