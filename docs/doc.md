## Doc

### 2024/08/17

- EVA02の実装&sub
  - AUROC: 0.5181, LB: 0.142
  - 参考
    - train: https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-eva02
    - infer: https://www.kaggle.com/code/motono0223/isic-inference-eva02-for-training-data
  - pseudo labelingを使ったコードもあるので試してみる
    - https://www.kaggle.com/code/motono0223/isic-pytorch-baseline-pseudo-labeling-eva02

- timmのモデルによってアーキテクチャが違い、コードを使い回せない
  - modelにif文で分岐させるようにする？
- pos:neg = 1:20にしているのをやめて学習させてみる
  - effnet_b0
  - AUROC: , LB:

### 2024/08/15

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

- 画像特徴量を得る公開notebookを見る
  - https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-image-only/notebook
    - backbone: tf_efficientnet_b0_ns
    - CosineAnnealingLR
  - 画像特徴量の重要性をFeatureImportanceで見る

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

### Survey


- Onboarding materials and references
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/515341

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

#### SIIM-ISIC Melanoma Classification
画像+メタデータ

-

#### PetFinderコンペ
画像+メタデータ

- 上位解法: https://qiita.com/taiga518/items/21c9fd96876293397e98
