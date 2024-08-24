## Doc

### 2024/08/25


### 2024/08/24

- Augmentation改善
  - 過去ISICコンペの1stのAugmentation: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412
- pAUCのscore関数を導入
- effinet b0, b1, b2, b5で試して、b1のCVが良かったので採用
  - CV: 0.1654, LB: 0.137

- TODO
  - 過去ISICコンペの解法調査
  - 過去ISICコンペのデータを使う
  - upsampling, downsampling改善
  - （余裕あれば）引数によってモデルを変えられるような実装にしたい
  - image+tabular

### 2024/08/23

- metadata調査

- TODO
  - image+tabularのNNを作りたい
  - image単体でどれが性能良いのか

### 2024/08/21

- 過去コンペ調査

### 2024/08/19

- swin_largeの実装&sub
  - pos:neg調整があったほうがLB良い

### 2024/08/18

- pos:negの調整をやめてeffnet_b0, eva02をsub
  - effnet_b0
  - eva02
- 他のversionのeffnetを試す
  - effnet_b1
  - effnet_b1

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
  - AUROC: 0.5495, LB: 0.139と (AUROC: 0.5177, LB: 0.132に比べて) 向上したb

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

- ISIC-2024 Pytorch Training Baseline (Swin) (ViT)
  - https://www.kaggle.com/code/qiaoyingzhang/isic-2024-pytorch-training-baseline-swin
  - https://www.kaggle.com/code/qiaoyingzhang/isic-2024-pytorch-training-baseline-vit
  - Image+Tabularのマルチモーダル
  - LB: 0.160

- Sharing my "best" ImageNet notebook
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/529457
  - 画像のみでLB: 0.151
  - Model
    - tf_eficientnetv2_b1
    - 小さいモデルから試したが、あまりモデルの大きさは重要ではない
  - Upsampling / Downsampling
    - positive sample x20
    - lesion_idのないnegative sampleは15%残す
    - lesion_idのあるnegative sampleは30%残す
    - -> positive:negative = 1:10
  - Augmentation
    - positive sampleをupsamplingしているので過学習を避けるため、augmentationを行う
    -


- Onboarding materials and references
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/515341
  - 過去の類似コンペの情報

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

- 参加記: https://qiita.com/tachyon777/items/05e7d35b7e0b53ef03dd
  - 不均衡データの扱いが難しい
    - Triple Stratified KFold: https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords
      - 同じ患者の写真枚数の均一化
      - 陽性例の均一化
      - 患者ごとの写真枚数の違いを考慮した均一化
    - upsampling
      - 陽性例を増やしてデータの均衡を保つ
      - 2nd place
  - EDAが参考になりそう
    - https://www.kaggle.com/code/tachyon777/fork-of-melanoma-tachyon-eda
  - TTA
    - test time augmentation
    - TestデータにもAugmentationをする
    - TTAをしたほうがスコアが出たそう
  - 外部データ
    - positive dataの割合が増えることによるモデルの精度向上
    - https://www.kaggle.com/code/shonenkov/merge-external-data
  - Preprocessing
    - 体毛除去より、体毛を追加してrobustなモデルにしたほうが精度良かったそう
      - 体毛除去のnotebook: https://www.kaggle.com/code/vatsalparsaniya/melanoma-hair-remove

- 1st: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412
- 2nd:
  - Backbone
    - EfficientNet-B6 512x512
    - EfficientNet B7 640x640
  -


- Nelder-Mead method
  - アンサンブルでよく用いられる
  - 最適化手法

#### PetFinderコンペ

画像+メタデータ

- 上位解法記事: https://qiita.com/taiga518/items/21c9fd96876293397e98

- 1st: https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301686
  - 画像特徴量をSVRで処理
- 2nd:

