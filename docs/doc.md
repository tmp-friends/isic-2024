## Doc

TODO
- 過去コンペデータの利用
  - 考えることが多すぎるので、後回し
  - 2024だけのデータでメタデータ, pseudoやって時間が余ったら実験する
- メタデータ追加
- pseudo labeling

- 様々なモデルを作成
- Stackingに組み込む
- Feature Engineering
- アンサンブル手法

### Image model

#### Augmentation

- 過去ISICコンペの1stのAugmentation: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412
- effnet 比較
  - b0 loss: 0.5474, CV: 0.1484
    - 2024-08-24/14-35-07
  - b1 CV: 0.1654, LB: 0.137
  - b2 loss: 0.2904, CV: 0.1584
    - 2024-08-24/14-29-23
  - b5 loss: 0.2429, CV: 0.1553
    - 2024-08-24/14-17-53

- Microscope CV: 0.1602, LB: 0.137
  - 入れてもLBに変化がないし、CVは下がるので入れない

#### 不均衡データ

- 1. data
  - ISIC archive
    - isic-cliで全データ(Dermoscopic?)を取得し、メタデータをcsv/画像データをhdf形式にしたDatasetがあった
      - https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629/data
      - ISIC archive: https://www.isic-archive.com/
      - isic-cli: https://github.com/ImageMarkup/isic-cli
    - ISIC archiveには重複があるそうなので、除去する
      - https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/161943
- 2. Sampler
  - WeightedRandomSampler
    - https://www.kaggle.com/code/syzygyfy/addressing-the-class-imbalance-in-pytorch
    - CV: 0.133828
    - overfitしてそう
  - DownSampler自作
    - https://www.kaggle.com/code/taikimori/overdownsampler-isic-cnn-training-inference

- 参考: https://arxiv.org/pdf/1710.05381

#### Backborn

- effnet_b0を使った手法を実装&sub
  - AUROC: 0.5177, LB: 0.132
  - 参考コードではLB0.144出ているのに低い
  - seedは合わせているので再現はできているはず
  - GPUマシンの違い？
  - 参考
    - train: https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-image-only
    - infer: https://www.kaggle.com/code/motono0223/isic-pytorch-inference-baseline-image-only

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

- pos:negの調整をやめてeffnet_b0, eva02をsub
  - effnet_b0
  - eva02

- 他のversionのeffnetを試す
  - effnet_b1
  - effnet_b1

- swin_largeの実装&sub
  - pos:neg調整があったほうがLB良い

### GBDT

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

- BaselineとしていたGBDT notebookを画像モデルのpreds抜きでどの程度スコアが出るかを見る
  - CV: lgb 0.15648, cat 0.16157
  - LB: 0.166

- 公開notebook(LB: 0.184)の変更を取り入れる
- AutoNN
  - 画像特徴量はなしで進める

