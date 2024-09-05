## Survey

- Removing one feature increased my CV by 0.002
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/532024
  - ある特徴量を削除したら、CVが0.002向上したらしい
  - commentで、mel_thick_mmをTargetにモデルを学習させる、といったものがあって良さそうに思えた

- Top Solutions Ensemble - [0.947]
  - https://www.kaggle.com/code/cdeotte/top-solutions-ensemble-0-947/notebook
  - np.stackでのensemble手法

- Some pointers from the dataset's official paper to follow through
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/531115
  - 公式のdataset論文の要約

- Infer from only images with balanced mixup
  - https://www.kaggle.com/code/yoshikuwano/infer-from-only-images-with-balanced-mixup
  - LB: 0.152

- OverDownSampler | ISIC-CNN training & Inference
  - https://www.kaggle.com/code/taikimori/overdownsampler-isic-cnn-training-inference/notebook
  - LB: 0.142
  - 不均衡データに対して、Samplerで対策している
  - WeightedRandomSampler
  - 自作DownSampler

- More Training Data (Processed JPEGs) - Upsampling Malignant Images
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/515356
  - 2020, 2019, 2018のデータセットを作った
  - 今回のコンペだと393しかPositive targetが存在しない
  - 2020 Distribution: 32542 - benign, 584 - malignant
  - 2019 Distribution: 20809 - benign, 4522 - malignant
  - 2018 Distribution: 8197 - benign, 785 - malignant
  - 全データ版: https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629/data
    - 重複データがあるそう

- What Image and CNN are working for you?
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/529487
  - Data sampling and weighted lossが効いたそう
  - > Am using Convnext pico, with adaptive pooling, cross entropy (2 outputs, rather than 1), heavy augmentations, Cosine Annealing, single fold, LB: 0.161
  - Convnext使う?
  - cross entropy (2 outputs, rather than 1)は過去コンペで分類数を2->3へ増やしたらスコアが出た(2nd)らしいので筋が良さそう
  - MixUp試してないかも

- ISIC-2024 Pytorch Training Baseline (Swin) (ViT)
  - https://www.kaggle.com/code/qiaoyingzhang/isic-2024-pytorch-training-baseline-swin
  - https://www.kaggle.com/code/qiaoyingzhang/isic-2024-pytorch-training-baseline-vit
  - Image+Tabularのマルチモーダル
  - Swin+ViTでLB: 0.160

- Sharing my "best" ImageNet notebook
  - https://www.kaggle.com/competitions/isic-2024-challenge/discussion/529457
  - notebook: https://www.kaggle.com/code/richolson/isic-2024-imagenet-lr-ramp-target-mods
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

- https://www.kaggle.com/code/datafan07/analysis-of-melanoma-metadata-and-effnet-ensemble
- https://www.kaggle.com/code/awsaf49/xgboost-tabular-data-ml-cv-85-lb-787#Image-Size

### SIIM-ISIC Melanoma Classification

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
  - Data
    - https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
    - 2020 JPEG Melanoma 256x256
    - 2019, 2018のデータも加える
    - cv_2020とcv_allの2つの指標を見て実験サイクルを回した
    - 過去データを加えることでデータの不均衡は改善される
  - Augmentations

    ```py
    transforms_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(limit=0.2, p=0.75),
        A.RandomContrast(limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        A.Normalize()
    ])

    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()
    ])

    ```

  - Model
    - Backbone
      - EfficientNet-B3-B7
      - se_resnext101
      - resnest101
    - 画像サイズ
      - 384~896
    - 画像CNNとメタNNをconcat
      - https://www.kaggle.com/code/awsaf49/xgboost-tabular-data-ml-cv-85-lb-787#Image-Size
      - https://www.kaggle.com/code/nroman/melanoma-pytorch-starter-efficientnet
    - Targets
      - BCE Loss(2クラス分類)からCE Loss(多クラス分類)にすることで0.01ほどスコアを向上できた
      - 2020年と2019年のデータでtargetが異なるので、2020年データの9クラスを手動でマッピング
  - Post processing
    - 異なるfold,modelをアンサンブルする場合、それぞれの推論結果にrankをつける
    - その後、均等に分布させる
  - 実装: https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/tree/master

- 2nd: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175324
  - Data
    - Positive targetが2019年のものばかりになるので、2020年のPositive targetをUpsampling x7
      - 2019年と2020年のPositive targetをほぼ同数にした
  - Model
    - Backbone
      - EfficientNet-B6 512x512
      - EfficientNet B7 640x640
    - Targets
      - benign nevi, melanoma, otherの3クラス分類
        - 細かく分類したほうが、benign, melanomaの2クラス分類よりモデルの性能が良くなるという仮説
      - melanomaの大部分は暗色で、benign nevi(良性母斑)との識別が最も困難
      - 2019年のデータにはneviを含む補助診断があったが、2020年のデータにはない
      - 2019年のデータのみでモデルを学習し、2020年のデータに適用
        - 2020年のunknownはneviかneviでないかの判定をすれば良い(melanomaではないため)
        - 2019年のモデル予測の5%タイルを使用し、neviのラベルを付けた

- Nelder-Mead method
  - アンサンブルでよく用いられる
  - 最適化手法

- 21st Public - 53rd Private - Trust Your CV
  - https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175344
  - chrisさんの解法
  - 同じ画像を複数回RandomCropしてUpSampling

### PetFinderコンペ

画像+メタデータ

- 上位解法記事: https://qiita.com/taiga518/items/21c9fd96876293397e98

- 1st: https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301686
  - 画像特徴量をSVRで処理
