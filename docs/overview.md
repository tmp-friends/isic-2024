## Overview
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
