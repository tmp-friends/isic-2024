from dataclasses import dataclass


@dataclass
class DirConfig:
    data: str
    train_image_data: str


@dataclass
class TrainConfig:
    dir: DirConfig
    seed: int
    batch_size: int
    n_epochs: int
    batch_size: int
    early_stopping_patience: int
    lr: float
    max_lr: float


@dataclass
class TrainEffnetConfig:
    dir: DirConfig
    seed: int
    n_epochs: int
    img_size: int
    model_name: str
    # https://www.kaggle.com/models/timm/tf-efficientnet/PyTorch/tf-efficientnet-b0/1
    checkpoint_path: str
    train_batch_size: int
    valid_batch_size: int
    scheduler: str
    lr: float
    min_lr: float
    T_max: int
    weight_decay: float
    fold: int
    n_folds: int
    n_accumulates: int
