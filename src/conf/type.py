from dataclasses import dataclass


@dataclass
class DirConfig:
    data_dir: str
    train_meta_csv: str
    train_image_dir: str
    test_meta_csv: str
    test_image_hdf: str
    sample_csv: str


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
    n_epochs: int
    img_size: int
    model_name: str
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


@dataclass
class InferEffnetConfig:
    dir: DirConfig
    img_size: int
    model_name: str
    valid_batch_size: int


@dataclass
class TrainTimmModelConfig:
    dir: DirConfig
    n_epochs: int
    img_size: int
    model_name: str
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


@dataclass
class InferTimmModelConfig:
    dir: DirConfig
    img_size: int
    model_name: str
    valid_batch_size: int
