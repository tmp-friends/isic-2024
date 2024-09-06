from dataclasses import dataclass
from typing import Any


@dataclass
class DirConfig:
    data_dir: str
    all_isic_data_dir: str


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class TrainConfig:
    dir: DirConfig
    model: ModelConfig
    n_epochs: int
    img_size: int
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
class TrainPseudoConfig:
    dir: DirConfig
    model: ModelConfig
    n_epochs: int
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
class InferConfig:
    dir: DirConfig
    model: ModelConfig
    model_dir: str
    valid_batch_size: int
