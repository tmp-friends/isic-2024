from dataclasses import dataclass


@dataclass
class DirConfig:
    data: str
    data_tfrec: str


@dataclass
class TrainConfig:
    dir: DirConfig
    seed: int
    batch_size: int
    num_epochs: int
    batch_size: int
    early_stopping_patience: int
    lr: float
    max_lr: float
