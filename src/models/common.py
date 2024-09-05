from models.efficientnet import EfficientNet
from models.eva02 import EVA02
from models.net import Net
from conf.type import ModelConfig


def get_model(cfg: ModelConfig, is_pretrained=False, n_meta_features=0):
    if cfg.name.startswith("EfficientNet"):
        model = EfficientNet(
            model_name=cfg.params.model_name,
            pretrained=is_pretrained,
            checkpoint_path=cfg.params.checkpoint_path if is_pretrained else None,
            n_meta_features=n_meta_features,
        )
    elif cfg.name == "EVA02":
        model = EVA02(
            model_name=cfg.params.model_name,
            pretrained=is_pretrained,
            checkpoint_path=None,
            n_meta_features=n_meta_features,
        )
    else:
        model = Net(
            model_name=cfg.params.model_name,
            pretrained=is_pretrained,
            checkpoint_path=None,
            n_meta_features=n_meta_features,
        )

    return model
