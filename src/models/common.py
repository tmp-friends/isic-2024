from models.efficientnet import EfficientNet
from conf.type import ModelConfig


def get_model(cfg: ModelConfig, is_pretrained=False, n_meta_features=0):
    if cfg.name.startswith("EfficientNet"):
        model = EfficientNet(
            model_name=cfg.params.model_name,
            pretrained=is_pretrained,
            checkpoint_path=cfg.params.checkpoint_path if is_pretrained else None,
            n_meta_features=n_meta_features,
        )

    return model
