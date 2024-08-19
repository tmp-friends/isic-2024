import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class Net(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes=1,
        pretrained=True,
        checkpoint_path=None,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, num_classes=1
        )

    def forward(self, images):
        x = self.model(images).sigmoid()

        return x
