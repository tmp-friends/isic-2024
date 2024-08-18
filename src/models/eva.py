import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class EVA(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=1,
        pretrained=True,
        checkpoint_path=None,
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        return self.sigmoid(self.model(images))
