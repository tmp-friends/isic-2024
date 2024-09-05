import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from models.functions import SwishModule


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p.data.tolist()[0]}), eps={str(self.eps)})"


class EfficientNet(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=False,
        checkpoint_path=None,
        n_meta_features=0,
        n_meta_dim=[256, 32],
        num_classes=1,
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        self.n_meta_features = n_meta_features
        in_features = self.model.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                SwishModule(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                SwishModule(),
            )
            in_features += n_meta_dim[1]
        self.linear = nn.Linear(in_features, num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.sigmoid = nn.Sigmoid()
        self.model.classifier = nn.Identity()

    def extract(self, x):
        return self.model(x)

    def forward(self, x, x_meta):
        """
        Args:
            x: images
            x_meta: metadata
        """
        x = self.extract(x).squeeze(-1).squeeze(-1)

        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.linear(dropout(x))
            else:
                out += self.linear(dropout(x))

        out /= len(self.dropouts)
        out = self.sigmoid(out)

        return out
