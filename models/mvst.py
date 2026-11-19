"""Multi-view slice transformer for AD classification.

Each scan is represented by K slices from each anatomical view. A shared
ImageNet-pretrained CNN embeds every slice, and a transformer encoder models
relations across views/slices.
"""
import torch
import torch.nn as nn
from torchvision import models as tv_models


class MultiViewSliceTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        num_slices: int = 12,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Linear(self.feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices, d_model))
        self.pos_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, 3, H, W)
        batch, slices = x.shape[:2]
        features = self.backbone(x.flatten(0, 1)).view(batch, slices, -1)
        tokens = self.proj(features)

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + torch.cat(
            [torch.zeros_like(self.cls_token), self.pos_embed], dim=1
        )
        tokens = self.pos_drop(tokens)
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0]))
