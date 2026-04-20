from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, image_channels: int, image_feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, image_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class BehaviorCloningPolicy(nn.Module):
    def __init__(
        self,
        image_channels: int,
        image_feature_dim: int,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_channels, image_feature_dim)
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(image_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        state_features = self.state_encoder(state)
        fused = torch.cat([image_features, state_features], dim=-1)
        return self.head(fused)


def build_model(config: dict) -> BehaviorCloningPolicy:
    model_cfg = config["model"]
    return BehaviorCloningPolicy(
        image_channels=int(model_cfg["image_channels"]),
        image_feature_dim=int(model_cfg["image_feature_dim"]),
        state_dim=int(model_cfg["state_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        action_dim=int(model_cfg["action_dim"]),
    )
