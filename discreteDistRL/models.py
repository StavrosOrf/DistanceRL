"""Neural network modules for the discrete Distance RL agent."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class AtariEncoder(nn.Module):
    """Nature-CNN style feature extractor for stacked Atari frames."""

    def __init__(self, in_channels: int, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, 84, 84)`` observations into latent features."""

        x = self.conv(obs)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DistanceTrunkDiscrete(nn.Module):
    """Action-conditioned representation trunk using action embeddings."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        embed_dim: int = 128,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(action_dim, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return embeddings ``z(s, a)`` for discrete actions."""

        if actions.dim() == 1:
            act_embed = self.embedding(actions)
            x = torch.cat([features, act_embed], dim=-1)
            return self.net(x)

        if actions.dim() == 2:
            act_embed = self.embedding(actions)  # (B, K, D)
            feat = features.unsqueeze(1).expand(-1, actions.size(1), -1)
            x = torch.cat([feat, act_embed], dim=-1)
            B, K, D = x.shape
            out = self.net(x.view(B * K, D))
            return out.view(B, K, -1)

        raise ValueError("Actions tensor must be 1D or 2D (batch or batch x samples).")


class CategoricalActor(nn.Module):
    """Categorical policy head operating on encoded Atari features."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits = self.policy(features)
        dist = Categorical(logits=logits)
        return dist, logits


class TwinQDiscrete(nn.Module):
    """Twin Q-value heads for discrete action spaces."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(features)
        q2 = self.q2(features)
        return q1, q2
