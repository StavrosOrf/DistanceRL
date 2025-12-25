"""Neural network modules for the discrete Distance RL agent.

Each module contains its own Atari encoder so the agent code can remain simple.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        x = self.conv(obs)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class CategoricalActorNet(nn.Module):
    """Encoder + categorical policy head."""

    def __init__(self, obs_channels: int, action_dim: int, feature_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.encoder = AtariEncoder(obs_channels, feature_dim)
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        state_latent = self.encoder(obs)
        logits = self.policy(state_latent)
        dist = Categorical(logits=logits)
        return dist, logits


class TwinQDiscreteNet(nn.Module):
    """Encoder + twin Q-value heads for discrete action spaces."""

    def __init__(self, obs_channels: int, action_dim: int, feature_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.encoder = AtariEncoder(obs_channels, feature_dim)
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

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_latent = self.encoder(obs)
        q1 = self.q1(state_latent)
        q2 = self.q2(state_latent)
        return q1, q2

class DistanceTrunkDiscreteNet(nn.Module):
    """Encoder + action-conditioned representation trunk using action embeddings."""

    def __init__(
        self,
        obs_channels: int,
        action_dim: int,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        use_one_hot_actions: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        embed_dim = hidden_dim // 4
        self.encoder = AtariEncoder(obs_channels, feature_dim)
        self.action_dim = action_dim
        self.use_one_hot_actions = use_one_hot_actions
        self.verbose = verbose
        self.embedding = nn.Embedding(action_dim, embed_dim)
        self.onehot_proj = nn.Linear(action_dim, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_latent = self.encoder(obs)
        return self._forward_latent(state_latent, actions)

    def _forward_latent(self, state_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.verbose:
            print(f'actions shape: {actions.shape}')

        # Floating inputs are treated as probability/simplex vectors and produce an expected embedding
        # from the shared embedding table to ensure the kernel path uses trained action embeddings.
        if torch.is_floating_point(actions):
            if actions.size(-1) != self.action_dim:
                raise ValueError(f"Floating-point actions must have last dim == action_dim; got {actions.shape}")

            embed_w = self.embedding.weight  # (A, D)

            if actions.dim() == 2:
                act_embed = actions @ embed_w  # (B,D)
                if self.verbose:
                    print('using probability-weighted embedding (B,A) -> (B,D)')
                    print(f'act_embed shape: {act_embed.shape}')
                    print(f'latent shape: {state_latent.shape}')
                x = torch.cat([state_latent, act_embed], dim=-1)
                if self.verbose:
                    print(f'x shape: {x.shape}')
                out = self.net(x)
                if self.verbose:
                    print(f'out shape: {out.shape}')
                return out

            if actions.dim() == 3:
                B, K, A = actions.shape
                act_embed = actions.view(B * K, A) @ embed_w  # (B*K,D)
                act_embed = act_embed.view(B, K, -1)
                if self.verbose:
                    print('using probability-weighted embedding (B,K,A) -> (B,K,D)')
                    print(f'act_embed shape: {act_embed.shape}')
                    print(f'latent shape: {state_latent.shape}')
                latent = state_latent.unsqueeze(1).expand(-1, K, -1)
                if self.verbose:
                    print(f'latent expanded shape: {latent.shape}')
                x = torch.cat([latent, act_embed], dim=-1)
                if self.verbose:
                    print(f'x shape: {x.shape}')
                out = self.net(x)
                if self.verbose:
                    print(f'out shape: {out.shape}')
                return out

            raise ValueError(f"Floating-point actions must be 2D or 3D; got {actions.dim()}D")

        if actions.dim() == 2:
            B, K = actions.shape
            if K == 1:
                idx = actions.view(-1).long()
                if self.verbose:
                    print(f'idx shape: {idx.shape}')
                    print(f'idx: {idx[0]}')
                if self.use_one_hot_actions:
                    one_hot = F.one_hot(idx, num_classes=self.action_dim).float()  # (B,A)
                    if self.verbose:
                        print(f'one_hot shape: {one_hot.shape}')
                        print(f'one_hot: {one_hot[0,:]}')
                    act_embed = self.onehot_proj(one_hot)
                else:
                    act_embed = self.embedding(idx)
                if self.verbose:
                    print(f'act_embed shape: {act_embed.shape}')
                    print(f'latent shape: {state_latent.shape}')
                x = torch.cat([state_latent, act_embed], dim=-1)
                if self.verbose:
                    print(f'x shape: {x.shape}')
                out = self.net(x)
                if self.verbose:
                    print(f'out shape: {out.shape}')
                return out

            idx = actions.long()
            if self.verbose:
                print(f'idx shape: {idx.shape}')
                print(f'idx[0]: {idx[0]}')
            if self.use_one_hot_actions:
                one_hot = F.one_hot(idx.view(-1), num_classes=self.action_dim).float()  # (B*K, A)
                if self.verbose:
                    print(f'one_hot shape: {one_hot.shape}')
                    print(f'one_hot[0]: {one_hot[0,:]}')
                act_embed = self.onehot_proj(one_hot).view(B, K, -1)
            else:
                act_embed = self.embedding(idx)  # (B,K,D)
            if self.verbose:
                print(f'act_embed shape: {act_embed.shape}')
                print(f'latent shape: {state_latent.shape}')
            latent = state_latent.unsqueeze(1).expand(-1, K, -1)
            if self.verbose:
                print(f'latent expanded shape: {latent.shape}')
            x = torch.cat([latent, act_embed], dim=-1)
            if self.verbose:
                print(f'x shape: {x.shape}')
            out = self.net(x)
            if self.verbose:
                print(f'out shape: {out.shape}')
            return out

        if actions.dim() == 3:
            idx = actions.long()
            if self.verbose:
                print(f'idx shape: {idx.shape}')
                print(f'idx[0]: {idx[0]}')
            B, K = idx.shape[:2]
            if self.use_one_hot_actions:
                one_hot = F.one_hot(idx.view(-1), num_classes=self.action_dim).float()  # (B*K, A)
                if self.verbose:
                    print(f'one_hot shape: {one_hot.shape}')
                    print(f'one_hot[0]: {one_hot[0,:]}')
                act_embed = self.onehot_proj(one_hot).view(B, K, -1)
            else:
                act_embed = self.embedding(idx)  # (B,K,D)
            if self.verbose:
                print(f'act_embed shape: {act_embed.shape}')
                print(f'latent shape: {state_latent.shape}')
            latent = state_latent.unsqueeze(1).expand(-1, K, -1)
            if self.verbose:
                print(f'latent expanded shape: {latent.shape}')
            x = torch.cat([latent, act_embed], dim=-1)
            if self.verbose:
                print(f'x shape: {x.shape}')
            out = self.net(x)
            if self.verbose:
                print(f'out shape: {out.shape}')
            return out

        if actions.dim() == 1:
            idx = actions.long()
            if self.verbose:
                print(f'idx shape: {idx.shape}')
                print(f'idx: {idx[0]}')
            if self.use_one_hot_actions:
                one_hot = F.one_hot(idx, num_classes=self.action_dim).float()  # (B,A)
                if self.verbose:
                    print(f'one_hot shape: {one_hot.shape}')
                    print(f'one_hot: {one_hot[0,:]}')
                act_embed = self.onehot_proj(one_hot)
            else:
                act_embed = self.embedding(idx)
            if self.verbose:
                print(f'act_embed shape: {act_embed.shape}')
                print(f'latent shape: {state_latent.shape}')
            x = torch.cat([state_latent, act_embed], dim=-1)
            if self.verbose:
                print(f'x shape: {x.shape}')
            out = self.net(x)
            if self.verbose:
                print(f'out shape: {out.shape}')
            return out

        raise ValueError(f"Unsupported action tensor shape: {actions.shape}")


