"""Neural network modules for the discrete Distance RL agent.

Each module contains its own Atari encoder so the agent code can remain simple.
"""
from __future__ import annotations

from typing import Optional, Tuple

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

    def __init__(
        self,
        obs_channels: int,
        action_dim: int,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else AtariEncoder(
            obs_channels, feature_dim)
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

    def __init__(
        self,
        obs_channels: int,
        action_dim: int,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else AtariEncoder(
            obs_channels, feature_dim)
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
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        embed_dim = hidden_dim // 4
        self.encoder = encoder if encoder is not None else AtariEncoder(
            obs_channels, feature_dim)
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

    def _forward_latent(
        self,
        state_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute latent representation z = f([h(obs), e(actions)]).

        The caller provides state_latent = h(obs) so we avoid double-encoding
        observations. Only action embeddings are computed here.

        Supports:
        - Integer actions:
            (B,) or (B, K) with dtype long/int
        - Floating actions (one-hot or probability vectors over actions):
            (B, A) or (B, K, A) with dtype float

        If self.use_one_hot_actions is True:
        - action embedding ALWAYS uses self.onehot_proj (for both int and float actions),
            ensuring consistency between rep loss and kernel path.

        If self.use_one_hot_actions is False:
        - int actions use nn.Embedding lookup
        - float actions use actions @ embedding.weight
        """

        # ---- 1) already have encoded observation ----
        h = state_latent

        # ---- 2) compute action embedding e(actions) ----
        A = self.action_dim  # number of discrete actions

        if torch.is_floating_point(actions):
            # actions is either (B, A) or (B, K, A)
            if actions.dim() not in (2, 3):
                raise ValueError(
                    f"Float actions must have dim 2 or 3, got shape {tuple(actions.shape)}"
                )
            if actions.size(-1) != A:
                raise ValueError(
                    f"Float actions last dim must be action_dim={A}, got {actions.size(-1)}"
                )

            if self.use_one_hot_actions:
                # Use onehot_proj for float action vectors too (CRUCIAL for consistency)
                if actions.dim() == 2:
                    # (B, A) -> (B, embed_dim)
                    act_embed = self.onehot_proj(actions)
                else:
                    # (B, K, A) -> (B, K, embed_dim)
                    B, K, _ = actions.shape
                    act_embed = self.onehot_proj(
                        actions.reshape(B * K, A)).reshape(B, K, -1)
            else:
                # Use embedding matrix directly: e = p^T * E
                embed_w = self.embedding.weight  # (A, embed_dim)
                if actions.dim() == 2:
                    # (B, A) @ (A, embed_dim) -> (B, embed_dim)
                    act_embed = actions @ embed_w
                else:
                    # (B, K, A) -> (B*K, A) @ (A, embed_dim) -> (B*K, embed_dim) -> (B, K, embed_dim)
                    B, K, _ = actions.shape
                    act_embed = (actions.reshape(B * K, A) @
                                 embed_w).reshape(B, K, -1)

        else:
            # actions is integer indices: (B,) or (B, K)
            if actions.dim() not in (1, 2):
                raise ValueError(
                    f"Int actions must have dim 1 or 2, got shape {tuple(actions.shape)}"
                )

            if self.use_one_hot_actions:
                # Convert indices -> one-hot -> onehot_proj
                if actions.dim() == 1:
                    # (B,) -> (B, A) -> (B, embed_dim)
                    one_hot = torch.nn.functional.one_hot(
                        actions.long(), num_classes=A).float()
                    act_embed = self.onehot_proj(one_hot)
                else:
                    # (B, K) -> (B*K, A) -> (B, K, embed_dim)
                    B, K = actions.shape
                    one_hot = torch.nn.functional.one_hot(
                        actions.long().reshape(B * K), num_classes=A).float()
                    act_embed = self.onehot_proj(one_hot).reshape(B, K, -1)
            else:
                # Classic embedding lookup
                if actions.dim() == 1:
                    # (B,) -> (B, embed_dim)
                    act_embed = self.embedding(actions.long())
                else:
                    # (B, K) -> (B, K, embed_dim)
                    act_embed = self.embedding(actions.long())

        # ---- 3) fuse obs and action embeddings and produce trunk latent ----
        if act_embed.dim() == 2:
            # (B, embed_dim): concatenate with h (B, hidden_dim)
            x = torch.cat([h, act_embed], dim=-1)
            z = self.net(x)  # (B, feature_dim)
        else:
            # (B, K, embed_dim): repeat h across K
            B, K, D = act_embed.shape
            h_rep = h.unsqueeze(1).expand(
                B, K, h.shape[-1])  # (B, K, hidden_dim)
            # (B, K, hidden_dim+embed_dim)
            x = torch.cat([h_rep, act_embed], dim=-1)
            z = self.net(x.reshape(B * K, -1)).reshape(B,
                                                       K, -1)  # (B, K, feature_dim)

        return z
