"""
Reward-aware cosine hinge (contrastive) loss.

Pairs with similar rewards are encouraged to have high cosine similarity.
Pairs with dissimilar rewards are encouraged to have cosine <= margin (repulsion).

Definition
----------
Let
  c(D1, D2) = <D1, D2> / (||D1|| * ||D2|| + eps)   in [-1, 1]
  a(v1, v2) = exp(-alpha * |v1 - v2|)              in (0, 1]

The loss for each pair is:
  L = a * ReLU(1 - c) + (1 - a) * ReLU(c - margin)

- alpha > 0 controls how quickly affinity decays w.r.t. |v1 - v2|
- margin <= 0 sets the target for dissimilar pairs (e.g., -0.2)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def reward_affinity(v1: torch.Tensor, v2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute soft reward affinity a(v1, v2) in (0, 1].

    Parameters
    ----------
    v1, v2 : (B,) tensors of scalars
    alpha  : positive float; larger => faster decay with |v1 - v2|

    Returns
    -------
    a : (B,) tensor in (0, 1]
    """
    return torch.exp(-alpha * (v1 - v2).abs()).clamp(0.0, 1.0)


def cosine_similarity(
    D1: torch.Tensor, D2: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Cosine similarity per pair.

    Parameters
    ----------
    D1, D2 : (B, d) tensors
    eps    : small constant for numerical stability

    Returns
    -------
    c : (B,) tensor in [-1, 1]
    """
    D1n = F.normalize(D1, p=2, dim=-1)
    D2n = F.normalize(D2, p=2, dim=-1)
    return (D1n * D2n).sum(dim=-1).clamp(-1.0, 1.0)


class RewardAwareCosineHingeLoss(nn.Module):
    """
    Reward-aware cosine hinge (contrastive) loss.

    Usage
    -----
    loss_fn = RewardAwareCosineHingeLoss(alpha=1.0, margin=-0.2)
    loss = loss_fn(D1, D2, v1, v2)
    """

    def __init__(self, alpha: float = 1.0, margin: float = -0.2, eps: float = 1e-8):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if margin > 0:
            raise ValueError("margin should be <= 0 for repulsion of dissimilar pairs")
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.eps = float(eps)

    def forward(
        self, D1: torch.Tensor, D2: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        D1, D2 : (B, d) embeddings (NN outputs)
        v1, v2 : (B,) reward scalars corresponding to each embedding

        Returns
        -------
        loss : scalar tensor
        """
        if D1.shape != D2.shape:
            raise ValueError(f"D1 and D2 must have same shape, got {D1.shape} vs {D2.shape}")
        if v1.shape != v2.shape or v1.ndim != 1 or v1.shape[0] != D1.shape[0]:
            raise ValueError("v1 and v2 must be shape (B,) and match batch of D1/D2")

        c = cosine_similarity(D1, D2, eps=self.eps)            # (B,)
        a = reward_affinity(v1, v2, alpha=self.alpha)          # (B,)

        # Positive term: when a≈1, push c→1
        pos = F.relu(1.0 - c)                                  # (B,)
        # Negative term: when a≈0, push c≤margin
        neg = F.relu(c - self.margin)                          # (B,)

        loss = (a * pos + (1.0 - a) * neg).mean()
        return loss
