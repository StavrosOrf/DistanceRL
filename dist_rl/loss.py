"""
Reward-aware cosine loss with saturating-exponential utility gap mapping.

Gap mapping (you set beta):
    g = |u_i - u_j| >= 0
    Δ = 1 - exp(-g / beta)  in (0, 1), with Δ(0)=0 and Δ→1 as g→∞

Target cosine (maps Δ ∈ [0,1] to [-1, 1]):
    t(Δ; γ) = 1 - 2 * Δ^γ
    - γ = 1.0 is linear; γ > 1 pushes faster toward -1 for moderate gaps.
    - γ < 1 keeps higher targets for small/mid gaps.

Loss (regression to target cosine):
    L = (s - t(Δ; γ))^2
where s is the cosine similarity between L2-normalized embeddings.

This file provides:
- NumPy utilities for Δ and t
- PyTorch loss for a batch: all-pairs cosine vs target, excluding the diagonal
"""

from typing import Tuple
import numpy as np

# --------------------------- PyTorch implementation ---------------------------

import torch
import torch.nn.functional as F


def _pairwise_gaps(u: torch.Tensor) -> torch.Tensor:
    """
    Pairwise absolute gaps |u_i - u_j| for u shape (B,).
    """
    u = u.view(-1, 1)         # (B,1)
    return (u - u.T).abs()    # (B,B)


def target_cosine_torch(delta: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    t(Δ; γ) = 1 - 2 * Δ^γ, with Δ ∈ [0,1].
    """
    delta = delta.clamp(0.0, 1.0)
    return 1.0 - 2.0 * (delta ** float(gamma))


def soft_window(x, a, b, eps=1e-2):
    m = 0.5*(a + b)
    w = 0.5*(b - a)

    # make torch
    eps = torch.tensor(eps).to(x.device)
    m = torch.tensor(m).to(x.device)
    w = torch.tensor(w).to(x.device)

    k = torch.log((1 - eps) / eps)
    return 1.0 / (1.0 + torch.exp(-k * ((x - m) / w)))


def reward_aware_cosine_loss_exp(
    embeddings: torch.Tensor,
    utilities: torch.Tensor,
    beta: float = None,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the regression-only loss:
        L = mean_{i != j} ( s_ij - t(Δ_ij; γ) )^2
    where:
        s_ij = cosine( z_i, z_j ) with z = L2-normalized embeddings
        Δ_ij = 1 - exp( -|u_i - u_j| / beta )

    Args:
        embeddings: (B, d) unnormalized features from the network
        utilities:  (B,)   real-valued utility per sample
        beta:       > 0    scale of the saturating exponential mapping
        gamma:      > 0    shape of target t(Δ; γ)
        eps:              small constant for numerical stability

    Returns:
        loss: scalar torch.Tensor
        info: dict with optional diagnostics
    """
    # Normalize embeddings => cosine via dot product
    z = F.normalize(embeddings, p=2, dim=1, eps=eps)  # (B,d)
    # (B,B) cosine similarities
    S = z @ z.T   

    # Pairwise utility gaps and Δ via saturating exponential
    G = _pairwise_gaps(utilities)                     # (B,B)

    if beta is None:
        # Normalize Delta within the batch
        beta = G.max().item()
        Delta = G / (beta + eps)                     # (B,B)
        # print(f"Setting beta to max gap: {beta:.4f}")
    else:
        Delta = soft_window(G, 0.0, beta)       # (B,B)

    # Target cosine
    T = target_cosine_torch(Delta, gamma=gamma)       # (B,B)

    # Exclude diagonal (i == j)
    Bsz = S.size(0)
    mask = torch.ones((Bsz, Bsz), dtype=torch.bool, device=S.device)
    mask.fill_diagonal_(False)
    # do not include the upper triangle
    mask = mask & torch.tril(torch.ones(
        (Bsz, Bsz), dtype=torch.bool, device=S.device), diagonal=-1)

    diff = (S - T)[mask]
    loss = (diff ** 2).mean()

    # print(f"Delta range: min {Delta[mask].min().item():.4f}, max {Delta[mask].max().item():.4f}")
    # print(f"Cosine range: min {S[mask].min().item():.4f}, max {S[mask].max().item():.4f}")

    info = {
        "beta": beta,
        "mean_gap": G[mask].mean().item(),
        "mean_delta": Delta[mask].mean().item(),
        "mean_cos": S[mask].mean().item(),
    }
    return loss, info

# --------------------------- Minimal usage example ---------------------------


if __name__ == "__main__":
    torch.manual_seed(0)

    B, d = 4, 32

    for _ in range(10):
        embeddings = torch.randn(B, d)              # your model's outputs
        # any real-valued utilities
        utilities = torch.randn(B) * 200.0 - 100
        print(f"-------------------------------------\n")
        print(
            f"utilities range: min {utilities.min().item():.2f}, max {utilities.max().item():.2f}")

        # Choose beta (scale of Δ mapping) and gamma (shape of target)
        beta = 1.0      # try median(|u_i - u_j|) or set manually
        gamma = 1.2     # >1 pushes faster toward -1 as Δ grows

        loss, info = reward_aware_cosine_loss_exp(
            embeddings=embeddings,
            utilities=utilities,
            beta=beta,
            gamma=gamma,
        )
        print(f"loss = {loss.item():.6f}")
        print("diagnostics:", {k: round(v, 4) for k, v in info.items()})
