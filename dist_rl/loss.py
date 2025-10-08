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

from dist_rl.utils import BetaEMA


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


def recursive_reward_aware_cosine_loss(
    embeddings: torch.Tensor,
    next_embeddings: torch.Tensor,
    dones: torch.Tensor,
    rewards: torch.Tensor,
    beta: float = None,
    gamma: float = 1.0,
    discount: float = 0.99,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:

    B = embeddings.size(0)
    # Normalize embeddings => cosine via dot product
    z = F.normalize(embeddings, p=2, dim=1, eps=eps)  # (B,d)
    z_next = F.normalize(next_embeddings, p=2, dim=1, eps=eps)  # (B,d)
    # (B,B) cosine similarities
    S = z @ z.T
    S_next = z_next @ z_next.T

    # Pairwise utility gaps and Δ via saturating exponential
    G = _pairwise_gaps(rewards)                     # (B,B)
    if beta is None:
        # Normalize Delta within the batch
        beta = G.max().item()
        Delta = G / (beta + eps)                     # (B,B)
        # print(f"Setting beta to max gap: {beta:.4f}")
    else:
        Delta = soft_window(G, 0.0, beta)       # (B,B)

    # Target cosine
    T = target_cosine_torch(Delta, gamma=gamma)       # (B,B)

    # Row-wise gating by dones: if row i terminal, don't bootstrap
    done_row = dones.view(B, 1).to(S.dtype)                     # (B,1)
    alive = 1.0 - done_row                                   # (B,1)
    targets = alive * (0.5 * (T + discount * S_next)) + \
        (1.0 - alive) * T  # (B,B)

    # targets = (T + discount * (1 - dones) * S_next)/ (2 - dones)

    # create a mask to exclude diagonal
    mask = torch.ones((B, B), dtype=torch.bool, device=S.device)
    mask.fill_diagonal_(False)
    diff = (S - targets)[mask]
    loss = (diff ** 2).mean()

    # print(f"Delta range: min {Delta[mask].min().item():.4f}, max {Delta[mask].max().item():.4f}")
    # print(f"Cosine range: min {S[mask].min().item():.4f}, max {S[mask].max().item():.4f}")

    info = {
        "beta": beta,
        "mean_gap": G[mask].mean().item(),
        "mean_delta": Delta[mask].mean().item(),
        "mean_targets": targets[mask].mean().item(),
        "mean_cos": S[mask].mean().item(),
    }
    return loss, info


def recursive_nstep_cosine_loss(
    embeddings: torch.Tensor,          # z(s,a)
    next_embeddings: torch.Tensor,     # z(s', π_targ(s'))
    dones: torch.Tensor,               # (B,)
    # (B,)  # <-- use n-step returns, not RTG
    nreturns: torch.Tensor,
    discount: float = 0.99,
    beta_ema: BetaEMA = None,
    # the n used in nreturns (for logging only)
    n: int = 20,
    gamma_shape: float = 1.0,          # your v_gamma: sharpness in t(Δ)=1-2Δ^γ
    lam: float = 0.5,                  # bootstrap mixing
    huber_delta: float = 0.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:

    B = embeddings.size(0)
    z = F.normalize(embeddings, p=2, dim=1)
    z_next = F.normalize(next_embeddings, p=2, dim=1)

    S = z @ z.T                        # (B,B)
    S_next = z_next @ z_next.T              # (B,B)

    # Pairwise future-aware gaps from n-step returns
    u = nreturns.view(-1, 1)
    G = (u - u.T).abs()                    # (B,B)

    # Robust scale β: 95th percentile per batch (avoids hand-tuning)
    # with torch.no_grad():
    #     beta = torch.quantile(G.reshape(-1), 0.95) + 1e-6
        
    with torch.no_grad():
        beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
        if beta_ema is not None:
            beta = embeddings.new_tensor(beta_ema.update(beta_batch))
        else:
            beta = beta_batch
            
    Delta = (G / beta).clamp(0., 1.)

    # Target cosine from gap
    T = 1.0 - 2.0 * (Delta ** float(gamma_shape))    # in [-1,1]

    # Bootstrap toward next similarities (mask terminals on the "row")
    alive = (1.0 - dones.view(-1, 1)).to(S.dtype)
    Y = (1.0 - lam) * T + lam * alive * (discount * S_next)

    # Exclude diagonal; use Huber for robustness
    mask = torch.ones((B, B), dtype=torch.bool, device=S.device)
    mask.fill_diagonal_(False)
    err = (S - Y)[mask]
    loss = F.smooth_l1_loss(err, torch.zeros_like(
        err), beta=huber_delta, reduction='mean')

    info = {
        "beta": float(beta),
        "mean_gap": float(G[mask].mean()),
        "mean_delta": float(Delta[mask].mean()),
        "mean_targets": float(Y[mask].mean()),
        "mean_cos": float(S[mask].mean()),
    }
    return loss, info


def recursive_nstep_twin_cosine_loss(
    embeddings: torch.Tensor,          # z(s,a)
    next_embeddings: torch.Tensor,     # z(s', π_targ(s'))
    dones: torch.Tensor,               # (B,)
    # (B,)  # <-- use n-step returns, not RTG
    nreturns: torch.Tensor,
    discount: float = 0.99,
    # the n used in nreturns (for logging only)
    n: int = 20,
    gamma_shape: float = 1.0,          # your v_gamma: sharpness in t(Δ)=1-2Δ^γ
    lam: float = 0.5,                  # bootstrap mixing
    huber_delta: float = 0.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    
    emb1, emb2 = embeddings
    next_emb1, next_emb2 = next_embeddings

    B = emb1.size(0)
    z1 = F.normalize(emb1, p=2, dim=1)
    z2 = F.normalize(emb2, p=2, dim=1)
    z_next1 = F.normalize(next_emb1, p=2, dim=1)
    z_next2 = F.normalize(next_emb2, p=2, dim=1)

    S1 = z1 @ z1.T                        # (B,B)
    S2 = z2 @ z2.T                        # (B,B)            
    
    S_next1 = z_next1 @ z_next1.T              # (B,B)
    S_next2 = z_next2 @ z_next2.T              # (B,B)
    
    # (B,B) - take the minimum similarity between the two critics
    S_next = torch.min(S_next1, S_next2)

    # Pairwise future-aware gaps from n-step returns
    u = nreturns.view(-1, 1)
    G = (u - u.T).abs()                    # (B,B)

    # Robust scale β: 95th percentile per batch (avoids hand-tuning)
    with torch.no_grad():
        beta = torch.quantile(G.reshape(-1), 0.95) + 1e-6
    Delta = (G / beta).clamp(0., 1.)

    # Target cosine from gap
    T = 1.0 - 2.0 * (Delta ** float(gamma_shape))    # in [-1,1]

    # Bootstrap toward next similarities (mask terminals on the "row")
    alive = (1.0 - dones.view(-1, 1)).to(S1.dtype)
    Y = (1.0 - lam) * T + lam * alive * (discount * S_next)
    # Y = (1 - Delta) + lam * alive * (discount * S_next - 1.0)

    # Exclude diagonal; use Huber for robustness
    mask = torch.ones((B, B), dtype=torch.bool, device=S1.device)
    mask.fill_diagonal_(False)
    err = (S1 - Y)[mask]
    err += (S2 - Y)[mask]
    loss = F.smooth_l1_loss(err, torch.zeros_like(
        err), beta=huber_delta, reduction='mean')

    info = {
        "beta": float(beta),
        "mean_gap": float(G[mask].mean()),
        "mean_delta": float(Delta[mask].mean()),
        "mean_targets": float(Y[mask].mean()),
        "mean_cos": float(S1[mask].mean()),
        "mean_cos2": float(S2[mask].mean()),
    }
    return loss, info

