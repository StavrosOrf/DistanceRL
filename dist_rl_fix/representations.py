# dist_rl_fix/representations.py
from typing import Tuple
import torch
import torch.nn.functional as F

class BetaEMA:
    """EMA of the 95th percentile return gap for stable scaling."""
    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.val = None

    def update(self, beta_batch: torch.Tensor) -> float:
        b = float(beta_batch.item())
        self.val = b if self.val is None else self.decay * self.val + (1 - self.decay) * b
        return self.val

def recursive_nstep_cosine_loss_ema(
    embeddings: torch.Tensor,          # z(s,a)
    next_embeddings: torch.Tensor,     # z_targ(s', a'_targ)  (already stop-grad in caller)
    dones: torch.Tensor,               # (B,)
    nreturns: torch.Tensor,            # (B,)
    discount: float = 0.99,    
    gamma_shape: float = 1.0,
    lam: float = 0.5,
    huber_delta: float = 0.2,
    beta_ema: BetaEMA = None,
) -> Tuple[torch.Tensor, dict]:
    z = F.normalize(embeddings, p=2, dim=1)
    z_next = F.normalize(next_embeddings, p=2, dim=1)

    S = z @ z.T
    S_next = z_next @ z_next.T

    u = nreturns.view(-1, 1)
    G = (u - u.T).abs()
    with torch.no_grad():
        beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
        beta = embeddings.new_tensor(beta_ema.update(beta_batch) if beta_ema is not None else beta_batch)
    Delta = (G / beta).clamp(0., 1.)
    T = 1.0 - 2.0 * (Delta ** float(gamma_shape))

    alive = (1.0 - dones.view(-1, 1)).to(S.dtype)
    Y = (1.0 - lam) * T + lam * alive * (discount * S_next)

    mask = torch.ones_like(S, dtype=torch.bool)
    mask.fill_diagonal_(False)
    err = (S - Y)[mask]
    loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=huber_delta, reduction='mean')

    info = {
        "beta_batch": float(beta_batch),
        "beta_ema": float(beta),
        "mean_gap": float(G[mask].mean()),
        "mean_targets": float(Y[mask].mean()),
        "mean_cos": float(S[mask].mean()),
    }
    return loss, info


@torch.no_grad()
def _adv_weights(q: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Advantage-like weights within a state: center and softmax.
    q: (B, K) critic values per state for K actions.
    """
    q = q - q.mean(dim=1, keepdim=True)
    return torch.softmax(q / max(1e-6, temperature), dim=1)

def instate_advantage_rep_loss(
    z_anchor: torch.Tensor,   # (B, D) z(s, a_pos) for best actions per state
    z_all: torch.Tensor,      # (B, K, D) z(s, a_k) for K proposals per state
    q_all: torch.Tensor,      # (B, K, 1) Q(s, a_k)
    margin_scale: float = 0.5,
    temp: float = 1.0,
    huber_delta: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    In-state, advantage-aware metric learning.
    Encourage z(s, a_pos) to be closer to high-Q actions than to low-Q actions.
    Uses a soft hinge implemented via smooth L1 over (d_pos - d_neg + m(Δq))_+.

    z_anchor: z(s, a_pos) for per-state best action (by Q among K)
    z_all: embeddings for all K actions
    q_all: corresponding Q values
    """
    B, K, D = z_all.shape
    with torch.no_grad():
        # best actions per state
        q_all_flat = q_all.squeeze(-1)             # (B, K)
        idx_pos = torch.argmax(q_all_flat, dim=1)  # (B,)
    # gather positives and negatives
    z_pos = z_all[torch.arange(B, device=z_all.device), idx_pos]  # (B, D)

    # pairwise squared distances to all actions
    d_pos = (z_pos.unsqueeze(1) - z_all).pow(2).sum(dim=-1)  # (B, K)
    d_anchor = (z_anchor.unsqueeze(1) - z_all).pow(2).sum(dim=-1)  # optional: use provided anchor

    # advantage-based margins
    with torch.no_grad():
        w = _adv_weights(q_all_flat, temperature=temp)       # (B, K)
        q_pos = q_all_flat[torch.arange(B, device=q_all.device), idx_pos].unsqueeze(1) # (B,1)
        adv = (q_pos - q_all_flat)                           # (B, K) >= 0
        margin = margin_scale * F.softplus(adv)              # smooth margin

    # soft hinge via smooth L1 on (d_pos - d_k + margin)_+
    # prefer using d_pos vs ALL, weighted by advantage softmax
    diff = d_pos - d_anchor + margin  # (B,K)  (anchor can be z_pos or external)
    relu_diff = F.relu(diff)
    loss = F.smooth_l1_loss(relu_diff * w, torch.zeros_like(relu_diff), beta=huber_delta, reduction="mean")

    info = {
        "rep_instate/mean_margin": float(margin.mean().item()),
        "rep_instate/mean_dpos": float(d_pos.mean().item()),
        "rep_instate/mean_reludiff": float(relu_diff.mean().item()),
    }
    return loss, info


def pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    X: (B, N, D), Y: (B, M, D) -> (B, N, M) with ||x - y||^2
    """
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    X2 = (X * X).sum(dim=-1, keepdim=True)         # (B, N, 1)
    Y2 = (Y * Y).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, M)
    XY = X @ Y.transpose(1, 2)                     # (B, N, M)
    return (X2 + Y2 - 2.0 * XY).clamp_min_(0.0)

def sinkhorn_log(
    C: torch.Tensor,          # (B, N, M) cost matrix
    a: torch.Tensor,          # (B, N) source weights (sum=1)
    b: torch.Tensor,          # (B, M) target weights (sum=1)
    epsilon: float = 0.05,
    n_iters: int = 10,
):
    """
    Log-domain Sinkhorn iterations for stability.
    Returns log_u, log_v potentials and transport plan pi (optional).
    """
    B, N, M = C.shape
    log_a = torch.log(a + 1e-12)
    log_b = torch.log(b + 1e-12)
    K = torch.exp(-C / max(1e-12, epsilon))        # (B,N,M)

    # initialize log_u, log_v
    log_u = torch.zeros(B, N, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(B, M, device=C.device, dtype=C.dtype)

    for _ in range(n_iters):
        # log_u = log a - logsumexp( (-C/eps + log_v) over columns )
        log_u = log_a - torch.logsumexp((-C / epsilon + log_v.unsqueeze(1)).float(), dim=2).type_as(C)
        # log_v = log b - logsumexp( (-C/eps + log_u) over rows )
        log_v = log_b - torch.logsumexp((-C / epsilon + log_u.unsqueeze(2)).float(), dim=1).type_as(C)

    # pi = diag(u) K diag(v)  with u=exp(log_u), v=exp(log_v)
    u = torch.exp(log_u).unsqueeze(2)  # (B,N,1)
    v = torch.exp(log_v).unsqueeze(1)  # (B,1,M)
    pi = u * K * v                     # (B,N,M)
    return log_u, log_v, pi

def ot_cost(
    X: torch.Tensor, Y: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor,
    epsilon: float = 0.05, n_iters: int = 10,
):
    """
    Entropic-regularized OT cost: <pi, C> + epsilon * sum pi (log pi - 1)
    """
    C = pairwise_sq_dists(X, Y)                   # (B,N,M)
    _, _, pi = sinkhorn_log(C, a, b, epsilon, n_iters)
    reg = (pi * (pi.clamp_min(1e-12).log() - 1.0)).sum(dim=(1,2))  # (B,)
    cost = (pi * C).sum(dim=(1,2)) + epsilon * reg
    return cost  # (B,)

def sinkhorn_divergence(
    X: torch.Tensor, Y: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor,
    epsilon: float = 0.05, n_iters: int = 10,
):
    """
    S_eps(mu,nu) = OT_eps(mu,nu) - 0.5 OT_eps(mu,mu) - 0.5 OT_eps(nu,nu)
    All terms computed with entropic regularization (same eps).
    """
    cost_xy = ot_cost(X, Y, a, b, epsilon, n_iters)                 # (B,)
    cost_xx = ot_cost(X, X, a, a, epsilon, n_iters)                 # (B,)
    cost_yy = ot_cost(Y, Y, b, b, epsilon, n_iters)                 # (B,)
    return cost_xy - 0.5 * (cost_xx + cost_yy)                      # (B,)
