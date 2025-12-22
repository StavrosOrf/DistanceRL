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

    S = z @ z.T # cosine similarities
    S_next = z_next @ z_next.T # cosine similarities

    u = nreturns.view(-1, 1) 
    G = (u - u.T).abs() # utility gap matrix    
    with torch.no_grad():
        beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
        beta = embeddings.new_tensor(beta_ema.update(beta_batch) if beta_ema is not None else beta_batch)
    Delta = (G / beta).clamp(0., 1.)
    T = 1.0 - 2.0 * (Delta ** float(gamma_shape)) # shaped targets in [-1, 1]

    alive = (1.0 - dones.view(-1, 1)).to(S.dtype)
    Y = (1.0 - lam) * T + lam * alive * (discount * S_next) # (B, B) target matrix

    mask = torch.ones_like(S, dtype=torch.bool)
    mask.fill_diagonal_(False)
    err = (S - Y)[mask] # exclude diagonal terms
    loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=huber_delta, reduction='mean')

    info = {
        "beta_batch": float(beta_batch),
        "beta_ema": float(beta),
        "mean_gap": float(G[mask].mean()),
        "mean_targets": float(Y[mask].mean()),
        "mean_cos": float(S[mask].mean()),
    }
    return loss, info
