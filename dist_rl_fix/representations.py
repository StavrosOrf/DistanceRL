from typing import Tuple
import torch
import torch.nn.functional as F

class BetaEMA:
    """EMA of the 95th percentile gap to stabilize scaling across batches."""
    def __init__(self, decay: float = 0.99, eps: float = 1e-6):
        self.decay = decay
        self.eps = eps
        self.val = None
    def update(self, beta_batch: torch.Tensor):
        b = float(beta_batch.item())
        if self.val is None:
            self.val = b
        else:
            self.val = self.decay * self.val + (1 - self.decay) * b
        return self.val

def recursive_nstep_cosine_loss_ema(
    embeddings: torch.Tensor,
    next_embeddings: torch.Tensor,
    dones: torch.Tensor,
    nreturns: torch.Tensor,
    discount: float = 0.99,
    n: int = 20,
    gamma_shape: float = 1.0,
    lam: float = 0.5,
    huber_delta: float = 0.2,
    beta_ema: BetaEMA = None,
) -> Tuple[torch.Tensor, dict]:
    z = F.normalize(embeddings, p=2, dim=1)
    z_next = F.normalize(next_embeddings, p=2, dim=1)

    S = z @ z.T
    S_next = z_next @ z_next.T

    u = nreturns.view(-1,1)
    G = (u - u.T).abs()
    with torch.no_grad():
        beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
        if beta_ema is not None:
            beta = embeddings.new_tensor(beta_ema.update(beta_batch))
        else:
            beta = beta_batch
    Delta = (G / beta).clamp(0., 1.)
    T = 1.0 - 2.0 * (Delta ** float(gamma_shape))

    alive = (1.0 - dones.view(-1,1)).to(S.dtype)
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
