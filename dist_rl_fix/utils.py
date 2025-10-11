
import torch

class RunningMeanStd:
    def __init__(self, shape, eps=1e-4, device='cpu'):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        new_var = M2 / tot

        self.mean, self.var, self.count = new_mean, new_var, tot

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

def polyak_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

class RMSNormalizer:
    """Running RMS tracker for scalar loss magnitudes."""
    def __init__(self, momentum: float = 0.99, eps: float = 1e-8, init: float = 1.0):
        self.momentum = momentum
        self.eps = eps
        self.rms2 = init * init

    def update(self, x: torch.Tensor) -> float:
        # x: scalar tensor (detached)
        v = float(x.detach().cpu().item())
        self.rms2 = self.momentum * self.rms2 + (1 - self.momentum) * (v * v)
        return (self.rms2 ** 0.5) + self.eps

def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is None: 
            continue
        total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5

def zero_grads(parameters):
    for p in parameters:
        if p.grad is not None:
            p.grad.zero_()

def pcgrad(primary_params, g_main, g_aux):
    """
    Project auxiliary gradient g_aux to not conflict with g_main:
    g_aux <- g_aux - proj_{g_main}(g_aux) if dot<0
    Returns a fused gradient tensor per parameter identical in structure to params.
    """
    # flatten helpers
    def _flat(grads):
        return torch.cat([g.reshape(-1) for g in grads if g is not None])
    gm = _flat(g_main)
    ga = _flat(g_aux)
    dot = torch.dot(gm, ga)
    if dot < 0:
        proj = (dot / (gm.norm()**2 + 1e-12)) * gm
        ga = ga - proj
    # rebuild into per-parameter gradients
    fused = []
    offset = 0
    for p, gm_i, ga_i in zip(primary_params, g_main, g_aux):
        if p.grad is None:
            fused.append(None)
            continue
        n = p.numel()
        fused.append(ga[offset:offset+n].view_as(p))
        offset += n
    return fused, float(dot.detach().cpu().item())

def capture_grads(params):
    """Return a list of cloned grads; if None, zeros of same shape."""
    gs = []
    for p in params:
        if p.grad is None:
            gs.append(torch.zeros_like(p))
        else:
            gs.append(p.grad.detach().clone())
    return gs

def set_grads(params, grads, add=False, scale: float = 1.0):
    for p, g in zip(params, grads):
        if g is None:
            continue
        if add:
            if p.grad is None:
                p.grad = g * scale
            else:
                p.grad.add_(g * scale)
        else:
            if p.grad is None:
                p.grad = g * scale
            else:
                p.grad.copy_(g * scale)

@torch.no_grad()
def approx_spearman_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Approximate Spearman correlation between x and y on the last dimension.
    x, y: (..., N)
    """
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    # rank via argsort twice
    rx = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
    ry = torch.argsort(torch.argsort(y, dim=-1), dim=-1).float()
    rx = (rx - rx.mean(dim=-1, keepdim=True)) / (rx.std(dim=-1, keepdim=True) + 1e-6)
    ry = (ry - ry.mean(dim=-1, keepdim=True)) / (ry.std(dim=-1, keepdim=True) + 1e-6)
    r = (rx * ry).mean().item()
    return float(r)


import torch
import torch.nn.functional as F

# ---------- geometry / cost ----------

def pairwise_cosine_cost(Zp: torch.Tensor, Zc: torch.Tensor) -> torch.Tensor:
    """
    Cosine cost C = 1 - cos(zp, zc).
    Zp: (B, N, D)  policy embeddings
    Zc: (B, K, D)  candidate embeddings
    returns: (B, N, K)
    """
    # cosine = <zp, zc> assuming L2-normalized inputs
    cos = torch.einsum('bnd,bkd->bnk', Zp, Zc).clamp(-1.0, 1.0)
    return 1.0 - cos


def sinkhorn_transport_cost(C: torch.Tensor,
                            a: torch.Tensor,
                            b: torch.Tensor,
                            epsilon: float = 0.05,
                            n_iters: int = 20) -> torch.Tensor:
    """
    Entropic OT cost <P, C>, with stable log-domain Sinkhorn.
    C: (B, N, K), a: (B, N), b: (B, K)
    """
    # safety: normalize a,b to sum=1 per batch
    a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
    b = b / (b.sum(dim=1, keepdim=True) + 1e-8)

    log_a = torch.log(a + 1e-8)                 # (B,N)
    log_b = torch.log(b + 1e-8)                 # (B,K)
    logK  = -C / max(epsilon, 1e-8)             # (B,N,K)

    # dual potentials
    u = torch.zeros_like(log_a)                 # (B,N)
    v = torch.zeros_like(log_b)                 # (B,K)

    for _ in range(n_iters):
        # u update: sum over K
        u = log_a - torch.logsumexp(logK + v.unsqueeze(1), dim=2)               # (B,N)
        # v update: sum over N  (NOTE: unsqueeze(1), not unsqueeze(2))
        v = log_b - torch.logsumexp(logK.transpose(1, 2) + u.unsqueeze(1), dim=2)  # (B,K)

    logP = u.unsqueeze(2) + logK + v.unsqueeze(1)   # (B,N,K)
    P    = torch.exp(logP)
    cost = (P * C).sum(dim=(1, 2))
    return cost


def sinkhorn_plan(C: torch.Tensor,
                  a: torch.Tensor,
                  b: torch.Tensor,
                  epsilon: float = 0.05,
                  n_iters: int = 20):
    """
    Returns transport plan P and cost.
    C: (B,N,K), a: (B,N), b: (B,K)
    """
    a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
    b = b / (b.sum(dim=1, keepdim=True) + 1e-8)

    log_a = torch.log(a + 1e-8)
    log_b = torch.log(b + 1e-8)
    logK  = -C / max(epsilon, 1e-8)

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(n_iters):
        u = log_a - torch.logsumexp(logK + v.unsqueeze(1), dim=2)                   # (B,N)
        v = log_b - torch.logsumexp(logK.transpose(1, 2) + u.unsqueeze(1), dim=2)  # (B,K)

    logP = u.unsqueeze(2) + logK + v.unsqueeze(1)   # (B,N,K)
    P    = torch.exp(logP)
    cost = (P * C).sum(dim=(1, 2))
    return P, cost


# ---------- soft-KNN state-conditioned selection (optional helper) ----------

def soft_knn_indices(z_anchor: torch.Tensor,
                     zC_full: torch.Tensor,
                     knn: int,
                     tau_sim: float) -> torch.Tensor:
    """
    z_anchor: (B, D)     per-state anchor (e.g., mean policy embedding)
    zC_full: (Kf, D)     full candidate pool embeddings (L2-normalized)
    returns: indices (B, knn) of candidates with highest soft-KNN mass
    """
    S = (z_anchor @ zC_full.T) / max(tau_sim, 1e-6)    # (B, Kf), temperature-scaled cosine
    # pick top-k by softmass (equivalent to top-k cosine since softmax is monotone)
    idx = torch.topk(S, k=min(knn, zC_full.size(0)), dim=1).indices
    return idx
