import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# -----------------------
# Vector/continuous path
# -----------------------

class MLP(nn.Module):
    """Plain MLP (no LayerNorm)."""

    def __init__(self, in_dim, out_dim, hidden=256, layers=2, act=nn.ReLU):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), act()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class DistanceTrunk(nn.Module):
    """
    State-conditional action embedding z(s,a) \in R^H for vector observations & continuous actions.
    Used for value-aware metric; NOT shared with critics.
    """

    def __init__(self, obs_dim, act_dim, hidden=256, out_dim=256, layers=3):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, out_dim,
                       hidden=hidden, layers=layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class GaussianActor(nn.Module):
    """Tanh-Gaussian policy for continuous action spaces."""

    def __init__(self, obs_dim, act_dim, hidden=256, log_std_bounds=(-5, 1)):
        super().__init__()
        self.net = MLP(obs_dim, hidden, hidden=hidden, layers=2)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        low, high = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = low + 0.5 * (high - low) * (log_std + 1.0)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        action = torch.tanh(pre_tanh)
        # log prob with tanh correction
        logp = (-0.5 * ((pre_tanh - mu) / std).pow(2) - torch.log(std) -
                0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        logp -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, logp, torch.tanh(mu)

    @torch.no_grad()
    def _atanh(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Stable inverse tanh for inputs in (-1, 1). Clamps to avoid inf.
        atanh(x) = 0.5 * (log1p(x) - log1p(-x))
        """
        x = x.clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _gaussian_params(self, obs: torch.Tensor):
        """
        Utility to get mean and std from your network's forward.
        Assumes forward() -> (mu, log_std) with shape (B, act_dim).
        Clamp log_std if you do that in sample().
        """
        mu, log_std = self.forward(obs)
        # keep in sync with your sample(): same clamps
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        std = log_std.exp()
        return mu, std

    def log_prob(self, obs: torch.Tensor, a_squash: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Log πθ(a|s) for a tanh-squashed Gaussian policy.
        Inputs:
          obs:      (B, obs_dim)
          a_squash: (B, act_dim), each component in [-1, 1] (what your actor outputs after tanh)
        Returns:
          logp:     (B,) tensor of log-probabilities under πθ(·|obs)
        Notes:
          log π(a) = log N(u; μ, σ^2) - Σ log(1 - tanh(u)^2)  with  a = tanh(u).
          We compute u = atanh(a) stably and apply the Jacobian correction.
        """
        mu, std = self._gaussian_params(obs)          # (B, A) each
        # inverse squash (stable)
        a = a_squash.clamp(-1 + eps, 1 - eps)
        u = self._atanh(a, eps=eps)                   # (B, A)

        # Gaussian log-prob of pre-tanh action u
        dist = Normal(mu, std)
        logp_u = dist.log_prob(u).sum(dim=-1)         # (B,)

        # change-of-variables correction: sum log(1 - tanh(u)^2) = sum log(1 - a^2)
        # subtract because p(a) = p(u) * |det du/da| and du/da = 1/(1 - a^2)
        log_det = torch.log(1 - a * a + eps).sum(dim=-1)  # (B,)
        logp = logp_u - log_det
        return logp


class TwinQ(nn.Module):
    """Twin critics for vector obs + continuous actions: Q(s,a) each as MLP."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden=hidden, layers=2)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden=hidden, layers=2)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)
