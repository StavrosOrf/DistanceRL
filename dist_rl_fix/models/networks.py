# dist_rl_fix/models/networks.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Representation trunk z(s,a) used for rep-loss / kernel (NOT for TD critics)."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = MLP(obs_dim + act_dim, hidden, hidden=hidden, layers=3)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.trunk(x)
    
class DistanceTrunkWs(nn.Module):
    """
    State-conditional action embedding z(s,a) \in R^H.
    Used for value-aware metric and OT transport; NOT used by critics.
    """
    def __init__(self, obs_dim, act_dim, hidden=256, out_dim=256, layers=3):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, out_dim, hidden=hidden, layers=layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

class GaussianActor(nn.Module):
    """Tanh-Gaussian policy."""
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
        logp = (-0.5 * ((pre_tanh - mu) / std).pow(2) - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        logp -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, logp, torch.tanh(mu)

class TwinQ(nn.Module):
    """
    Twin critics with separate encoders (no sharing with rep trunk).
    Input: (obs, act), both in normalized / [-1,1] spaces.
    """
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden=hidden, layers=2)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden=hidden, layers=2)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)
