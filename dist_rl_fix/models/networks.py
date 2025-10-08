from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, layers=2, ln=False, act=nn.ReLU):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden)]
            if ln: mods += [nn.LayerNorm(hidden)]
            mods += [act()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x)

class DistanceTrunk(nn.Module):
    """Shared trunk z(s,a) used by critics and the representation loss."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = MLP(obs_dim + act_dim, hidden, hidden=hidden, layers=3, ln=False)
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        z = self.trunk(x)
        return z

class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy with state encoder."""
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_bounds=(-5, 2)):
        super().__init__()
        self.net = MLP(obs_dim, hidden, hidden=hidden, layers=2, ln=False)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        low, high = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = low + 0.5*(high - low)*(log_std + 1)  # scale to [low, high]
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        action = torch.tanh(pre_tanh)
        # Logprob with tanh correction
        log_prob = (-0.5*((pre_tanh - mu)/std).pow(2) - torch.log(std) - 0.5*math.log(2*math.pi)).sum(-1, keepdim=True)
        # Tanh correction (sum over dims)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)  # action, logprob, deterministic

class TwinQ(nn.Module):
    """Twin Q heads that read the shared distance trunk features."""
    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(feat_dim, 1, hidden=hidden, layers=2, ln=False)
        self.q2 = MLP(feat_dim, 1, hidden=hidden, layers=2, ln=False)
    def forward(self, feat):
        return self.q1(feat), self.q2(feat)
