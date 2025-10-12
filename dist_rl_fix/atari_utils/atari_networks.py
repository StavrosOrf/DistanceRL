from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Fallback NoisyLinear (in case your project doesn't already provide one) ----
class NoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyNet layer (Fortunato et al., 2018).
    """
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_in", torch.empty(in_features))
        self.register_buffer("eps_out", torch.empty(out_features))
        self.reset_parameters(sigma0)

    def reset_parameters(self, sigma0: float):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma0 / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(sigma0 / math.sqrt(self.out_features))

    def forward(self, x):
        def f(x): return torch.sign(x) * torch.sqrt(torch.abs(x))
        self.eps_in = self.eps_in.to(x.device)
        self.eps_out = self.eps_out.to(x.device)
        self.eps_in.normal_()
        self.eps_out.normal_()
        eps_w = f(self.eps_out).unsqueeze(1) * f(self.eps_in).unsqueeze(0)
        eps_b = f(self.eps_out)
        w = self.weight_mu + self.weight_sigma * eps_w
        b = self.bias_mu + self.bias_sigma * eps_b
        return F.linear(x, w, b)

# ---- Nature CNN (Mnih et al. 2015) ----
class NatureCNN(nn.Module):
    """
    Input: [B, C, 84, 84] in [0,1]
    Output: [B, 512]
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(self.conv(x))

# ---- IMPALA CNN (Espeholt et al. 2018) ----
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1(x), inplace=True)
        h = self.conv2(h)
        return F.relu(x + h, inplace=True)

class ImpalaCNN(nn.Module):
    """
    [Conv -> Residual -> Residual -> Pool] x 3, then FC(512).
    Input: [B,C,84,84] float in [0,1]. Output: [B,512]
    """
    def __init__(self, in_channels: int, channels=(16, 32, 32)):
        super().__init__()
        blocks = []
        c_in = in_channels
        for c in channels:
            blocks += [nn.Conv2d(c_in, c, 3, stride=1, padding=1), nn.ReLU(inplace=True),
                       ResidualBlock(c), ResidualBlock(c),
                       nn.MaxPool2d(3, stride=2, padding=1)]
            c_in = c
        self.conv_tower = nn.Sequential(*blocks)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(11*11*channels[-1], 512), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_tower(x)
        return self.fc(x)

# ---- Heads & distance trunks (Nature + IMPALA variants) ----
class VisualDistanceTrunkNature(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, out_dim: int = 256, hidden: int = 512, emb_dim: int = 128):
        super().__init__()
        self.encoder = NatureCNN(obs_channels)
        self.a_emb = nn.Embedding(n_actions, emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(512 + emb_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, obs_img: torch.Tensor, a_idx: torch.Tensor):
        phi = self.encoder(obs_img)
        ea = self.a_emb(a_idx.long())
        return self.fuse(torch.cat([phi, ea], dim=-1))

class VisualDistanceTrunkIMPALA(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, out_dim: int = 256, hidden: int = 512, emb_dim: int = 128):
        super().__init__()
        self.encoder = ImpalaCNN(obs_channels)
        self.a_emb = nn.Embedding(n_actions, emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(512 + emb_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, obs_img: torch.Tensor, a_idx: torch.Tensor):
        phi = self.encoder(obs_img)
        ea = self.a_emb(a_idx.long())
        return self.fuse(torch.cat([phi, ea], dim=-1))

class CategoricalActorVisualNature(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, use_noisy: bool = False):
        super().__init__()
        self.encoder = NatureCNN(obs_channels)
        linear = NoisyLinear if use_noisy else nn.Linear
        self.policy = nn.Sequential(linear(512, 512), nn.ReLU(inplace=True), linear(512, n_actions))

    def forward(self, obs_img: torch.Tensor):
        return self.policy(self.encoder(obs_img))

class CategoricalActorVisualIMPALA(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, use_noisy: bool = False):
        super().__init__()
        self.encoder = ImpalaCNN(obs_channels)
        linear = NoisyLinear if use_noisy else nn.Linear
        self.policy = nn.Sequential(linear(512, 512), nn.ReLU(inplace=True), linear(512, n_actions))

    def forward(self, obs_img: torch.Tensor):
        return self.policy(self.encoder(obs_img))

class DiscreteTwinQVisualNature(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, use_noisy: bool = True):
        super().__init__()
        self.encoder = NatureCNN(obs_channels)
        def dueling_head():
            linear = NoisyLinear if use_noisy else nn.Linear
            value = nn.Sequential(linear(512, 512), nn.ReLU(inplace=True), linear(512, 1))
            adv = nn.Sequential(linear(512, 512), nn.ReLU(inplace=True), linear(512, n_actions))
            return value, adv
        self.v1, self.a1 = dueling_head()
        self.v2, self.a2 = dueling_head()

    def _dueling(self, h, v_head, a_head):
        v = v_head(h)
        a = a_head(h)
        return v + a - a.mean(dim=1, keepdim=True)

    def forward(self, obs_img: torch.Tensor):
        h = self.encoder(obs_img)
        q1 = self._dueling(h, self.v1, self.a1)
        q2 = self._dueling(h, self.v2, self.a2)
        return q1, q2

class DiscreteTwinQVisualIMPALA(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, use_noisy: bool = True):
        super().__init__()
        self.encoder = ImpalaCNN(obs_channels)
        def dueling_head():
            linear = NoisyLinear if use_noisy else nn.Linear
            value = nn.Sequential(linear(512, 512), nn.ReLU(inplace=True), linear(512, 1))
            adv = nn.Sequential(linear(512, 512), nn.ReLU(inplace=True), linear(512, n_actions))
            return value, adv
        self.v1, self.a1 = dueling_head()
        self.v2, self.a2 = dueling_head()

    def _dueling(self, h, v_head, a_head):
        v = v_head(h)
        a = a_head(h)
        return v + a - a.mean(dim=1, keepdim=True)

    def forward(self, obs_img: torch.Tensor):
        h = self.encoder(obs_img)
        q1 = self._dueling(h, self.v1, self.a1)
        q2 = self._dueling(h, self.v2, self.a2)
        return q1, q2