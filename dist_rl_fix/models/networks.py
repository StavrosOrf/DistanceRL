# dist_rl_fix/models/networks.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# -----------------------
# Shared building blocks
# -----------------------


class NoisyLinear(nn.Module):
    """
    Factorized NoisyNet layer (Fortunato et al., 2018).
    Use 'use_noisy=True' on Q heads for Rainbow-style exploration on Atari.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_w", torch.empty(out_features, in_features))
        self.register_buffer("eps_b", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.out_features))

    def _f(self, x):
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        self.eps_w.copy_(self._f(eps_out).outer(self._f(eps_in)))
        self.eps_b.copy_(self._f(eps_out))

    def forward(self, x):
        if self.training:
            self.sample_noise()
            w = self.weight_mu + self.weight_sigma * self.eps_w
            b = self.bias_mu + self.bias_sigma * self.eps_b
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class NatureCNN(nn.Module):
    """
    Classic DQN encoder:
      Conv(32,8x8,s4) -> Conv(64,4x4,s2) -> Conv(64,3x3,s1) -> Flatten -> Linear(512)
    Expects channel-first images [B, C, H, W] with values in [0,1].
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8,
                      stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4,
                      stride=2),          nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,
                      stride=1),          nn.ReLU(inplace=True),
        )
        # 84x84 input → 7x7 after convs; 64*7*7 = 3136
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [B, 512]


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


# -----------------------
# Image/discrete (Atari)
# -----------------------

class VisualDistanceTrunk(nn.Module):
    """
    z(s,a) for images + discrete actions.
    - Encode s with NatureCNN
    - Embed a with nn.Embedding
    - Fuse [phi(s), e(a)] via MLP → z(s,a)
    """

    def __init__(self, obs_channels: int, n_actions: int, out_dim: int = 256, hidden: int = 512, emb_dim: int = 128):
        super().__init__()
        self.encoder = NatureCNN(obs_channels)
        self.a_emb = nn.Embedding(n_actions, emb_dim)
        self.fuse = MLP(512 + emb_dim, out_dim, hidden=hidden, layers=2)

    def forward(self, obs_img: torch.Tensor, a_idx: torch.Tensor):
        # obs_img: [B,C,H,W]; a_idx: [B] (long)
        phi = self.encoder(obs_img)                 # [B, 512]
        ea = self.a_emb(a_idx.long())               # [B, emb_dim]
        return self.fuse(torch.cat([phi, ea], dim=-1))


class CategoricalActorVisual(nn.Module):
    """
    Categorical policy π(a|s) over discrete actions for images.
    Produces logits; sampling returns (a, logp_a).
    """

    def __init__(self, obs_channels: int, n_actions: int, use_noisy: bool = False):
        super().__init__()
        self.encoder = NatureCNN(obs_channels)
        linear = NoisyLinear if use_noisy else nn.Linear
        self.policy = nn.Sequential(
            linear(512, 512), nn.ReLU(inplace=True),
            linear(512, n_actions)
        )

    def forward(self, obs_img: torch.Tensor):
        h = self.encoder(obs_img)                   # [B, 512]
        logits = self.policy(h)                     # [B, A]
        return logits

    @torch.no_grad()
    def act_greedy(self, obs_img: torch.Tensor):
        logits = self.forward(obs_img)
        return torch.argmax(logits, dim=-1)         # [B]

    def sample(self, obs_img: torch.Tensor):
        logits = self.forward(obs_img)
        log_pi = torch.log_softmax(logits, dim=-1)  # [B, A]
        pi = log_pi.exp()
        a = torch.distributions.Categorical(probs=pi).sample()  # [B]
        # log prob of sampled actions
        logp_a = log_pi.gather(1, a.view(-1, 1))   # [B,1]
        return a, logp_a, logits


class DiscreteTwinQVisual(nn.Module):
    """
    Twin Q for images + discrete actions.
    Dueling heads (Rainbow ingredient) with optional NoisyLinear.
    Forward returns Q1(s, :) and Q2(s, :) => [B, A].
    """

    def __init__(self, obs_channels: int, n_actions: int, use_noisy: bool = True):
        super().__init__()
        self.encoder = NatureCNN(obs_channels)

        def dueling_head():
            linear = NoisyLinear if use_noisy else nn.Linear
            value = nn.Sequential(linear(512, 512), nn.ReLU(
                inplace=True), linear(512, 1))
            adv = nn.Sequential(linear(512, 512), nn.ReLU(
                inplace=True), linear(512, n_actions))
            return value, adv

        self.v1, self.a1 = dueling_head()
        self.v2, self.a2 = dueling_head()

    def _dueling(self, h, v_head, a_head):
        v = v_head(h)                # [B,1]
        a = a_head(h)                # [B,A]
        return v + a - a.mean(dim=1, keepdim=True)

    def forward(self, obs_img: torch.Tensor):
        h = self.encoder(obs_img)    # [B,512]
        q1 = self._dueling(h, self.v1, self.a1)
        q2 = self._dueling(h, self.v2, self.a2)
        return q1, q2
