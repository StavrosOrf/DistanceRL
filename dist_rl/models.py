from typing import Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64, max_action: float = 2.0, min_action: float = -2.0):
        super().__init__()
        self.max_action = max_action
        self.min_action = min_action

        # Policy network outputs mean and log_std (state-independent log_std for simplicity)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: action, log_prob, value, mean_action (deterministic)
        """
        return self.actor(obs) * self.max_action


class Distance(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.dist = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Concatenate observations and actions for distance computation
        x = torch.cat([obs, actions], dim=-1)
        return self.dist(x)
    
class DistanceTwin(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.dist = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.dist_2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Concatenate observations and actions for distance computation
        x = torch.cat([obs, actions], dim=-1)
        return self.dist(x), self.dist_2(x)
    
    def f_1(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Concatenate observations and actions for distance computation
        x = torch.cat([obs, actions], dim=-1)
        return self.dist(x)

LOG_STD_MIN = -20.0
LOG_STD_MAX =  2.0

class StochasticActor(nn.Module):
    """
    Unified stochastic policy:
    - Continuous (Box): tanh-Gaussian with log-prob correction (SAC-style).
    - Discrete: Categorical with Gumbel-Softmax straight-through for differentiable one-hot.
    
    API:
      forward(obs) -> (a_train, logp, a_mean_train)
        * Box:        a_train, a_mean_train are in env bounds (float)
        * Discrete:   a_train, a_mean_train are one-hot vectors (float, shape [B, nA])
      act(obs, deterministic=False) -> env_action
        * Box:        float vector in bounds
        * Discrete:   integer action ids (LongTensor or numpy)
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_size: int = 256,
                 max_action: Optional[float] = 1.0,
                 min_action: Optional[float] = -1.0,
                 action_space_type: str = "box",    # "box" or "discrete"
                 gumbel_tau: float = 1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden_size
        self.action_space_type = action_space_type.lower()
        self.gumbel_tau = gumbel_tau  # temperature for Gumbel-Softmax (training)

        if self.action_space_type not in ("box", "discrete"):
            raise ValueError("action_space_type must be 'box' or 'discrete'")

        # Common trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

        if self.action_space_type == "box":
            # Box (continuous) params
            self.max_action = float(max_action)
            self.min_action = float(min_action)
            self.mu_head      = nn.Linear(hidden_size, act_dim)
            self.log_std_head = nn.Linear(hidden_size, act_dim)

        else:
            # Discrete (categorical) params
            # logits over n actions
            self.logits_head = nn.Linear(hidden_size, act_dim)

    # ------------------ Box helpers ------------------
    def _box_params(self, h):
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std, log_std

    def _box_squash_scale(self, x):
        # tanh squash to [-1,1], then affine map to [min, max]
        y = torch.tanh(x)
        a = 0.5 * ((y + 1.0) * (self.max_action - self.min_action)) + self.min_action
        return a, y

    # ---------------- Discrete helpers ----------------
    @staticmethod
    def _one_hot(idx: torch.Tensor, n: int) -> torch.Tensor:
        # idx: [B] long
        oh = F.one_hot(idx.long(), num_classes=n).float()
        return oh

    def _gumbel_softmax_st(self, logits: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Straight-through Gumbel-Softmax:
          returns (one_hot_like_for_forward, soft_probs_for_backward)
        """
        # Sample with Gumbel-Softmax (relaxed)
        probs = F.softmax(logits, dim=-1)
        y_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        # Make a hard one-hot with straight-through estimator
        idx = torch.argmax(y_soft, dim=-1)
        y_hard = self._one_hot(idx, logits.size(-1))
        y = y_hard + (y_soft - y_soft.detach())
        # log-prob of chosen action (idx) under categorical
        logp = torch.log(probs.clamp_min(1e-8).gather(-1, idx.unsqueeze(-1)))
        return y, logp  # y is one-hot (ST), logp is [B,1]

    # ---------------- Unified API ----------------
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          a_train:      Box:   [B, act_dim] in bounds (float)
                        Discr: [B, act_dim] one-hot (float, ST-Gumbel)
          log_prob:     [B, 1]
          a_mean_train: Box:   deterministic action (in bounds)
                        Discr: argmax one-hot
        """
        h = self.trunk(obs)

        if self.action_space_type == "box":
            mu, std, log_std = self._box_params(h)
            eps = torch.randn_like(mu)
            pre_tanh = mu + std * eps

            a_train, y = self._box_squash_scale(pre_tanh)
            a_mean, y_mu = self._box_squash_scale(mu)

            # log prob with tanh correction
            log_prob_gauss = -0.5 * (((pre_tanh - mu) / (std + 1e-8))**2 + 2*log_std + math.log(2*math.pi))
            log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
            correction = torch.log(1 - y.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            logp = log_prob_gauss - correction
            return a_train, logp, a_mean

        else:
            logits = self.logits_head(h)  # [B, nA]
            # ST Gumbel-Softmax for differentiable one-hot
            a_train, logp = self._gumbel_softmax_st(logits, self.gumbel_tau)  # one-hot, [B,nA]
            # deterministic (argmax) one-hot for logging/mean
            idx = torch.argmax(logits, dim=-1)  # [B]
            a_mean = self._one_hot(idx, self.act_dim)  # [B,nA]
            return a_train, logp, a_mean

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Action for the environment:
          - Box: float vector in bounds.
          - Discrete: integer action ids (LongTensor).
        """
        h = self.trunk(obs)
        if self.action_space_type == "box":
            mu, std, _ = self._box_params(h)
            pre_tanh = mu if deterministic else (mu + std * torch.randn_like(mu))
            a, _ = self._box_squash_scale(pre_tanh)
            return a
        else:
            logits = self.logits_head(h)
            if deterministic:
                idx = torch.argmax(logits, dim=-1)  # [B]
            else:
                probs = F.softmax(logits, dim=-1)
                idx = torch.distributions.Categorical(probs=probs).sample()  # [B]
            return idx  # integer ids

class ValueNetLSTM(nn.Module):
    """
    LSTM-based value network over the last `seq_len` (s,a) steps.

    __init__(obs_dim: int, act_dim: int, hidden_size: int = 64, seq_len: int = 10)
    forward(obs_seq: (B,seq_len,obs_dim), act_seq: (B,seq_len,act_dim)) -> (B,1)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64, seq_len: int = 10):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        in_dim = obs_dim + act_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        # assert obs_seq.shape[1] == self.seq_len and act_seq.shape[1] == self.seq_len, \
        #     f"expected seq_len={self.seq_len}"

        x = torch.cat([obs_seq, act_seq], dim=-1)  # (B,T,obs+act)
        out, _ = self.lstm(x)                      # (B,T,H)
        last = out[:, -1]                          # (B,H)
        v = self.head(last)                        # (B,1)
        return v

