from typing import Tuple
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64, max_action: float = 2.0, min_action: float = -2.0):
        super().__init__()
        self.max_action = max_action
        self.min_action = min_action

        # Policy network outputs mean and log_std (state-independent log_std for simplicity)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
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
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Concatenate observations and actions for distance computation
        x = torch.cat([obs, actions], dim=-1)
        return self.dist(x)
