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
            num_layers=1,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        assert obs_seq.shape[1] == self.seq_len and act_seq.shape[1] == self.seq_len, \
            f"expected seq_len={self.seq_len}"
        x = torch.cat([obs_seq, act_seq], dim=-1)  # (B,T,obs+act)
        out, _ = self.lstm(x)                      # (B,T,H)
        last = out[:, -1]                          # (B,H)
        v = self.head(last)                        # (B,1)
        return v


class ValueNetTransformer(nn.Module):
    """
    Transformer-encoder value network over the last `seq_len` (s,a) steps.

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
        self.embed = nn.Linear(in_dim, hidden_size)

        # learned positional embeddings for variable seq_len
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        nn.init.normal_(self.pos, std=0.02)

        # fixed, sensible defaults: 2 layers, 4 heads
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        assert obs_seq.shape[1] == self.seq_len and act_seq.shape[1] == self.seq_len, \
            f"expected seq_len={self.seq_len}"
        # If seq_len changed dynamically, resize positional embeddings
        if self.pos.shape[1] != self.seq_len:
            self.pos = nn.Parameter(self.pos.new_zeros(
                1, self.seq_len, self.hidden_size))
            nn.init.normal_(self.pos, std=0.02)

        x = torch.cat([obs_seq, act_seq], dim=-1)  # (B,T,obs+act)
        h = self.embed(x) + self.pos               # (B,T,H)
        h = self.encoder(h)                        # (B,T,H)

        # mean pooling (robust for value estimation)
        h = self.norm(h.mean(dim=1))               # (B,H)
        v = self.head(h)                           # (B,1)
        return v
