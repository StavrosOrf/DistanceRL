"""Replay buffer tailored for Atari experiments."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


class AtariReplayBuffer:
    """Replay buffer that stores uint8 frames and serves stacked batches."""

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, int, int],
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.dtype = dtype

        self.obs_shape = observation_shape
        self.frames = observation_shape[0]

        self.obs = np.zeros((self.capacity,) + observation_shape, dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity,) + observation_shape, dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.idx] = np.asarray(obs * 255.0, dtype=np.uint8)
        self.next_obs[self.idx] = np.asarray(next_obs * 255.0, dtype=np.uint8)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        assert len(self) >= batch_size, "Not enough elements in replay buffer."

        max_idx = self.capacity if self.full else self.idx
        indices = np.random.randint(0, max_idx, size=batch_size)

        obs = torch.as_tensor(self.obs[indices], device=self.device, dtype=self.dtype) / 255.0
        next_obs = torch.as_tensor(self.next_obs[indices], device=self.device, dtype=self.dtype) / 255.0
        actions = torch.as_tensor(self.actions[indices], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device, dtype=self.dtype)
        dones = torch.as_tensor(self.dones[indices], device=self.device, dtype=self.dtype)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }


__all__ = ["AtariReplayBuffer"]
