import numpy as np
import torch
from typing import Tuple


class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, hidden_size_dim: int, device):
        self.obs = torch.zeros((buffer_size, obs_dim),
                               dtype=torch.float32, device=device)
        self.actions = torch.zeros(
            (buffer_size, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.entry_count = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, action, reward, done):
        self.ptr += 1
        self.entry_count += 1
        if self.ptr >= self.max_size:
            self.ptr = 0  # Overwrite when buffer is full

        self.obs[self.ptr] = torch.tensor(
            obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(
            action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor(
            reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(
            done, dtype=torch.float32, device=self.device)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(f"Buffer ptr: {self.ptr}, max size: {self.max_size}, batch size: {batch_size}, entry count: {self.entry_count}")
        if self.entry_count >= self.max_size:
            idxs = np.random.choice(
                self.max_size, size=batch_size, replace=False)
        else:
            idxs = np.random.choice(self.ptr, size=batch_size, replace=False)

        return (
            self.obs[idxs].detach(),
            self.actions[idxs].detach(),
            self.rewards[idxs].detach(),
            self.dones[idxs].detach(),
        )
