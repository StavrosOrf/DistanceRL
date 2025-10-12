from __future__ import annotations
import numpy as np
import torch

class ReplayBufferAtari:
    """
    Memory‑efficient uint8 replay buffer for stacked Atari frames.
    Stores obs/next_obs as uint8 (C,H,W) and converts to float32 [0,1] on sample.
    """
    def __init__(self, capacity: int, obs_shape: tuple[int, int, int], device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        C, H, W = obs_shape
        self.obs = np.zeros((self.capacity, C, H, W), dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity, C, H, W), dtype=np.uint8)
        self.acts = np.zeros((self.capacity,), dtype=np.int64)
        self.rews = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs: np.ndarray, next_obs: np.ndarray, act: int, rew: float, done: bool):
        self.obs[self.idx] = (obs * 255.0).clip(0, 255).astype(np.uint8) if obs.dtype != np.uint8 else obs
        self.next_obs[self.idx] = (next_obs * 255.0).clip(0, 255).astype(np.uint8) if next_obs.dtype != np.uint8 else next_obs
        self.acts[self.idx] = int(act)
        self.rews[self.idx] = float(rew)
        self.dones[self.idx] = bool(done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        n = len(self)
        assert n >= batch_size, "Not enough samples"
        idxs = np.random.randint(0, n, size=batch_size)
        obs = torch.as_tensor(self.obs[idxs], device=self.device, dtype=torch.float32) / 255.0
        next_obs = torch.as_tensor(self.next_obs[idxs], device=self.device, dtype=torch.float32) / 255.0
        acts = torch.as_tensor(self.acts[idxs], device=self.device, dtype=torch.long)
        rews = torch.as_tensor(self.rews[idxs], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(self.dones[idxs], device=self.device, dtype=torch.float32)
        return obs, next_obs, acts, rews, dones