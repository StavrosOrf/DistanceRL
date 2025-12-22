import torch
import numpy as np
from typing import Tuple

class RunningMeanStd:
    def __init__(self, shape, eps=1e-4, device='cpu'):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = eps
        self.device = device

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        new_var = M2 / tot

        self.mean, self.var, self.count = new_mean, new_var, tot

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def state_dict(self):
        return {
            'mean': self.mean.detach().cpu(),
            'var': self.var.detach().cpu(),
            'count': float(self.count),
        }

    def load_state_dict(self, state):
        device = self.mean.device
        self.mean = state['mean'].to(device)
        self.var = state['var'].to(device)
        self.count = state['count']

def polyak_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device):
        self.obs = torch.zeros((buffer_size, obs_dim),
                               dtype=torch.float32, device=device)
        self.next_obs = torch.zeros(
            (buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(
            (buffer_size, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.ptr = -1
        self.entry_count = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, next_obs, action, reward, done):
        self.ptr += 1
        self.entry_count += 1
        if self.ptr >= self.max_size:
            self.ptr = 0  # Overwrite when buffer is full

        self.obs[self.ptr] = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device)
        self.next_obs[self.ptr] = torch.as_tensor(
            next_obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(
            reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(
            done, dtype=torch.float32, device=self.device)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:        
        if self.entry_count >= self.max_size:
            idxs = np.random.choice(
                self.max_size, size=batch_size, replace=False)
        else:
            idxs = np.random.choice(self.ptr, size=batch_size, replace=False)

        return (
            self.obs[idxs].detach(),
            self.next_obs[idxs].detach(),
            self.actions[idxs].detach(),
            self.rewards[idxs].detach(),
            self.dones[idxs].detach(),
        )

    # alias for compatibility
    def sample(self, batch_size: int):
        return self.get_batch(batch_size)