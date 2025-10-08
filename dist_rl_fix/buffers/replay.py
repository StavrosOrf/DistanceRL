from typing import Tuple
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim, device, gamma=0.99, n_step=20):
        self.max_size = int(size)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.n_step = int(n_step)

        self.obs = torch.zeros((self.max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((self.max_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.max_size, act_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.max_size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.max_size,), dtype=torch.float32, device=self.device)
        self.rtg = torch.zeros((self.max_size,), dtype=torch.float32, device=self.device)
        self.nreturn = torch.zeros((self.max_size,), dtype=torch.float32, device=self.device)

        self.ptr = -1
        self.entry_count = 0
        self.active_entries = 0
        self._ep_idx = []

    def _advance(self):
        self.ptr = (self.ptr + 1) % self.max_size
        self.entry_count = min(self.entry_count + 1, self.max_size)
        return self.ptr

    def add(self, obs, next_obs, action, reward, done):
        i = self._advance()
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[i] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[i] = torch.as_tensor(float(done), dtype=torch.float32, device=self.device)
        self._ep_idx.append(i)
        if done:
            self._finish_episode(self._ep_idx)
            self._ep_idx.clear()
            self.active_entries = self.entry_count

    @torch.no_grad()
    def _finish_episode(self, ep_idx):
        G = 0.0
        for idx in reversed(ep_idx):
            r = float(self.rewards[idx].item())
            G = r + self.gamma * G
            self.rtg[idx] = G
        # n-step
        T = len(ep_idx)
        r = self.rewards[ep_idx].detach().cpu().numpy()
        import numpy as _np
        gammas = _np.power(self.gamma, _np.arange(self.n_step, dtype=_np.float32))
        for t in range(T):
            i = ep_idx[t]
            h = min(self.n_step, T - t)
            self.nreturn[i] = float((r[t:t+h] * gammas[:h]).sum())

    def sample(self, batch_size: int):
        effective = min(self.max_size, self.active_entries)
        replace = effective < batch_size
        idx = np.random.choice(effective, size=batch_size, replace=replace)
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        return (
            self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx],
            self.rtg[idx], self.nreturn[idx], idx
        )

    def sample_candidates(self, num: int):
        effective = max(1, min(self.max_size, self.active_entries))
        num = min(num, effective)
        idx = np.random.choice(effective, size=num, replace=False)
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        return (
            self.obs[idx], self.actions[idx], self.rtg[idx], self.nreturn[idx], idx
        )
