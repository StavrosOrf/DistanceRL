import torch

class RunningMeanStd:
    def __init__(self, shape, eps=1e-4, device='cpu'):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

def polyak_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

import os, json

class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.f = open(os.path.join(logdir, 'metrics.jsonl'), 'a', encoding='utf-8')
    def log(self, d: dict):
        self.f.write(json.dumps(d) + "\n"); self.f.flush()
    def close(self): self.f.close()

import random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
