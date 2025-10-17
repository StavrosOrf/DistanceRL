"""Utilities for the discrete Distance RL agent."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polyak_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


@dataclass
class LinearSchedule:
    start: float
    end: float
    duration: int

    def value(self, step: int) -> float:
        if step >= self.duration:
            return self.end
        mix = step / float(max(1, self.duration))
        return (1.0 - mix) * self.start + mix * self.end


def cosine_decay(step: int, total: int, max_val: float, min_val: float) -> float:
    if total <= 0:
        return min_val
    ratio = min(1.0, step / float(total))
    cos_inner = math.pi * ratio
    return min_val + 0.5 * (max_val - min_val) * (1.0 + math.cos(cos_inner))


__all__ = ["LinearSchedule", "cosine_decay", "polyak_update", "set_seed"]
