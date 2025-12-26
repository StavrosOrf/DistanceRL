"""Utilities for the discrete Distance RL agent."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# Reuse RunningMeanStd from continuous utils to keep behavior aligned.
from dist_rl.utils import RunningMeanStd


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


__all__ = [ "polyak_update", "set_seed", "RunningMeanStd"]
