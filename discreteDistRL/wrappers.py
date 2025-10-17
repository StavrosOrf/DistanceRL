"""Gymnasium wrappers and helpers tailored for Atari experiments."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    ClipReward,
)

# Handle FrameStack/FrameStackObservation across Gymnasium versions
try:
    from gymnasium.wrappers import FrameStackObservation as FrameStack
except ImportError:
    try:
        from gymnasium.wrappers.frame_stack import FrameStack
    except ImportError:
        from gymnasium.wrappers import FrameStack  # type: ignore


class ScaleObservation(gym.ObservationWrapper):
    """Scale observations to [0, 1] range as float32."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Update observation space to reflect float32 dtype
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Scale observation from uint8 [0, 255] to float32 [0, 1]."""
        return obs.astype(np.float32) / 255.0


def make_atari_env(
    env_id: str,
    seed: int,
    frames: int = 4,
    sticky: bool = True,
    clip_rewards: bool = True,
) -> gym.Env:
    """Create a wrapped Atari environment with DQN-style preprocessing.

    Args:
        env_id: Gymnasium Atari environment identifier (``"ALE/Pong-v5"`` etc.).
        seed: Random seed for determinism.
        frames: Number of history frames to stack.
        sticky: Whether to enable sticky actions (repeat-action probability 0.25).
        clip_rewards: Whether to clip rewards to ``{-1, 0, +1}``.

    Returns:
        A Gymnasium environment yielding stacked 84x84 grayscale observations
        matching the preprocessing popularised by the DQN/Nature paper.
    
    Note:
        Requires ale-py and AutoROM for ALE environments.
        Install via: pip install ale-py AutoROM[accept-rom-license]
    """
    # Ensure ALE environments are registered
    try:
        import ale_py  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "ale-py is required for Atari environments. "
            "Install via: pip install ale-py AutoROM[accept-rom-license]"
        ) from err

    kwargs = {"frameskip": 1}
    if sticky:
        kwargs["repeat_action_probability"] = 0.25
    else:
        kwargs["repeat_action_probability"] = 0.0

    try:
        env = gym.make(env_id, **kwargs)
    except gym.error.NamespaceNotFound as err:
        raise RuntimeError(
            f"Atari environment '{env_id}' not found. "
            "Ensure ale-py and AutoROM are installed: "
            "pip install ale-py AutoROM[accept-rom-license]"
        ) from err
    env.reset(seed=seed)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
        terminal_on_life_loss=False,
    )
    env = FrameStack(env, frames)

    if clip_rewards:
        env = ClipReward(env, min_reward=-1.0, max_reward=1.0)

    # Convert to float32 in [0, 1] to match common deep RL implementations.
    env = ScaleObservation(env)

    return env


__all__ = ["make_atari_env", "ScaleObservation"]
