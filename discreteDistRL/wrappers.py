"""Gymnasium wrappers and helpers tailored for Atari experiments."""
from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    ClipReward,
    FrameStack,
    TransformObservation,
)


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
    """

    kwargs = {"frameskip": 1}
    if sticky:
        kwargs["repeat_action_probability"] = 0.25
    else:
        kwargs["repeat_action_probability"] = 0.0

    env = gym.make(env_id, **kwargs)
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
    env = TransformObservation(env, lambda obs: obs.astype("float32") / 255.0)

    return env


__all__ = ["make_atari_env"]
