from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

# Handle FrameStack/FrameStackObservation across Gymnasium versions
try:
    # Gymnasium 1.0+ renamed it to FrameStackObservation
    from gymnasium.wrappers import FrameStackObservation as FrameStack
except ImportError:
    try:
        # Older versions had FrameStack in frame_stack submodule
        from gymnasium.wrappers.frame_stack import FrameStack
    except ImportError:
        try:
            # Even older versions kept it at top level
            from gymnasium.wrappers import FrameStack  # type: ignore
        except ImportError as err:
            raise ImportError(
                "FrameStack/FrameStackObservation wrapper is unavailable. "
                "Install gymnasium[atari] or upgrade Gymnasium."
            ) from err

class ChannelFirstFloat(gym.ObservationWrapper):
    """
    Convert observations to CxHxW float32 in [0, 1].
    Handles both FrameStackObservation (already CxHxW) and old FrameStack (HxWxC).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        assert len(obs_space.shape) == 3, f"Expect 3D stacked images, got {obs_space.shape}"
        
        shape = obs_space.shape
        # FrameStackObservation in Gymnasium 1.0+ produces (num_frames, H, W) directly
        # Old FrameStack produced (H, W, num_frames)
        
        # Check if already in (C, H, W) format by looking at dimensions
        # Typically num_frames (4) < spatial dims (84)
        if shape[0] < shape[1] and shape[0] < shape[2]:
            # Already (C, H, W) from FrameStackObservation
            self.needs_transpose = False
            c, h, w = shape
        else:
            # (H, W, C) from old FrameStack - need transpose
            self.needs_transpose = True
            self.transpose_axes = (2, 0, 1)  # (H,W,C) -> (C,H,W)
            h, w, c = shape
            
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)

    def observation(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if self.needs_transpose:
            obs = np.transpose(obs, self.transpose_axes)
        return obs.astype(np.float32) / 255.0

def make_atari_env(
    env_id: str,
    seed: int = 0,
    frame_skip: int = 4,
    grayscale_obs: bool = True,
    scale_obs: bool = False,  # AtariPreprocessing keeps uint8; we scale in ChannelFirstFloat
    terminal_on_life_loss: bool = True,
    frame_stack: int = 4,
    noop_max: int = 30,
) -> gym.Env:
    """
    Create a modern Atari pipeline (Gymnasium ALE v5):
      - AtariPreprocessing: no-ops, frame skip + max-pool, grayscale, terminal on life loss (optional)
      - FrameStack: 4 frames by default
      - ChannelFirstFloat: convert to CxHxW float32 in [0,1]
    
    Note: Requires ale-py and AutoROM for ALE environments.
    Install via: pip install ale-py AutoROM[accept-rom-license]
    """
    try:
        # Ensure ALE environments are registered
        import ale_py  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "ale-py is required for Atari environments. "
            "Install via: pip install ale-py AutoROM[accept-rom-license]"
        ) from err
    
    try:
        # frameskip=1 disables built-in frame skip; we apply it via AtariPreprocessing
        env = gym.make(env_id, full_action_space=False, render_mode=None, frameskip=1)
    except gym.error.NamespaceNotFound as err:
        raise RuntimeError(
            f"Atari environment '{env_id}' not found. "
            "Ensure ale-py and AutoROM are installed: "
            "pip install ale-py AutoROM[accept-rom-license]"
        ) from err
    env.reset(seed=seed)

    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=84,
        grayscale_obs=grayscale_obs,
        scale_obs=scale_obs,
        terminal_on_life_loss=terminal_on_life_loss,
    )
    # FrameStackObservation uses 'stack_size' parameter in Gymnasium 1.0+
    env = FrameStack(env, stack_size=frame_stack)
    env = ChannelFirstFloat(env)
    return env