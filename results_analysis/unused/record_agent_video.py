"""Record evaluation videos for a trained SACDistance agent."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import sys
import os

import gymnasium as gym
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dist_rl_fix.models.networks import GaussianActor


def infer_actor_dims(state_dict: dict) -> Tuple[int, int]:
    """Infer observation/action dims from a GaussianActor state dict."""
    try:
        obs_dim = state_dict["net.net.0.weight"].shape[1]
        act_dim = state_dict["mu.weight"].shape[0]
        return int(obs_dim), int(act_dim)
    except KeyError as err:
        raise KeyError(
            "Unexpected actor state dict structure; cannot infer dims") from err


def _create_video_env(env_id: str, output_dir: Path, name_prefix: str, mujoco_gl: Optional[str]):
    """Create a RecordVideo-wrapped environment with MUJOCO_GL fallbacks."""
    attempt_order = []

    if mujoco_gl:
        attempt_order.append(mujoco_gl)

    current = os.environ.get("MUJOCO_GL")
    if current and current not in attempt_order:
        attempt_order.append(current)

    for candidate in ("egl", "osmesa", None):
        if candidate not in attempt_order:
            attempt_order.append(candidate)

    last_exc: Optional[Exception] = None
    for gl_backend in attempt_order:
        if gl_backend is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = gl_backend
        try:
            base_env = gym.make(env_id, render_mode="rgb_array")
            return gym.wrappers.RecordVideo(
                base_env,
                video_folder=str(output_dir),
                episode_trigger=lambda episode_idx: True,
                name_prefix=name_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    raise RuntimeError(
        "Failed to create video-enabled environment after trying MUJOCO_GL backends: "
        f"{attempt_order}. Install headless OpenGL support and try again."
    ) from last_exc


@dataclass
class EnvSpec:
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained actor checkpoint and record evaluation videos."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="./saved_models/Qbalanced_ActorTemp_InState64_Check_lrAdjust_MaxGradNorm_QTarget_NoisyRep_SACDistanceAgentNew_1011_235204/best.pt",
        help="Path to the saved model checkpoint (e.g. saved_models/.../best.pt)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetah-v5",
        help="Gymnasium environment id (must support rgb_array rendering)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden size used for the Gaussian actor network (matches training config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to record",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_analysis/videos/"),
        help="Directory to save recorded videos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to run the policy on (e.g. cpu, cuda:0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    parser.add_argument(
        "--deterministic",
        default=True,
        help="Use the policy mean (deterministic) instead of sampling actions",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of env steps per episode (None = unlimited)",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="eval",
        help="Name prefix for saved video files",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default='egl',
        help="Optional MUJOCO_GL backend (e.g. egl, osmesa). Set when offscreen rendering fails.",
    )
    return parser.parse_args()


def fetch_env_spec(env_id: str) -> EnvSpec:
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        if not hasattr(obs_space, "shape") or len(obs_space.shape) != 1:
            raise ValueError("Only vector observations are supported for now")
        if not isinstance(act_space, gym.spaces.Box):
            raise ValueError(
                "Environment must have a continuous Box action space")

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        action_low = np.asarray(act_space.low, dtype=np.float32)
        action_high = np.asarray(act_space.high, dtype=np.float32)
        return EnvSpec(obs_dim, act_dim, action_low, action_high)
    finally:
        env.close()


def load_actor(spec: EnvSpec, checkpoint: Path, hidden_size: int, device: str, env_id: str) -> GaussianActor:
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data.get("actor", checkpoint_data)

    expected_obs_dim, expected_act_dim = infer_actor_dims(state_dict)
    if expected_obs_dim != spec.obs_dim or expected_act_dim != spec.act_dim:
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"env '{env_id}' provides obs_dim={spec.obs_dim}, act_dim={spec.act_dim} "
            f"but checkpoint expects obs_dim={expected_obs_dim}, act_dim={expected_act_dim}. "
            "Choose the environment used during training or supply matching dimensions."
        )

    actor = GaussianActor(spec.obs_dim, spec.act_dim,
                          hidden=hidden_size).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def scale_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    # The actor outputs actions in [-1, 1] due to tanh; map back to env bounds.
    return ((action + 1.0) * 0.5) * (high - low) + low


def run_episode(
    env: gym.Env,
    actor: GaussianActor,
    spec: EnvSpec,
    device: torch.device,
    deterministic: bool,
    max_steps: Optional[int],
    seed: Optional[int],
) -> float:
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=device).view(1, -1)
        with torch.no_grad():
            if deterministic:
                mu, _ = actor(obs_tensor)
                action = torch.tanh(mu)
            else:
                action, _, _ = actor.sample(obs_tensor)
        action_np = action.squeeze(0).cpu().numpy()
        env_action = scale_action(action_np, spec.action_low, spec.action_high)
        env_action = np.clip(env_action, spec.action_low, spec.action_high)

        obs, reward, done, truncated, _ = env.step(
            env_action.astype(np.float32))
        total_reward += float(reward)
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return total_reward


def record_videos(
    checkpoint: Path,
    env_id: str,
    hidden_size: int,
    episodes: int,
    output_dir: Path,
    device: str,
    seed: Optional[int],
    deterministic: bool,
    max_steps: Optional[int],
    name_prefix: str,
    mujoco_gl: Optional[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = fetch_env_spec(env_id)
    actor = load_actor(spec, checkpoint, hidden_size, device, env_id)
    print(f"Loaded actor with obs_dim={spec.obs_dim}, act_dim={spec.act_dim}")

    video_env = _create_video_env(env_id, output_dir, name_prefix, mujoco_gl)

    if seed is not None:
        video_env.action_space.seed(seed)

    torch_device = torch.device(device)
    returns = []
    for episode in range(episodes):
        episode_seed = (seed + episode) if seed is not None else None
        ret = run_episode(
            video_env,
            actor,
            spec,
            torch_device,
            deterministic,
            max_steps,
            seed=episode_seed,
        )
        returns.append(ret)
        print(f"Episode {episode + 1}/{episodes}: return={ret:.2f}")

    video_env.close()

    if returns:
        returns_arr = np.asarray(returns, dtype=np.float32)
        print(
            f"Recorded {len(returns)} episode(s). "
            f"Mean return={returns_arr.mean():.2f}, std={returns_arr.std(ddof=0):.2f}."
        )
    print(f"Videos saved to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    record_videos(
        checkpoint=args.checkpoint,
        env_id=args.env,
        hidden_size=args.hidden_size,
        episodes=args.episodes,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        name_prefix=args.name_prefix,
        mujoco_gl=args.mujoco_gl,
    )


if __name__ == "__main__":
    main()
