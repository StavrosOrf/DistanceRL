"""Train Stable-Baselines3 agents on Atari environments with curated hyperparameters."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import linear_schedule
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage
from wandb.integration.sb3 import WandbCallback

try:
    from sb3_contrib import C51, QRDQN
except ImportError:  # pragma: no cover - optional dependency for distributional DQN variants
    C51 = None
    QRDQN = None


def _load_hparams(algo: str, env_id: str, base_dir: str) -> Tuple[Dict[str, Any], str]:
    """Load algorithm/environment specific hyperparameters."""

    algo = algo.lower()
    cfg_path = os.path.join(base_dir, f"{algo}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No hyperparameter file found for '{algo}' under {base_dir}.")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if env_id not in data and "default" not in data:
        raise KeyError(f"Environment '{env_id}' not present in {cfg_path}. Available: {list(data.keys())}")

    env_cfg = data.get(env_id) or data["default"]
    return env_cfg, cfg_path


def _make_vec_env(env_id: str, seed: int, n_envs: int, monitor_dir: str | None) -> VecEnv:
    """Construct a vectorized Atari environment with sticky actions and preprocessing."""

    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)

    env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        env_kwargs={"repeat_action_probability": 0.25},
        wrapper_kwargs={"clip_rewards": True, "frame_stack": 4, "scale": True},
    )
    # Ensure channel-first tensors for PyTorch policies
    env = VecTransposeImage(env)
    return env


def _linear_or_constant(value: float, schedule: str | None):
    if schedule is None:
        return value
    schedule = schedule.lower()
    if schedule == "linear":
        return linear_schedule(value)
    return value


def _build_model(
    algo: str,
    train_env: VecEnv,
    config: Dict[str, Any],
    device: str,
    seed: int,
    tensorboard_log: str,
) -> Any:
    algo = algo.lower()
    policy = config.get("policy", "CnnPolicy")
    lr_value = float(config.get("learning_rate", 1e-4))
    learning_rate = _linear_or_constant(lr_value, config.get("learning_rate_schedule"))
    policy_kwargs = config.get("policy_kwargs")

    if algo == "dqn":
        return DQN(
            policy,
            train_env,
            learning_rate=learning_rate,
            buffer_size=int(config.get("buffer_size", 1_000_000)),
            learning_starts=int(config.get("learning_starts", 80_000)),
            batch_size=int(config.get("batch_size", 32)),
            train_freq=int(config.get("train_freq", 4)),
            gradient_steps=int(config.get("gradient_steps", 1)),
            target_update_interval=int(config.get("target_update_interval", 10_000)),
            exploration_fraction=float(config.get("exploration_fraction", 0.1)),
            exploration_initial_eps=float(config.get("exploration_initial_eps", 1.0)),
            exploration_final_eps=float(config.get("exploration_final_eps", 0.01)),
            gamma=float(config.get("gamma", 0.99)),
            max_grad_norm=float(config.get("max_grad_norm", 10.0)),
            prioritized_replay=bool(config.get("prioritized_replay", False)),
            prioritized_replay_kwargs=config.get("prioritized_replay_kwargs"),
            tensorboard_log=tensorboard_log,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

    if algo == "qrdqn":
        if QRDQN is None:
            raise ImportError(
                "QRDQN is part of sb3-contrib. Install it via `pip install sb3-contrib`."
            )
        return QRDQN(
            policy,
            train_env,
            learning_rate=learning_rate,
            buffer_size=int(config.get("buffer_size", 1_000_000)),
            learning_starts=int(config.get("learning_starts", 80_000)),
            batch_size=int(config.get("batch_size", 32)),
            train_freq=int(config.get("train_freq", 4)),
            gradient_steps=int(config.get("gradient_steps", 1)),
            target_update_interval=int(config.get("target_update_interval", 10_000)),
            exploration_fraction=float(config.get("exploration_fraction", 0.1)),
            exploration_initial_eps=float(config.get("exploration_initial_eps", 1.0)),
            exploration_final_eps=float(config.get("exploration_final_eps", 0.01)),
            gamma=float(config.get("gamma", 0.99)),
            max_grad_norm=float(config.get("max_grad_norm", 10.0)),
            n_quantiles=int(config.get("n_quantiles", 200)),
            kappa=float(config.get("kappa", 1.0)),
            prioritized_replay=bool(config.get("prioritized_replay", False)),
            prioritized_replay_kwargs=config.get("prioritized_replay_kwargs"),
            tensorboard_log=tensorboard_log,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

    if algo == "c51":
        if C51 is None:
            raise ImportError("C51 is part of sb3-contrib. Install it via `pip install sb3-contrib`.")
        return C51(
            policy,
            train_env,
            learning_rate=learning_rate,
            buffer_size=int(config.get("buffer_size", 1_000_000)),
            learning_starts=int(config.get("learning_starts", 80_000)),
            batch_size=int(config.get("batch_size", 32)),
            train_freq=int(config.get("train_freq", 4)),
            gradient_steps=int(config.get("gradient_steps", 1)),
            target_update_interval=int(config.get("target_update_interval", 10_000)),
            exploration_fraction=float(config.get("exploration_fraction", 0.1)),
            exploration_initial_eps=float(config.get("exploration_initial_eps", 1.0)),
            exploration_final_eps=float(config.get("exploration_final_eps", 0.01)),
            gamma=float(config.get("gamma", 0.99)),
            max_grad_norm=float(config.get("max_grad_norm", 10.0)),
            v_min=float(config.get("v_min", -10.0)),
            v_max=float(config.get("v_max", 10.0)),
            n_atoms=int(config.get("n_atoms", 51)),
            prioritized_replay=bool(config.get("prioritized_replay", False)),
            prioritized_replay_kwargs=config.get("prioritized_replay_kwargs"),
            tensorboard_log=tensorboard_log,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

    if algo == "ppo":
        return PPO(
            policy,
            train_env,
            n_steps=int(config.get("n_steps", 128)),
            batch_size=int(config.get("batch_size", 256)),
            n_epochs=int(config.get("n_epochs", 4)),
            gamma=float(config.get("gamma", 0.99)),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            ent_coef=float(config.get("ent_coef", 0.01)),
            vf_coef=float(config.get("vf_coef", 1.0)),
            max_grad_norm=float(config.get("max_grad_norm", 0.5)),
            clip_range=float(config.get("clip_range", 0.1)),
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=1,
            policy_kwargs=policy_kwargs,
            normalize_advantage=bool(config.get("normalize_advantage", True)),
            seed=seed,
        )

    if algo == "a2c":
        return A2C(
            policy,
            train_env,
            n_steps=int(config.get("n_steps", 5)),
            gamma=float(config.get("gamma", 0.99)),
            gae_lambda=float(config.get("gae_lambda", 1.0)),
            ent_coef=float(config.get("ent_coef", 0.01)),
            vf_coef=float(config.get("vf_coef", 0.5)),
            max_grad_norm=float(config.get("max_grad_norm", 0.5)),
            rms_prop_eps=float(config.get("rms_prop_eps", 1e-5)),
            use_rms_prop=bool(config.get("use_rms_prop", True)),
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=1,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

    raise ValueError(f"Unsupported algorithm '{algo}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SB3 Atari training entrypoint")
    parser.add_argument("--env-id", type=str, required=True, help="Gymnasium Atari env id, e.g. ALE/Breakout-v5")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name: dqn, qrdqn, c51, ppo, a2c")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of vectorized envs for training")
    parser.add_argument("--tensorboard-log", type=str, default="./logs/atari", help="Tensorboard log directory")
    parser.add_argument("--output-dir", type=str, default="./classic_rl/atari_runs", help="Checkpoint output directory")
    parser.add_argument(
        "--hyperparams-dir",
        type=str,
        default="./classic_rl/hyperparams/atari",
        help="Directory containing per-algorithm Atari YAML files",
    )
    parser.add_argument("--eval-freq", type=int, default=None, help="Override evaluation interval (env steps)")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override evaluation episode count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.manual_seed_all(args.seed)

    config, hp_path = _load_hparams(args.algo, args.env_id, args.hyperparams_dir)
    print(f"[Atari] Algo={args.algo} Env={args.env_id} Seed={args.seed}")
    print(f"[Atari] Loaded hyperparameters from {hp_path}")
    print(config)

    n_envs = args.n_envs or int(config.get("n_envs", 1))
    eval_interval = args.eval_freq or int(config.get("eval_interval", 100_000))
    eval_episodes = args.eval_episodes or int(config.get("eval_episodes", 10))

    train_monitor_dir = os.path.join(args.output_dir, "monitor")
    train_env = _make_vec_env(args.env_id, args.seed, n_envs, train_monitor_dir)
    eval_env = _make_vec_env(args.env_id, args.seed + 10_000, 1, None)

    model = _build_model(args.algo, train_env, config, args.device, args.seed, args.tensorboard_log)

    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_dir,
        log_path=eval_dir,
        eval_freq=max(eval_interval // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks = [WandbCallback(model_save_path=args.output_dir, verbose=2), eval_callback]

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)

    checkpoint_path = os.path.join(
        args.output_dir, f"{args.algo.lower()}_{args.env_id.replace('/', '_')}_seed{args.seed}_final"
    )
    model.save(checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
