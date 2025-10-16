import os
import torch
import wandb
import yaml
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback


def _load_hparams(algo: str, env_id: str, base_dir: str = "./classic_rl/hyperparams"):
    """
    Load per-env config from YAML. Prefer '<algo>_filled.yaml', fallback to '<algo>.yaml'.
    """
    algo = algo.lower()
    cand = [f"{algo}.yaml", f"{algo}.yaml"]
    hp_path = None
    for name in cand:
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            hp_path = p
            break
    if hp_path is None:
        raise FileNotFoundError(
            f"No YAML found for {algo}. Searched: {', '.join(os.path.join(base_dir, n) for n in cand)}"
        )
    with open(hp_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    env_cfg = cfg.get(env_id)
    if env_cfg is None:
        raise KeyError(f"Environment '{env_id}' not found in {hp_path}. Keys: {list(cfg.keys())[:8]} ...")
    return env_cfg, hp_path


def _maybe_wrap_vecnormalize(env, eval_env, use_norm_obs: bool, use_norm_reward: bool = False):
    """
    For PPO on MuJoCo we often normalize observations.
    Wrap both train and eval envs consistently.
    """
    if not use_norm_obs and not use_norm_reward:
        return env, eval_env, None

    # Make them Vec envs
    train_v = DummyVecEnv([lambda: env])
    eval_v = DummyVecEnv([lambda: eval_env])

    train_v = VecNormalize(train_v, norm_obs=use_norm_obs, norm_reward=use_norm_reward)
    eval_v = VecNormalize(eval_v, norm_obs=use_norm_obs, norm_reward=use_norm_reward, training=False)
    return train_v, eval_v, train_v


def train_sb3_agent(
    algo: str,
    seed: int,
    env_id: str,
    device: str,
    total_steps: int,
    **kwargs
):
    # ---- seeds ----
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ---- load config ----
    config, hp_file = _load_hparams(algo, env_id)
    print(f"[SB3] Algo={algo}  Env={env_id}  Seed={seed}  Device={device}")
    print(f"[SB3] Loaded hyperparams from: {hp_file}")
    print(config)

    # ---- envs ----
    # Seed envs via reset below (Gymnasium API)
    train_env = gym.make(env_id)
    _obs, _info = train_env.reset(seed=seed)
    eval_env = gym.make(env_id)
    _obs_e, _info_e = eval_env.reset(seed=seed + 123)

    # ---- evaluation callback (freq and episodes from YAML) ----
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=int(config.get("eval_interval", 10_000)),
        n_eval_episodes=int(config.get("eval_episodes", 5)),
        deterministic=True,
        render=False,
    )

    algo = algo.lower()
    model = None

    if algo == "ppo":
        # Optional obs normalization (your YAML uses this for MuJoCo PPO)
        normalize_obs = bool(config.get("normalize_obs", False))
        normalize_reward = bool(config.get("normalize_reward", False))
        train_env_wrapped, eval_env_wrapped, vecnorm_ref = _maybe_wrap_vecnormalize(
            train_env, eval_env, normalize_obs, normalize_reward
        )
        # Update eval callback env if we wrapped it
        if vecnorm_ref is not None:
            eval_callback.eval_env = eval_env_wrapped

        # PPO expects n_steps / batch_size / n_epochs keys
        model = PPO(
            config.get("policy", "MlpPolicy"),
            train_env_wrapped if vecnorm_ref is not None else train_env,
            verbose=1,
            device=device,
            tensorboard_log="./logs/",
            n_steps=int(config.get("n_steps", config.get("rollout_steps", 2048))),
            batch_size=int(config.get("batch_size", config.get("minibatch_size", 64))),
            n_epochs=int(config.get("n_epochs", config.get("update_epochs", 10))),
            gamma=float(config.get("gamma", 0.99)),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            clip_range=float(config.get("clip_range", config.get("clip_coef", 0.2))),
            ent_coef=float(config.get("ent_coef", 0.0)),
            vf_coef=float(config.get("vf_coef", 0.5)),
            max_grad_norm=float(config.get("max_grad_norm", 0.5)),
            learning_rate=float(config.get("learning_rate", 3e-4)),
            policy_kwargs=config.get("policy_kwargs")
                or {"net_arch": config.get("net_arch", [config.get("hidden_size", 256),
                                                        config.get("hidden_size", 256)])},
            seed=seed,
        )

    elif algo == "td3":
        # Action noise for exploration
        n_actions = int(np.prod(train_env.action_space.shape))
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=float(config.get("expl_noise", 0.1)) * np.ones(n_actions)
        )

        model = TD3(
            config.get("policy", "MlpPolicy"),
            train_env,
            verbose=1,
            device=device,
            tensorboard_log="./logs/",
            buffer_size=int(config.get("buffer_size", 1_000_000)),
            batch_size=int(config.get("batch_size", 256)),
            learning_starts=int(config.get("start_timesteps", config.get("learning_starts", 25_000))),
            gamma=float(config.get("gamma", 0.99)),
            tau=float(config.get("tau", 0.005)),
            learning_rate=float(config.get("learning_rate", config.get("actor_lr", 3e-4))),
            policy_delay=int(config.get("policy_freq", 2)),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", 1),
            target_policy_noise=float(config.get("policy_noise", 0.2)),
            target_noise_clip=float(config.get("noise_clip", 0.5)),
            action_noise=action_noise,
            policy_kwargs=config.get("policy_kwargs")
                or {"net_arch": config.get("net_arch", [config.get("hidden_size", 256),
                                                        config.get("hidden_size", 256)])},
            seed=seed,
        )

    elif algo == "sac":
        model = SAC(
            config.get("policy", "MlpPolicy"),
            train_env,
            verbose=1,
            device=device,
            tensorboard_log="./logs/",
            buffer_size=int(config.get("buffer_size", 1_000_000)),
            batch_size=int(config.get("batch_size", 256)),
            learning_starts=int(config.get("learning_starts", config.get("start_steps", 10_000))),
            gamma=float(config.get("gamma", 0.99)),
            tau=float(config.get("tau", 0.005)),
            learning_rate=float(config.get("learning_rate", config.get("actor_lr", 3e-4))),
            ent_coef=config.get("ent_coef", "auto"),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", 1),
            policy_kwargs=config.get("policy_kwargs")
                or {"net_arch": config.get("net_arch", [256, 256])},
            seed=seed,
        )

    elif algo == "tqc":
        try:
            from sb3_contrib import TQC
        except ImportError as e:
            raise ImportError(
                "TQC requires sb3-contrib. Install: `pip install stable-baselines3[extra] sb3-contrib`"
            ) from e

        # Prefer full policy_kwargs from YAML (n_critics, n_quantiles, net_arch)
        policy_kwargs = config.get("policy_kwargs") or {"net_arch": config.get("net_arch", [256, 256])}

        model = TQC(
            config.get("policy", "MlpPolicy"),
            train_env,
            verbose=1,
            device=device,
            tensorboard_log="./logs/",
            buffer_size=int(config.get("buffer_size", 1_000_000)),
            batch_size=int(config.get("batch_size", 256)),
            learning_starts=int(config.get("learning_starts", config.get("start_steps", 10_000))),
            gamma=float(config.get("gamma", 0.99)),
            tau=float(config.get("tau", 0.005)),
            learning_rate=float(config.get("learning_rate", 3e-4)),
            ent_coef=config.get("ent_coef", "auto"),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", 1),
            top_quantiles_to_drop_per_net=int(config.get("top_quantiles_to_drop_per_net", 2)),
            target_update_interval=int(config.get("target_update_interval", 1)),
            target_entropy=config.get("target_entropy", "auto"),
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown algorithm '{algo}' (supported: ppo, td3, sac, tqc)")

    # ---- learn ----    
    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
        callback=[WandbCallback(verbose=2), eval_callback],
    )

    # Optionally return the model for downstream saving
    return model
