import torch
import wandb
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from wandb.integration.sb3 import WandbCallback
import gymnasium as gym
import yaml
import numpy as np


def train_sb3_agent(algo,
                    seed,
                    env_id,
                    device,
                    **kwargs):

    # set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    hyperparam_file = f"./classic_rl/hyperparams/{algo.lower()}.yaml"
    with open(hyperparam_file, 'r') as f:
        config = yaml.safe_load(f)
    config = config.get(env_id, {})

    print(
        f'Training SB3 agent with {algo} on {env_id} with seed {seed} on {device}')
    print(config)

    env = gym.make(env_id)
    o, _ = env.reset(seed=seed)

    eval_env = gym.make(env_id)

    eval_callback = EvalCallback(eval_env,
                                 # best_model_save_path=save_dir,
                                 # log_path=eval_log_dir,
                                 eval_freq=10000,
                                 n_eval_episodes=config.get(
                                     "eval_episodes", 5),
                                 deterministic=True,
                                 render=False)

    if algo == "td3":
        # Create action noise for exploration
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=config.get("expl_noise", 0.1) * np.ones(n_actions))

        model = TD3("MlpPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log="./logs/",
                    buffer_size=config.get("buffer_size", 1000000),
                    batch_size=config.get("batch_size", 256),
                    learning_starts=config.get("start_timesteps", 25000),
                    policy_kwargs=dict(net_arch=[config.get(
                        "hidden_size", 256), config.get("hidden_size", 256)]),
                    gamma=config.get("gamma", 0.99),
                    tau=config.get("tau", 0.005),
                    learning_rate=config.get("actor_lr", 0.0003),
                    policy_delay=config.get("policy_freq", 2),
                    target_policy_noise=config.get("policy_noise", 0.2),
                    target_noise_clip=config.get("noise_clip", 0.5),
                    action_noise=action_noise,
                    seed=seed)
    elif algo == "sac":
        model = SAC("MlpPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log="./logs/",
                    buffer_size=config.get("buffer_size", 1000000),
                    batch_size=config.get("batch_size", 256),
                    learning_starts=config.get("start_steps", 10000),
                    policy_kwargs=dict(net_arch=[config.get(
                        "hidden_size", 256), config.get("hidden_size", 256)]),
                    gamma=config.get("gamma", 0.99),
                    tau=config.get("tau", 0.005),
                    # SB3 SAC uses single LR
                    learning_rate=config.get("actor_lr", 0.0003),
                    seed=seed)
    elif algo == "ppo":
        model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log="./logs/",
                    n_steps=config.get("rollout_steps", 2048),
                    batch_size=config.get("minibatch_size", 64),
                    n_epochs=config.get("update_epochs", 10),
                    gamma=config.get("gamma", 0.99),
                    gae_lambda=config.get("gae_lambda", 0.95),
                    clip_range=config.get("clip_coef", 0.2),
                    ent_coef=config.get("ent_coef", 0.0),
                    vf_coef=config.get("vf_coef", 0.5),
                    max_grad_norm=config.get("max_grad_norm", 0.5),
                    learning_rate=config.get("learning_rate", 0.0003),
                    policy_kwargs=dict(net_arch=[config.get(
                        "hidden_size", 256), config.get("hidden_size", 256)]),
                    seed=seed)
    else:
        # Add support for TQC (from sb3_contrib) as a drop-in off-policy option
        if algo == "tqc":
            try:
                from sb3_contrib import TQC
            except ImportError as e:
                raise ImportError(
                    "TQC requires 'stable-baselines3-contrib'. Install it with: `pip install stable-baselines3[extra] stable-baselines3-contrib`"
                ) from e
            policy_name = config.get("policy", "MlpPolicy")
            net_arch = config.get("net_arch", [config.get("hidden_size", 256), config.get("hidden_size", 256)])

            model = TQC(policy_name,
                        env,
                        verbose=1,
                        device=device,
                        tensorboard_log="./logs/",
                        # common off-policy hyperparams
                        buffer_size=config.get("buffer_size", 1000000),
                        batch_size=config.get("batch_size", 256),
                        learning_starts=config.get("learning_starts", config.get("start_steps", 100)),
                        policy_kwargs=dict(net_arch=net_arch),
                        gamma=config.get("gamma", 0.99),
                        tau=config.get("tau", 0.005),
                        learning_rate=config.get("actor_lr", config.get("learning_rate", 0.0003)),
                        # TQC-specific args (legal parameters only)
                        top_quantiles_to_drop_per_net=config.get("top_quantiles_to_drop_per_net", 2),
                        train_freq=config.get("train_freq", 1),
                        gradient_steps=config.get("gradient_steps", 1),
                        target_update_interval=config.get("target_update_interval", 1),
                        target_entropy=config.get("target_entropy", "auto"),
                        ent_coef=config.get("ent_coef", "auto"),
                        use_sde=config.get("use_sde", False),
                        sde_sample_freq=config.get("sde_sample_freq", -1),
                        use_sde_at_warmup=config.get("use_sde_at_warmup", False),
                        seed=seed)
        else:
            raise ValueError("Unknown algorithm")

    model.learn(total_timesteps=2_500_000,
                progress_bar=True,
                callback=[
                    WandbCallback(
                        verbose=2),
                    eval_callback])

    # model.save(f"{save_path}/last_model.zip")
