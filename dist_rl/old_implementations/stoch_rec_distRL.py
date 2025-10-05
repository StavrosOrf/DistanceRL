import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import random
import wandb
import time

from dist_rl.models import StochasticActor, Distance
from dist_rl.loss import recursive_reward_aware_cosine_loss
from dist_rl.utils import RolloutBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StochRecDistanceAgent:
    def __init__(
        self,
        env_id,
        seed,
        K=5,
        total_steps=20_000,
        buffer_size=10**5,
        update_epochs_policy=1,
        update_epochs_val=1,
        batch_size=64,
        policy_training_start=500,
        val_training_start=500,
        lr=3e-4,
        hidden_size=64,
        device="cpu",
        wandb_run=None,
        eval_episodes=5,
        eval_freq=1000,
        v_gamma=1.0,
        dynamic_beta=False,
        value_model_type="LSTM",  # "LSTM" or "Transformer"
        **kwargs,
    ):
        set_seed(seed)
        self.device = torch.device(device)

        # Env
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)

        # how to get maximum step per episode
        if hasattr(self.env, '_max_episode_steps'):
            self.max_episode_steps = self.env._max_episode_steps
        else:
            raise Warning(
                "Max episode steps not found!!!!, using default: 1000")

        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        min_action = self.env.action_space.low[0]
        max_action = self.env.action_space.high[0]

        # Print nicely and compact info about the training parameters
        print("="*65)
        print("            TRAINING CONFIGURATION")
        print("="*65)
        print(f"Environment: {env_id:15s} | Seed:         {seed:5}")
        print(f"Total Steps: {total_steps:15d} | Buffer Size: {buffer_size:5}")
        print(f"Batch Size:  {batch_size:15d} | Hidden Size:  {hidden_size:5}")
        print(f"Learn. Rate: {lr:15} | Device:         {device:5}")
        print(
            f"Dist. Start: {val_training_start:15} | Policy Start: {policy_training_start:5}")
        print(
            f"Val Start:   {val_training_start:15} | Eval Ep.:     {eval_episodes:5}")
        print(
            f"Train Ep.:   {update_epochs_policy:15} | Val Ep.:      {update_epochs_val:5}")
        print("="*65)
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        print(f"Max episode steps: {self.max_episode_steps}")
        print("="*65)

        # Model
        self.actor = StochasticActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=hidden_size,
            max_action=max_action,
            min_action=min_action
        ).to(self.device)

        self.distance = Distance(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=hidden_size,
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.distance_target = copy.deepcopy(self.distance)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.distance_optimizer = optim.Adam(self.distance.parameters(), lr=lr)

        # entropy temperature (manual or auto)
        # manual temperature
        self.alpha = kwargs.get("alpha_entropy", 0.2)
        # SAC-style auto tune (optional)
        self.auto_alpha = kwargs.get("auto_alpha", True)
        if self.auto_alpha:
            self.target_entropy = -float(self.act_dim)        # SAC default
            self.log_alpha = torch.tensor(
                np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=kwargs.get("alpha_lr", 3e-4))

        # similarity weight (scales the SGBC term)
        self.sim_coef = kwargs.get("sim_coef", 1.0)

        self.buffer = RolloutBuffer(
            buffer_size, self.obs_dim, self.act_dim, self.device)

        self.K = K
        self.distance_training_start = val_training_start
        self.total_steps = total_steps
        self.update_epochs_policy = update_epochs_policy
        self.update_epochs_val = update_epochs_val
        self.batch_size = batch_size
        self.policy_training_start = policy_training_start
        self.val_training_start = val_training_start
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.tau = 0.005
        self.discount = 0.99

        self.v_gamma = v_gamma
        self.wandb_run = wandb_run

        if self.wandb_run is not None:
            # log code
            wandb.run.log_code(".")

        if not dynamic_beta:
            if env_id == "LunarLanderContinuous-v3":
                self.beta = 5
            elif env_id == "Pendulum-v1":
                self.beta = 0.1
            elif env_id == "MountainCarContinuous-v0":
                self.beta = 0.1
            else:
                assert False, "Please set beta manually for this env!!!"
        else:
            self.beta = None  # will be set dynamically during training
        print(f'Setting beta to: {self.beta}')

    def _exploration_sigma(self):
        """Linear decay: 0..decay_steps ⇒ sigma goes start → final."""
        t = float(self.steps_collected)
        frac = min(1.0, t / max(1, self.expl_decay_steps))
        # linear interpolation
        return self.expl_sigma_start + frac * (self.expl_sigma_final - self.expl_sigma_start)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.act_deterministic(
                obs_tensor).detach().cpu().numpy()[0]
        else:
            action, _, _ = self.actor.sample(obs_tensor)
            action = action.detach().cpu().numpy()[0]
        return action

    def train_distance(self):

        for _ in range(self.update_epochs_val):
            start_time = time.time()
            obs, next_obs, actions, rewards, dones = self.buffer.get_batch(
                self.batch_size)

            d_embeddings = self.distance(obs, actions)

            with torch.no_grad():
                next_actions = self.actor_target.act_deterministic(next_obs)
                d_embeddings_next = self.distance_target(
                    next_obs, next_actions)

                # embeddings = d_embeddings + (self.discount * (1 - dones).unsqueeze(1) * d_embeddings_next)

            distance_loss, info = recursive_reward_aware_cosine_loss(
                embeddings=d_embeddings,
                next_embeddings=d_embeddings_next,
                dones=dones,
                rewards=rewards,
                beta=self.beta,
                discount=self.discount,
                gamma=self.v_gamma,
            )

            self.distance_optimizer.zero_grad()
            distance_loss.backward()
            # calculate the norm of the gradients, dont clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.distance.parameters(), max_norm=1.0)
            self.distance_optimizer.step()

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"train/dist_loss": distance_loss.item(),
                     **{f"train/dist_{k}": v for k, v in info.items()},
                     "train/dist_grad_norm": grad_norm,
                     "time/dist_step_time": time.time() - start_time},
                    step=self.steps_collected)

    def train_policy(self):
        for _ in range(self.update_epochs_policy):
            start_time = time.time()
            obs, next_obs, _, rewards, dones = self.buffer.get_batch(
                self.batch_size)

            obs_comp, next_obs_comp, actions_comp, rewards_comp, dones_comp = self.buffer.get_batch(
                4 * self.batch_size)

            q = torch.quantile(rewards_comp, 0.5)
            keep = rewards_comp >= q
            # print(f'Keeping {keep.sum()}/{len(keep)} samples with rewards >= {q}')
            obs_comp, actions_comp = obs_comp[keep], actions_comp[keep]
            rewards_comp = rewards_comp[keep]

            # for p in self.distance.parameters():
            #     p.requires_grad = False

            # stochastic action + log_prob
            actions, logp, _ = self.actor.sample(
                obs)              # [B, act_dim], [B]
            # grads -> actor only
            d = self.distance(obs, actions)

            with torch.no_grad():
                d_comp = self.distance(obs_comp, actions_comp)

            # calculate cosine similarity matrix
            d = nn.functional.normalize(d, p=2, dim=1)
            d_comp = nn.functional.normalize(d_comp, p=2, dim=1)

            # calculate cosine similarity matrix between all pairs in the batch
            S = torch.matmul(d, d_comp.T)  # (B, B)

            # rank-log weights (keep this!)
            ranks = torch.argsort(torch.argsort(
                rewards_comp, descending=True)).float() + 1
            M = len(rewards_comp)
            w_log = torch.log(torch.tensor(
                M + 0.5, device=rewards_comp.device)) - torch.log(ranks)
            w = (w_log / (w_log.sum() + 1e-8))                    # [B]
            # print(f'Log weights: {w}')
            W = w.unsqueeze(0).repeat(S.size(0), 1)           # [B,B]
            # print(f'Log weights: {W}')

            # keep only top-k neighbors as positives (avoid "all pairs positive")
            k = min(max(8, self.batch_size // 8),M)  # at least 8, at most batch size
            topk_vals, topk_idx = torch.topk(S, k=k, dim=1)
            # print(f'Top-k values: {topk_vals}')
            mask = torch.zeros_like(
                S, dtype=torch.bool).scatter(1, topk_idx, True)
            # print(f'Top-k mask: {mask}')

            # print(f'Shapes: S {S.shape}, W {W.shape}, mask {mask.shape}\n\n')

            # similarity-guided behavior cloning term (positives only, reward-weighted)
            sgbc = - ((S * W) * mask.float()).sum(dim=1).mean()
            
                # SAC-style entropy regularization (maximize entropy -> minimize alpha*logp)
            alpha = self.alpha if not self.auto_alpha else float(self.log_alpha.exp().item())
            entropy_term = alpha * logp.mean()

            policy_loss = self.sim_coef * sgbc + entropy_term

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            # calculate the norm of the gradients, dont clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # --- unfreeze distance for its own updates later
            # for p in self.distance.parameters():
            #     p.requires_grad = True

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"train_p/policy_loss": float(policy_loss.item()),
                    "train_p/sgbc": float(sgbc.item()),
                    "train_p/logp_mean": float(logp.mean().item()),
                    "train_p/alpha": float(self.alpha),
                     "train_p/policy_grad_norm": grad_norm,
                     "train_p/sim_pos_mean": S[mask].mean().item(),
                     "train_p/sim_all_mean": S.mean().item(),
                     "train_p/q": q.item(),
                     "time/policy_step_time": time.time() - start_time},
                    step=self.steps_collected)
                
            # SAC auto-temperature (optional)
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().item())

        # Update the frozen target models
        for param, target_param in zip(self.distance.parameters(), self.distance_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def evaluate_policy(self):
        total_reward = 0.0
        ep_steps = []
        for _ in range(self.eval_episodes):
            eval_obs, _ = self.eval_env.reset()
            eval_done = False
            eval_truncated = False
            env_steps = 0
            while not eval_done and not eval_truncated:
                eval_action = self.get_action(eval_obs, deterministic=True)
                eval_obs, eval_reward, eval_done, eval_truncated, _ = self.eval_env.step(
                    eval_action)
                total_reward += eval_reward
                env_steps += 1

            ep_steps.append(env_steps)

        avg_reward = total_reward / self.eval_episodes
        print(
            f"[Eval.] Reward {avg_reward:10.2f}, Steps: {np.mean(ep_steps):6.1f}")  # (Episodes: {self.eval_episodes})")

        if self.wandb_run is not None:
            self.wandb_run.log({"eval/avg_reward": avg_reward,
                                "eval/avg_ep_length": np.mean(ep_steps)},
                               step=self.steps_collected)
        return avg_reward

    def train(self):
        self.steps_collected = 0
        steps_since_eval = 0
        env_step = 0
        ep_reward = 0

        obs, _ = self.env.reset()

        self.evaluate_policy()

        while self.steps_collected < self.total_steps:

            # remove external np.random.normal noise
            action = self.get_action(obs, deterministic=False)
            next_obs, reward, done, truncated, _ = self.env.step(action)

            ep_reward += reward

            self.buffer.add(obs, next_obs, action, reward, done)

            self.steps_collected += 1
            steps_since_eval += 1
            env_step += 1

            # Train distance model
            if self.steps_collected > self.val_training_start:
                self.train_distance()

            # Training policy
            if self.steps_collected > self.policy_training_start:
                self.train_policy()

            obs = next_obs

            if done or truncated:
                print(
                    f"[Train: {self.steps_collected}/{self.total_steps:<5d}] Reward {ep_reward:10.2f}, Steps: {np.mean(env_step):6.1f}")

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"rollout/ep_reward": ep_reward,
                         "rollout/ep_length": env_step},
                        step=self.steps_collected)

                env_step = 0
                ep_reward = 0
                obs, _ = self.env.reset()

            if steps_since_eval >= self.eval_freq and self.steps_collected > self.policy_training_start:
                steps_since_eval = 0
                eval_reward = self.evaluate_policy()
