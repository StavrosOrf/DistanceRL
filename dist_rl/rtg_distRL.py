from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import random
import wandb
import time

from dist_rl.models import Actor, Distance
from dist_rl.loss import recursive_reward_aware_cosine_loss
from dist_rl.utils import RTGRolloutBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RTGRecDistanceAgent:
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
        beta=10,
        v_gamma=1.0,
        q_percentile=0.7,
        top_k=32,
        dynamic_beta=False,
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
        self.actor = Actor(
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

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),    lr=lr)
        self.distance_optimizer = optim.Adam(
            self.distance.parameters(), lr=lr)

        self.buffer = RTGRolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_step=K,
            device=self.device,
        )

        self.K = K
        self.distance_training_start = val_training_start
        self.total_steps = total_steps
        self.update_epochs_policy = update_epochs_policy
        self.update_epochs_val = update_epochs_val
        self.batch_size = batch_size
        self.policy_training_start = policy_training_start
        self.val_training_start = val_training_start
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.tau = 0.005
        self.discount = 0.99

        self.expl_sigma_start = 0.3
        self.expl_sigma_final = 0.05
        self.expl_decay_steps = 120_000
        self.q_percentile = q_percentile
        self.top_k = top_k

        self.v_gamma = v_gamma
        self.wandb_run = wandb_run

        if self.wandb_run is not None:
            # log code
            wandb.run.log_code(".")

        if not dynamic_beta:
            self.beta = beta
        else:
            self.beta = None  # will be set dynamically during training
        print(f'Setting beta to: {self.beta}')

    def _sched(self, t, t0, t1):
        t = (t - t0) / max(1, (t1 - t0))
        return float(np.clip(t, 0.0, 1.0))

    def _exploration_sigma(self):
        """Linear decay: 0..decay_steps ⇒ sigma goes start → final."""
        t = float(self.steps_collected)
        frac = min(1.0, t / max(1, self.expl_decay_steps))
        # linear interpolation
        return self.expl_sigma_start + frac * (self.expl_sigma_final - self.expl_sigma_start)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_tensor).detach().cpu().numpy()[0]
        return action

    def train_distance(self):

        for _ in range(self.update_epochs_val):
            start_time = time.time()
            obs, next_obs, actions, rewards, dones, _, _ = self.buffer.get_batch(
                self.batch_size)

            d_embeddings = self.distance(obs, actions)

            with torch.no_grad():
                next_actions = self.actor_target(next_obs)
                d_embeddings_next = self.distance_target(
                    next_obs, next_actions)

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
        """
        Deterministic policy update that uses ONLY the distance metric's cosine similarity.
        - For each batch state s_i, embed z_i = distance(s_i, actor(s_i)).
        - Build a large candidate pool {(s_c, a_c, rtg_c)} from replay.
        - Embed all candidates z_c = distance(s_c, a_c) (no-grad).
        - Select top-K neighbors per row by cosine(z_i, z_c).
        - Convert their RTGs to a row-wise target distribution P_tgt (softmax with temperature).
        - Convert similarities to P_pred (softmax with temperature).
        - Minimize CE(P_tgt || P_pred), optionally a tiny aux regression to RTG-weighted a*.
        """        
        start_time = time.time()        

        # ---- hyperparams ----
        K = min(32, self.batch_size // 2)     # number of cosine-nearest neighbors per row
        comp_mult = 16                        # candidate pool multiplier (pool = comp_mult * batch_size)
        comp_size = comp_mult * self.batch_size

              # ---- sample current batch & large candidate pool ----
        # buffer.get_batch must return (..., rtg, nreturn)
        obs, _, _, _, _, _, _ = self.buffer.get_batch(self.batch_size)
        obs_c, _, act_c, _, _, rtg_c_dis, rtg_c = self.buffer.get_batch(comp_size)

        # ---- freeze distance during policy update ----
        for p in self.distance.parameters():
            p.requires_grad = False

        # current actions and embeddings
        a_pred = self.actor(obs)                               # [B, A]
        z_i    = self.distance(obs, a_pred)                    # [B, H]
        z_i    = nn.functional.normalize(z_i, p=2, dim=1)      # cosine

        # candidate embeddings (no-grad)
        with torch.no_grad():
            z_c = self.distance(obs_c, act_c)                  # [M, H]
            z_c = nn.functional.normalize(z_c, p=2, dim=1)

        # ---- cosine similarities & top-K selection by cosine ----
        # S = z_i @ z_c^T / tau_sim
        S_full = (z_i @ z_c.T) #/ max(1e-6, tau_sim)           # [B, M]
        K_eff = min(K, S_full.size(1))
        top_vals, top_idx = torch.topk(S_full, k=K_eff, dim=1, largest=True)

        # gather per-row top-K candidates
        RTG_top = rtg_c.index_select(0, top_idx.reshape(-1)).reshape(a_pred.size(0), K_eff)      # [B,K]
        S_top   = top_vals                                                                          # [B,K]

        # ---- target (future-aware) distribution from RTG ----
        # per-row baseline for stability (advantage-like)
        G_base  = RTG_top.mean(dim=1, keepdim=True)
        scores  = (RTG_top - G_base)
        P_tgt   = torch.softmax(scores, dim=1)                                                     # [B,K]

        # ---- predicted distribution from similarities ----
        S_shift = S_top - S_top.max(dim=1, keepdim=True).values
        P_pred  = torch.softmax(S_shift, dim=1)                                                    # [B,K]

        # ---- main loss: cross-entropy CE(P_tgt || P_pred) ----
        eps = 1e-8
        ce = -(P_tgt * (P_pred.clamp_min(eps).log())).sum(dim=1).mean()
        policy_loss = ce

        # ---- optimize actor ----
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # unfreeze distance
        for p in self.distance.parameters():
            p.requires_grad = True

        # ---- logs ----
        if self.wandb_run is not None:
            self.wandb_run.log({
                "train_p/policy_loss": policy_loss.item(),
                "train_p/rtg_c_mean": rtg_c.mean().item(),
                "train_p/rtg_c_max": rtg_c.max().item(),
                "train_p/topk_mean": S_top.mean().item(),
                # "train_p/tau_sim": float(tau_sim),
                "time/policy_step_time": time.time() - start_time
            }, step=self.steps_collected)
            
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
                eval_action = self.get_action(eval_obs)
                eval_obs, eval_reward, eval_done, eval_truncated, _ = self.eval_env.step(
                    eval_action)
                total_reward += eval_reward
                env_steps += 1

            ep_steps.append(env_steps)

        avg_reward = total_reward / self.eval_episodes
        print(
            f"[Eval.] Reward {avg_reward:10.2f}, Steps: {np.mean(ep_steps):6.1f}")  # (Episodes: {self.eval_episodes})")

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            print(f"  New best reward! Models saved.")

        if self.wandb_run is not None:
            self.wandb_run.log({"eval/avg_reward": avg_reward,
                                "eval/best_reward": self.best_reward,
                                "eval/avg_ep_length": np.mean(ep_steps)},
                               step=self.steps_collected)

    def train(self):
        self.steps_collected = 0
        self.steps_since_eval = 0
        env_step = 0
        ep_reward = 0
        self.best_reward = -float('inf')

        obs, _ = self.env.reset()

        self.evaluate_policy()

        while self.steps_collected < self.total_steps:

            action = self.get_action(obs)

            sigma = self._exploration_sigma()          # decays 0.3 → 0.05
            noise = np.random.normal(loc=0.0, scale=sigma, size=action.shape)

            # per-dimension clip to env bounds
            low, high = self.env.action_space.low, self.env.action_space.high
            action = np.clip(action + noise, low, high)

            next_obs, reward, done, truncated, _ = self.env.step(action)

            ep_reward += reward

            self.buffer.add(obs, next_obs, action, reward, done)

            self.steps_collected += 1
            self.steps_since_eval += 1
            env_step += 1

            # Train distance model
            if self.steps_collected > self.val_training_start:
                self.train_distance()

            # Training policy
            if self.steps_collected > self.policy_training_start:
                self.train_policy()

            if self.steps_since_eval >= self.eval_freq and self.steps_collected > self.policy_training_start:
                self.evaluate_policy()
                self.steps_since_eval = 0

            obs = next_obs

            if done or truncated:
                if self.steps_collected < self.policy_training_start:
                    print(
                        f"[Collect: {self.steps_collected}/{self.policy_training_start:<5d}] Reward {ep_reward:10.2f}, Steps: {np.mean(env_step):6.1f}")
                else:
                    print(
                        f"[Train: {self.steps_collected}/{self.total_steps:<5d}] Reward {ep_reward:10.2f}, Steps: {np.mean(env_step):6.1f}")

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"rollout/ep_reward": ep_reward,
                         "rollout/ep_length": env_step,
                         "rollout/sigma": sigma},
                        step=self.steps_collected)

                env_step = 0
                ep_reward = 0
                obs, _ = self.env.reset()
