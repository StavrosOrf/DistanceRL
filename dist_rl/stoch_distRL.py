from collections import deque
import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

import random
import wandb
import time

from dist_rl.models import StochasticActor, Distance
from dist_rl.loss import recursive_nstep_cosine_loss
from dist_rl.utils import RTGRolloutBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StochasticDistanceAgent:
    def __init__(
        self,
        env_id,
        seed,
        K=5,
        total_steps=20_000,
        comp_samples=256,
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
        **kwargs,
    ):
        set_seed(seed)
        self.device = torch.device(device)

        # Env
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)

        # how to get maximum step per episode
        if hasattr(self.env, '_max_episode_steps'):
            self.max_episode_steps = self.env._max_episode_steps
        else:
            raise Warning(
                "Max episode steps not found!!!!, using default: 1000")

        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))

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

        space = self.env.action_space
        if isinstance(space, gym.spaces.Box):
            self.action_space_type = "box"
            self.act_dim = int(np.prod(space.shape))
            self.min_action = float(space.low[0])
            self.max_action = float(space.high[0])
            # SAC default for Box
            target_entropy = -float(self.act_dim)
        else:
            assert isinstance(space, gym.spaces.Discrete)
            self.action_space_type = "discrete"
            self.act_dim = int(space.n)
            self.min_action = None
            self.max_action = None
            # good default for discrete
            target_entropy = 0.98 * math.log(self.act_dim + 1e-8)

        # actor with type
        self.actor = StochasticActor(self.obs_dim, self.act_dim, hidden_size,
                                     max_action=self.max_action if self.action_space_type == "box" else 1.0,
                                     min_action=self.min_action if self.action_space_type == "box" else 0.0,
                                     action_space_type=self.action_space_type,
                                     gumbel_tau=1.0).to(self.device)

        # set target entropy / alpha
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(
            0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

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
        self.comp_samples = comp_samples
        self.tau = 0.005
        self.discount = 0.99

        self.expl_sigma_start = 0.3
        self.expl_sigma_final = 0.05
        self.expl_decay_steps = 120_000

        self.v_gamma = v_gamma
        self.wandb_run = wandb_run

        if self.wandb_run is not None:
            wandb.run.log_code(".")

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _one_hot(self, idx: int) -> np.ndarray:
        v = np.zeros(self.act_dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _sched(self, t, t0, t1):
        t = (t - t0) / max(1, (t1 - t0))
        return float(np.clip(t, 0.0, 1.0))

    def _exploration_sigma(self):
        """Linear decay: 0..decay_steps ⇒ sigma goes start → final."""
        t = float(self.steps_collected)
        frac = min(1.0, t / max(1, self.expl_decay_steps))
        # linear interpolation
        return self.expl_sigma_start + frac * (self.expl_sigma_final - self.expl_sigma_start)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.actor.act(obs_tensor, deterministic=deterministic)
        if self.action_space_type == "box":
            return out.cpu().numpy()[0]
        else:
            # return python int
            return int(out.item())

    def train_distance(self):

        for _ in range(self.update_epochs_val):
            start_time = time.time()
            obs, next_obs, actions, rewards, dones, _, n_returns = self.buffer.get_batch(
                self.batch_size)

            d_embeddings = self.distance(obs, actions)

            with torch.no_grad():
                if self.action_space_type == "box":
                    next_a = self.actor_target.act(
                        next_obs, deterministic=True)           # [B, A]
                else:
                    # integer ids -> one-hot vectors
                    next_ids = self.actor_target.act(
                        next_obs, deterministic=True)         # [B]
                    next_a = F.one_hot(
                        next_ids.long(), num_classes=self.act_dim).float()
                d_embeddings_next = self.distance_target(next_obs, next_a)

            distance_loss, info = recursive_nstep_cosine_loss(
                embeddings=d_embeddings,
                next_embeddings=d_embeddings_next,
                dones=dones,
                nreturns=n_returns,
                discount=self.discount,
                n=self.K,
                gamma_shape=self.v_gamma,
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
        - Convert their RTGs to a row-wise target distribution P_tgt.
        - Convert similarities to P_pre.
        - Minimize CE(P_tgt || P_pred).
        """
        start_time = time.time()

        # ---- hyperparams ----
        # number of cosine-nearest neighbors per row
        K = min(32, self.batch_size // 2)
        # ---- sample current batch & large candidate pool ----
        obs, _, _, _, _, _, _ = self.buffer.get_batch(self.batch_size)
        obs_c, _, act_c, _, _, _, nret_c = self.buffer.get_batch(
            self.comp_samples)

        # ---- freeze distance during policy update ----
        for p in self.distance.parameters():
            p.requires_grad = False

        # current actions and embeddings
        # a_pred = self.actor(obs)                               # [B, A]
        # Box: [B,A] floats; Discr: [B,nA] one-hot (ST)
        a_pred, logp, a_mean = self.actor(obs)
        z_i = nn.functional.normalize(self.distance(obs, a_pred), p=2, dim=1)

        # ---------- candidate embeddings (no grad) ----------
        with torch.no_grad():
            z_c = F.normalize(self.distance(obs_c, act_c), p=2, dim=1)  # [M,H]

        # ---------- cosine similarities & top-K ----------
        S_full = (z_i @ z_c.T)                                          # [B,M]
        K_eff = min(self.K, S_full.size(1))
        S_top, idx_top = torch.topk(
            S_full, k=K_eff, dim=1, largest=True)  # [B,K]

        # gather n-step returns aligned with top-K
        N_top = nret_c.index_select(
            0, idx_top.reshape(-1)).reshape(S_top.size(0), K_eff)  # [B,K]

        # ---------- future-aware weights from n-step returns ----------
        # center per-row (advantage-like), then softmax with temperature tau_n
        Nc = (N_top - N_top.mean(dim=1, keepdim=True))
        # [B,K], sum=1
        W = torch.softmax(Nc, dim=1)

        # ---------- predicted probs from similarities ----------
        S_shift = S_top - S_top.max(dim=1, keepdim=True).values
        P_pred = torch.softmax(S_shift, dim=1)                         # [B,K]

        # ---------- geometry loss: CE(W || P_pred) ----------
        eps = 1e-8
        ce = -(W * (P_pred.clamp_min(eps).log())).sum(dim=1).mean()

        # ---------- entropy term (SAC-style) ----------
        alpha = self.alpha  # exp(log_alpha), clamped inside property
        ent_term = alpha * logp.mean()  # maximize entropy -> +alpha * logπ

        policy_loss = ce + ent_term

        # ---- optimize actor ----
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # ---------- alpha auto-tuning toward target entropy ----------
        # after actor step
        with torch.no_grad():
            entropy = (-logp).mean()  # H ≈ E[-log π]

        alpha_loss = -(self.log_alpha *
                       (self.target_entropy - entropy).detach())
        # or equivalently: -(log_alpha * (target - H))

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # unfreeze distance
        for p in self.distance.parameters():
            p.requires_grad = True

        # ---- logs ----
        if self.wandb_run is not None:
            with torch.no_grad():
                sim_w = (S_top * W).sum(dim=1).mean()
                w_entropy = -(W.clamp_min(1e-8) *
                              W.clamp_min(1e-8).log()).sum(dim=1).mean()
                self.wandb_run.log({
                    "train_p/policy_loss": policy_loss.item(),
                    "train_p/ce_main": ce.item(),
                    "train_p/entropy": ent_term.item(),
                    "train_p/alpha_loss": alpha_loss.item(),
                    "train_p/alpha": alpha.item(),
                    "train_p/sim_weighted": sim_w.item(),
                    "train_p/topk_mean": S_top.mean().item(),
                    "train_p/weight_entropy": w_entropy.item(),
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
                eval_action = self.get_action(eval_obs, deterministic=True)
                eval_obs, eval_reward, eval_done, eval_truncated, _ = self.eval_env.step(
                    eval_action)
                total_reward += eval_reward
                env_steps += 1

            ep_steps.append(env_steps)

        avg_reward = total_reward / self.eval_episodes
        print(
            f"[Eval.] Reward {avg_reward:10.3f}, Steps: {np.mean(ep_steps):6.1f}")  # (Episodes: {self.eval_episodes})")

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

        while self.steps_collected < self.total_steps:

            action_env = self.get_action(obs, deterministic=False)

            # step env
            next_obs, reward, done, truncated, _ = self.env.step(action_env)

            # encode action vector for buffer (for Distance)
            if self.action_space_type == "box":
                action_vec = action_env.astype(np.float32)
            else:
                action_vec = self._one_hot(action_env)  # one-hot vector

            ep_reward += reward

            self.buffer.add(obs, next_obs, action_vec, reward, done)

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
                        f"[Collect: {self.steps_collected}/{self.policy_training_start:<5d}] Reward {ep_reward:10.3f}, Steps: {np.mean(env_step):6.1f}")
                else:
                    print(
                        f"[Train: {self.steps_collected}/{self.total_steps:<5d}] Reward {ep_reward:10.3f}, Steps: {np.mean(env_step):6.1f}")

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"rollout/ep_reward": ep_reward,
                         "rollout/ep_length": env_step},
                        step=self.steps_collected)

                env_step = 0
                ep_reward = 0
                obs, _ = self.env.reset()
