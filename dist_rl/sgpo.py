import copy
import random
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dist_rl.models import Actor, Critic, StateEncoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.device = device
        self._size = 0
        self.ptr = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = float(reward)
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        idx = np.random.randint(0, self._size, size=batch_size)
        return TransitionBatch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            dones=self.dones[idx],
        )

    @property
    def size(self) -> int:
        return self._size


class ActionArchive:
    def __init__(self, capacity: int, embed_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.embeddings = torch.zeros((capacity, embed_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.qualities = torch.full((capacity,), float("-inf"), dtype=torch.float32, device=device)
        self.count = 0

    def add(self, z: torch.Tensor, actions: torch.Tensor, q_values: torch.Tensor):
        if z.numel() == 0:
            return

        z = z.detach()
        actions = actions.detach()
        q_values = q_values.detach()

        for zi, ai, qi in zip(z, actions, q_values):
            qi_val = float(qi.item())
            if self.count < self.capacity:
                idx = self.count
                self.count += 1
            else:
                current = self.qualities[:self.count]
                min_val, min_idx = torch.min(current, dim=0)
                if qi_val <= float(min_val.item()):
                    continue
                idx = int(min_idx.item())

            self.embeddings[idx] = zi
            self.actions[idx] = ai
            self.qualities[idx] = qi_val

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise ValueError("Archive is empty")
        return (
            self.embeddings[: self.count],
            self.actions[: self.count],
            self.qualities[: self.count],
        )

    def __len__(self) -> int:
        return self.count


class SGPOAgent:
    def __init__(
        self,
        env_id,
        seed,
        total_steps=20_000,
        buffer_size=10**5,
        update_epochs_policy=1,
        update_epochs_val=1,
        batch_size=64,
        policy_training_start=500,
        val_training_start=500,
        lr=3e-4,
        hidden_size=128,
        device="cpu",
        wandb_run=None,
        eval_episodes=5,
        v_gamma=0.99,
        sgpo_archive_size=2048,
        sgpo_kernel_tau=0.5,
        sgpo_quality_beta=2.0,
        sgpo_lambda=0.1,
        sgpo_embed_dim=64,
        policy_noise=0.2,
        policy_noise_clip=0.5,
        actor_update_frequency=1,
        **kwargs,
    ):
        
        set_seed(seed)
        self.device = torch.device(device)

        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)

        if hasattr(self.env, "_max_episode_steps"):
            self.max_episode_steps = self.env._max_episode_steps
        else:
            raise Warning("Max episode steps not found! Using default: 1000")

        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.min_action = float(self.env.action_space.low[0])
        self.max_action = float(self.env.action_space.high[0])

        print("=" * 65)
        print("            TRAINING CONFIGURATION")
        print("=" * 65)
        print(f"Environment: {env_id:15s} | Seed:         {seed:5}")
        print(f"Total Steps: {total_steps:15d} | Buffer Size: {buffer_size:5}")
        print(f"Batch Size:  {batch_size:15d} | Hidden Size:  {hidden_size:5}")
        print(f"Learn. Rate: {lr:15} | Device:         {device:5}")
        print(f"Policy Start:{policy_training_start:15d} | Val Start:    {val_training_start:5d}")
        print("=" * 65)
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        print(f"Max episode steps: {self.max_episode_steps}")
        print("=" * 65)

        self.actor = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=hidden_size,
            max_action=self.max_action,
            min_action=self.min_action,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(
            obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_size=hidden_size * 2
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = StateEncoder(
            obs_dim=self.obs_dim, hidden_size=hidden_size, embed_dim=sgpo_embed_dim
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, self.obs_dim, self.act_dim, self.device)
        self.archive = ActionArchive(
            capacity=sgpo_archive_size,
            embed_dim=sgpo_embed_dim,
            act_dim=self.act_dim,
            device=self.device,
        )

        self.discount = 0.99
        self.total_steps = total_steps
        self.update_epochs_policy = update_epochs_policy
        self.update_epochs_val = update_epochs_val
        self.batch_size = batch_size
        self.policy_training_start = policy_training_start
        self.val_training_start = val_training_start
        self.eval_episodes = eval_episodes
        self.actor_update_frequency = max(1, actor_update_frequency)

        self.tau = 0.005
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip

        self.similarity_tau = sgpo_kernel_tau
        self.similarity_beta = sgpo_quality_beta
        self.policy_lambda = sgpo_lambda

        self.wandb_run = wandb_run

        if self.wandb_run is not None:
            self.wandb_run.log_code(".")

    def get_action(self, obs: np.ndarray, add_noise: bool = False) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        if add_noise:
            noise = torch.randn_like(action) * self.policy_noise
            action = action + noise
        action = action.clamp(self.min_action, self.max_action)
        return action.cpu().numpy()[0]

    def soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def sample_batch(self) -> TransitionBatch:
        return self.buffer.sample(self.batch_size)

    def bisimulation_loss(self, batch: TransitionBatch) -> torch.Tensor:
        perm = torch.randperm(batch.obs.size(0), device=batch.obs.device)

        z = self.encoder(batch.obs)
        z_perm = self.encoder(batch.obs[perm])

        with torch.no_grad():
            z_next = self.encoder(batch.next_obs).detach()
            z_next_perm = self.encoder(batch.next_obs[perm]).detach()

        reward_diff = (batch.rewards - batch.rewards[perm]).abs()
        done_mask = (1.0 - batch.dones) * (1.0 - batch.dones[perm])

        next_dist = (z_next - z_next_perm).pow(2).sum(dim=1).sqrt()
        target = reward_diff + self.discount * done_mask * next_dist

        current_dist = (z - z_perm).pow(2).sum(dim=1).sqrt()
        loss = F.mse_loss(current_dist, target)
        return loss

    def update_critic(self, batch: TransitionBatch) -> torch.Tensor:
        with torch.no_grad():
            next_action = self.actor_target(batch.next_obs)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.policy_noise_clip, self.policy_noise_clip
            )
            next_action = (next_action + noise).clamp(self.min_action, self.max_action)
            target_q1, target_q2 = self.critic_target(batch.next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = batch.rewards + self.discount * (1.0 - batch.dones) * target_q.squeeze(-1)

        current_q1, current_q2 = self.critic(batch.obs, batch.actions)
        current_q1 = current_q1.squeeze(-1)
        current_q2 = current_q2.squeeze(-1)

        loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        with torch.no_grad():
            qualities = torch.min(
                *self.critic_target(batch.obs, batch.actions)
            ).squeeze(-1)
            embeddings = self.encoder(batch.obs)
            self.archive.add(embeddings, batch.actions, qualities)

        return loss.detach()

    def compute_archive_target(self, z: torch.Tensor) -> torch.Tensor:
        archive_z, archive_a, archive_q = self.archive.get()

        diff = z.unsqueeze(1) - archive_z.unsqueeze(0)
        dist2 = diff.pow(2).sum(dim=-1)
        weights = torch.exp(-dist2 / (2 * self.similarity_tau ** 2))

        q_centered = archive_q - archive_q.max()
        weights = weights * torch.exp(self.similarity_beta * q_centered.unsqueeze(0))
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
        weights = weights / weights_sum

        targets = weights @ archive_a
        return targets

    def update_actor(self, batch: TransitionBatch) -> torch.Tensor:
        mu = self.actor(batch.obs)
        z = self.encoder(batch.obs).detach()

        if len(self.archive) > 0:
            a_target = self.compute_archive_target(z).detach()
            bc_loss = F.mse_loss(mu, a_target)
        else:
            bc_loss = torch.tensor(0.0, device=self.device)

        q1_mu = self.critic.q1_only(batch.obs, mu).squeeze(-1)
        rl_loss = -q1_mu.mean()

        loss = bc_loss + self.policy_lambda * rl_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return loss.detach()

    def evaluate_policy(self):
        total_reward = 0.0
        ep_steps = []
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            steps = 0
            while not done and not truncated:
                action = self.get_action(obs, add_noise=False)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                steps += 1
            ep_steps.append(steps)

        avg_reward = total_reward / self.eval_episodes
        print(f"[Eval.] Reward {avg_reward:10.2f}, Steps: {np.mean(ep_steps):6.1f}")
        if self.wandb_run is not None:
            self.wandb_run.log(
                {"eval/avg_reward": avg_reward, "eval/avg_ep_length": np.mean(ep_steps)},
                step=self.steps_collected,
            )
        return avg_reward

    def train(self):
        self.steps_collected = 0
        obs, _ = self.env.reset()

        episode_reward = 0.0
        episode_steps = 0

        self.evaluate_policy()

        while self.steps_collected < self.total_steps:
            action = self.get_action(obs, add_noise=True)
            next_obs, reward, done, truncated, _ = self.env.step(action)

            self.buffer.add(obs, action, reward, next_obs, done or truncated)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            self.steps_collected += 1

            if self.buffer.size >= self.batch_size and self.steps_collected > self.val_training_start:
                for _ in range(self.update_epochs_val):
                    batch = self.sample_batch()
                    bisim_loss = self.bisimulation_loss(batch)
                    self.encoder_optimizer.zero_grad()
                    bisim_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                    self.encoder_optimizer.step()

                    critic_loss = self.update_critic(batch)

                    actor_loss = None
                    if (
                        self.steps_collected > self.policy_training_start
                        and self.steps_collected % self.actor_update_frequency == 0
                    ):
                        for _ in range(self.update_epochs_policy):
                            actor_loss = self.update_actor(batch)

                    if self.wandb_run is not None:
                        log_data = {
                            "train/bisim_loss": bisim_loss.item(),
                            "train/critic_loss": critic_loss.item(),
                        }
                        if actor_loss is not None:
                            log_data["train/actor_loss"] = actor_loss.item()
                        self.wandb_run.log(log_data, step=self.steps_collected)

                    self.soft_update(self.critic, self.critic_target)
                    self.soft_update(self.actor, self.actor_target)

            if done or truncated:
                print(
                    f"[Train: {self.steps_collected}/{self.total_steps:<5d}] Reward {episode_reward:10.2f}, Steps: {episode_steps:6.1f}"
                )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"rollout/ep_reward": episode_reward, "rollout/ep_length": episode_steps},
                        step=self.steps_collected,
                    )

                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_steps = 0

                self.evaluate_policy()
