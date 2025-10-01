import random
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        self.max_size = size
        self.ptr = 0
        self.full = False
        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        max_index = self.max_size if self.full else self.ptr
        idx = torch.randint(0, max_index, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.net = mlp(obs_dim, hidden_size, act_dim)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.net(obs))
        return (self.action_high + self.action_low) / 2.0 + out * (self.action_high - self.action_low) / 2.0


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.q_net = mlp(obs_dim + act_dim, hidden_size, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q_net(torch.cat([obs, actions], dim=-1)).squeeze(-1)


class TD3Agent:
    def __init__(
        self,
        env_id: str,
        seed: int,
        total_steps: int,
        buffer_size: int,
        batch_size: int,
        start_timesteps: int,
        expl_noise: float,
        policy_noise: float,
        noise_clip: float,
        policy_freq: int,
        gamma: float,
        tau: float,
        actor_lr: float,
        critic_lr: float,
        hidden_size: int,
        eval_interval: int,
        eval_episodes: int,
        device: str = "cpu",
        wandb_run=None,
    ):
        set_seed(seed)
        self.device = torch.device(device)
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)

        self.total_steps = total_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.start_timesteps = start_timesteps
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.tau = tau
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.wandb_run = wandb_run

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high

        self.actor = Actor(obs_dim, act_dim, hidden_size, self.act_low, self.act_high).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_size, self.act_low, self.act_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic1_target = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic2_target = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr
        )

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, self.device)
        self._min_action = torch.as_tensor(self.act_low, dtype=torch.float32, device=self.device)
        self._max_action = torch.as_tensor(self.act_high, dtype=torch.float32, device=self.device)

    def _evaluate(self, step: int):
        returns = []
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor(obs_tensor)
                action_np = action.cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = self.eval_env.step(action_np)
                done = terminated or truncated
                total_reward += reward
            returns.append(total_reward)
        if self.wandb_run is not None:
            self.wandb_run.log({"eval/episode_return": np.mean(returns)}, step=step)

    def _update(self, step: int):
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_obs) + noise
            next_actions = torch.max(torch.min(next_actions, self._max_action), self._min_action)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic_loss = nn.functional.mse_loss(current_q1, target) + nn.functional.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if step % self.policy_freq == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "train/critic_loss": critic_loss.item(),
                        "train/actor_loss": actor_loss.item(),
                    },
                    step=step,
                )
        else:
            if self.wandb_run is not None:
                self.wandb_run.log({"train/critic_loss": critic_loss.item()}, step=step)

    def train(self):
        obs, _ = self.env.reset()
        global_step = 0
        episode_return = 0.0
        episode_length = 0

        while global_step < self.total_steps:
            if global_step < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor(obs_tensor).cpu().numpy()[0]
                action += np.random.normal(0, self.expl_noise, size=action.shape)
                action = np.clip(action, self.act_low, self.act_high)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            episode_length += 1
            global_step += 1

            if done:
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "rollout/episode_return": episode_return,
                            "rollout/episode_length": episode_length,
                        },
                        step=global_step,
                    )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0

            if global_step >= self.start_timesteps:
                self._update(global_step)

            if global_step % self.eval_interval == 0:
                self._evaluate(global_step)

        if self.wandb_run is not None:
            self._evaluate(global_step)

