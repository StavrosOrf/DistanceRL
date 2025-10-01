import math
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


def layer_init(layer: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Gaussian actor-critic with Tanh-squashed actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

        self.pi_body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size), std=math.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size), std=math.sqrt(2)),
            nn.Tanh(),
        )
        self.mu = layer_init(nn.Linear(hidden_size, act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.value_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size), std=math.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size), std=math.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u) * self.action_scale + self.action_bias

    def _log_prob(self, u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        base_log_prob = -0.5 * (((u - mean) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))
        base_log_prob = base_log_prob.sum(-1)
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(-1)
        return base_log_prob - correction

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pi_body(obs)
        mean = self.mu(h)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        u = mean + eps * std
        action = self._squash(u)
        log_prob = self._log_prob(u, mean, log_std)
        value = self.value_net(obs).squeeze(-1)
        with torch.no_grad():
            mean_action = self._squash(mean)
        return action, log_prob, value, mean_action

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scaled = (actions - self.action_bias) / (self.action_scale + 1e-8)
        scaled = torch.clamp(scaled, -0.999999, 0.999999)
        u = 0.5 * torch.log((1 + scaled) / (1 - scaled))
        h = self.pi_body(obs)
        mean = self.mu(h)
        log_std = self.log_std.expand_as(mean)
        log_prob = self._log_prob(u, mean, log_std)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)
        value = self.value_net(obs).squeeze(-1)
        return log_prob, entropy, value


@dataclass
class TrajectoryBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, action, logprob, reward, done, value):
        assert self.ptr < self.max_size, "Buffer overflow"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> TrajectoryBatch:
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.max_size)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.max_size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch = TrajectoryBatch(
            obs=self.obs.clone(),
            actions=self.actions.clone(),
            logprobs=self.logprobs.clone(),
            returns=returns.clone(),
            advantages=advantages.clone(),
        )
        self.ptr = 0
        return batch


class PPOAgent:
    def __init__(
        self,
        env_id: str,
        seed: int,
        total_steps: int,
        rollout_steps: int,
        update_epochs: int,
        minibatch_size: int,
        gamma: float,
        gae_lambda: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        learning_rate: float,
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
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.wandb_run = wandb_run

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            action_low=self.env.action_space.low,
            action_high=self.env.action_space.high,
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.buffer = RolloutBuffer(self.rollout_steps, obs_dim, act_dim, self.device)

    def _evaluate(self, step: int) -> None:
        returns = []
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    _, _, _, mean_action = self.actor_critic.act(obs_tensor)
                action = mean_action.cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            returns.append(total_reward)
        if self.wandb_run is not None:
            self.wandb_run.log({"eval/episode_return": np.mean(returns)}, step=step)

    def train(self) -> None:
        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        global_step = 0
        episode_return = 0.0
        episode_length = 0
        episode_counter = 0

        while global_step < self.total_steps:
            for _ in range(self.rollout_steps):
                global_step += 1
                action, logprob, value, _ = self.actor_critic.act(obs.unsqueeze(0))
                action_np = action.squeeze(0).detach().cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.add(
                    obs.detach(),
                    torch.tensor(action_np, dtype=torch.float32, device=self.device),
                    logprob.detach(),
                    reward,
                    float(done),
                    value.detach(),
                )

                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                episode_return += reward
                episode_length += 1

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
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    episode_return = 0.0
                    episode_length = 0
                    episode_counter += 1

                if global_step % self.eval_interval == 0:
                    self._evaluate(global_step)

                if global_step >= self.total_steps:
                    break

            with torch.no_grad():
                _, _, last_value, _ = self.actor_critic.act(obs.unsqueeze(0))
            batch = self.buffer.compute_returns_advantages(last_value.squeeze(0), self.gamma, self.gae_lambda)

            b_inds = torch.randperm(self.rollout_steps, device=self.device)
            clip_fracs = []
            for epoch in range(self.update_epochs):
                for start in range(0, self.rollout_steps, self.minibatch_size):
                    end = start + self.minibatch_size
                    idx = b_inds[start:end]
                    logprobs, entropy, value = self.actor_critic.evaluate_actions(batch.obs[idx], batch.actions[idx])
                    ratio = (logprobs - batch.logprobs[idx]).exp()
                    clip_fracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                    pg_loss1 = -batch.advantages[idx] * ratio
                    pg_loss2 = -batch.advantages[idx] * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value_clipped = batch.values[idx] + torch.clamp(
                        value - batch.values[idx],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    vf_loss1 = (value - batch.returns[idx]) ** 2
                    vf_loss2 = (value_clipped - batch.returns[idx]) ** 2
                    vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "train/policy_loss": pg_loss.item(),
                                "train/value_loss": vf_loss.item(),
                                "train/entropy": entropy_loss.item(),
                                "train/clip_fraction": np.mean(clip_fracs) if clip_fracs else 0.0,
                            },
                            step=global_step,
                        )

        if self.wandb_run is not None:
            self._evaluate(global_step)

