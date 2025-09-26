import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, max_action: float = 2.0, min_action: float = -2.0):
        super().__init__()
        assert act_dim == 1, "This quick test targets 1D action space (Pendulum)."

        self.max_action = max_action
        self.min_action = min_action

        # Policy network outputs mean and log_std (state-independent log_std for simplicity)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: action, log_prob, value, mean_action (deterministic)
        """
        return self.actor(obs) * self.max_action


class Distance(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.dist = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Concatenate observations and actions for distance computation
        x = torch.cat([obs, actions], dim=-1)
        return self.dist(x)


class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, hidden_dim: int, device):
        self.obs = torch.zeros((buffer_size, obs_dim),
                               dtype=torch.float32, device=device)
        self.actions = torch.zeros(
            (buffer_size, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.distance = torch.zeros(
            (buffer_size, hidden_dim), dtype=torch.float32, device=device)
        self.ptr = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, action, reward, done):
        self.ptr += 1
        if self.ptr >= self.max_size:
            self.ptr = 0  # Overwrite when buffer is full

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = np.random.choice(self.ptr, size=batch_size, replace=False)

        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs]
        )

# -----------------------------
# Distance Agent
# -----------------------------
class DistanceAgent:
    def __init__(
        self,
        env_id="Pendulum-v1",
        seed=0,
        total_steps=20_000,
        buffer_size=2048,
        update_epochs=10,
        minibatch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.00,
        max_grad_norm=0.5,
        lr=3e-4,
        hidden=64,
        device="cpu",
        eval_episodes=5,
    ):
        set_seed(seed)
        self.device = torch.device(device)

        # Env
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        assert self.env.action_space.shape == (1,)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))

        # Model
        self.actor = Actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            max_action=2.0,
            min_action=-2.0
        ).to(self.device)

        self.distance = Distance(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.distance_optimizer = optim.Adam(self.distance.parameters(), lr=lr)

        # Buffer + hyperparams
        self.buffer = RolloutBuffer(
            buffer_size, obs_dim, act_dim, hidden,
            self.device)

        self.total_steps = total_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.eval_episodes = eval_episodes

    def train_distance(self, buffer: RolloutBuffer):
        if buffer.ptr < buffer.max_size:
            return  # Not enough data to train

        obs, actions, rewards, dones = buffer.get_batch(self.minibatch_size)

        # Compute target distances (using rewards as negative distances)
        target_distances = -rewards.unsqueeze(-1)  # Shape: (batch_size, 1)

        # Predict distances
        pred_distances = self.distance(obs, actions)

        # Compute loss
        loss_fn = nn.MSELoss()
        distance_loss = loss_fn(pred_distances, target_distances)

        # Optimize distance model
        self.distance_optimizer.zero_grad()
        distance_loss.backward()
        self.distance_optimizer.step()

    def train(self):
        steps_collected = 0
        log_every = max(1, self.total_steps // 50)

        obs = self.env.reset()

        while steps_collected < self.total_steps:

            action = self.actor(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, done)

            # Train distance model
            self.train_distance(self.buffer)

            obs = next_obs
            if done:
                obs = self.env.reset()

            print(
                f"Collected {self.rollout_steps} steps, avg_train_ep_ret={avg_tr_ret if avg_tr_ret is not None else float('nan'):.1f}")
            self.ppo_update(batch)
            steps_collected += self.rollout_steps

            if steps_collected % log_every == 0 or steps_collected >= self.total_steps:
                eval_ret = self.evaluate(self.eval_episodes)
                print(f"[{steps_collected:>7}/{self.total_steps}] "
                      f"avg_train_ep_ret={avg_tr_ret if avg_tr_ret is not None else float('nan'):.1f}  "
                      f"eval_ret={eval_ret:.1f}  "
                      f"mean_adv={batch.advantages.mean().item():+.3f}")

        # Final evaluation
        final_eval = self.evaluate(self.eval_episodes)
        print(
            f"Training complete. Final eval return over {self.eval_episodes} episodes: {final_eval:.1f}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Quick custom PPO for Gymnasium Pendulum-v1")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    agent = PPOAgent(
        env_id=args.env_id,
        seed=args.seed,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        eval_episodes=args.eval_episodes,
    )
    agent.train()


if __name__ == "__main__":
    main()
