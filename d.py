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

from loss import RewardAwareCosineHingeLoss

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
        self.max_action = max_action
        self.min_action = min_action

        # Policy network outputs mean and log_std (state-independent log_std for simplicity)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
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
        self.ptr = 0
        self.entry_count = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, action, reward, done):
        self.ptr += 1
        self.entry_count += 1
        if self.ptr >= self.max_size:
            self.ptr = 0  # Overwrite when buffer is full

        self.obs[self.ptr] = torch.tensor(
            obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(
            action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor(
            reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(
            done, dtype=torch.float32, device=self.device)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:        
        # print(f"Buffer ptr: {self.ptr}, max size: {self.max_size}, batch size: {batch_size}, entry count: {self.entry_count}")
        if self.entry_count >= self.max_size:
            idxs = np.random.choice(self.max_size, size=batch_size, replace=False)
        else:
            idxs = np.random.choice(self.ptr, size=batch_size, replace=False)

        return (
            self.obs[idxs].detach(),
            self.actions[idxs].detach(),
            self.rewards[idxs].detach(),
            self.dones[idxs].detach(),
        )

# -----------------------------
# Distance Agent
# -----------------------------


class DistanceAgent:
    def __init__(
        self,
        # env_id="Ant-v4",
        # env_id="Ant-v3",
        env_id="Pendulum-v1",
        seed=0,
        total_steps=20_000,
        buffer_size=10**5,
        update_epochs=1,
        batch_size=64,
        distance_training_start=64,
        policy_training_start=500,
        number_of_comparisons=500,
        number_of_topk=5,
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
        assert policy_training_start >= number_of_comparisons, "Policy training start must be >= number of comparisons."

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        min_action = self.env.action_space.low[0]
        max_action = self.env.action_space.high[0]

        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")

        # Model
        self.actor = Actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            max_action=max_action,
            min_action=min_action
        ).to(self.device)

        self.distance = Distance(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
        ).to(self.device)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.distance_optimizer = optim.AdamW(self.distance.parameters(), lr=lr)

        # Buffer + hyperparams
        self.buffer = RolloutBuffer(
            buffer_size, obs_dim, act_dim, hidden,
            self.device)

        self.distance_training_start = distance_training_start
        self.total_steps = total_steps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.policy_training_start = policy_training_start
        self.number_of_comparisons = number_of_comparisons
        self.number_of_topk = number_of_topk
        self.eval_episodes = eval_episodes

        assert batch_size % 2 == 0, "Batch size must be even for distance training."

    def train_distance(self, buffer: RolloutBuffer):
        obs, actions, rewards, dones = buffer.get_batch(self.batch_size)

        # Compute pairwise distances and rewards
        d1 = self.distance(obs[:self.batch_size // 2],
                           actions[:self.batch_size // 2])
        d2 = self.distance(obs[self.batch_size // 2:],
                           actions[self.batch_size // 2:])
        v1 = rewards[:self.batch_size // 2]
        v2 = rewards[self.batch_size // 2:]

        # Optimize distance model
        self.distance_optimizer.zero_grad()
        distance_loss = RewardAwareCosineHingeLoss()(d1, d2, v1, v2)
        distance_loss.backward()
        self.distance_optimizer.step()

    def train_policy(self):
        for _ in range(self.update_epochs):
            obs_batch, _, _, _ = self.buffer.get_batch(self.batch_size)
            # Compute distance features

            action_pred = self.actor(obs_batch).detach()
            d_batch = self.distance(obs_batch, action_pred)

            # Sample N random transitions for comparison
            obs_comp_batch, action_comp_batch, reward_comp_batch, _ = self.buffer.get_batch(self.number_of_comparisons)
            
            with torch.no_grad():
                d_comp_batch = self.distance(obs_comp_batch, action_comp_batch)
            
            # Compare each in d_batch to each in d_comp_batch and select the rewards of the top K
            # Using differentiable soft top-k approximation with temperature scaling
             
            topk_rewards = []
            temperature = 10.0  # Higher temperature makes selection sharper
            
            for d in d_batch:
                similarities = torch.cosine_similarity(
                    d.unsqueeze(0), d_comp_batch, dim=-1)
                
                # Use softmax with temperature to create differentiable weights
                # Higher similarities get higher weights
                weights = torch.softmax(similarities * temperature, dim=0)
                
                # Take weighted average of rewards (differentiable)
                weighted_reward = torch.sum(weights * reward_comp_batch)
                topk_rewards.append(weighted_reward)

            topk_rewards = torch.stack(topk_rewards)
            
            # Policy loss (maximize distance features)
            policy_loss = -topk_rewards.mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

    def train(self):
        steps_collected = 0

        obs, _ = self.env.reset()

        while steps_collected < self.total_steps:

            torch_obs = torch.tensor(
                obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(torch_obs).detach().cpu().numpy()[0]
            noise = np.random.normal(0, 0.3, size=action.shape)
            action = (action + noise).clip(
                self.env.action_space.low, self.env.action_space.high)
            
            next_obs, reward, done, truncated, _ = self.env.step(action)

            self.buffer.add(obs, action, reward, done)
            steps_collected += 1

            # Train distance model
            if steps_collected > self.distance_training_start:             
                self.train_distance(self.buffer)

            # Training policy
            if steps_collected > self.policy_training_start:
                self.train_policy()
                
            #evaluate policy every 1000 steps
            if steps_collected % 100 == 0:
                print(f'Evaluating at step {steps_collected}...')
                total_reward = 0.0
                for _ in range(self.eval_episodes):
                    eval_obs, _ = self.eval_env.reset()
                    done = False
                    truncated = False
                    while not done and not truncated:
                        torch_eval_obs = torch.tensor(
                            eval_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        eval_action = self.actor(torch_eval_obs).detach().cpu().numpy()[0]
                        eval_obs, eval_reward, done, truncated, _ = self.eval_env.step(eval_action)
                        total_reward += eval_reward
                        
                avg_reward = total_reward / self.eval_episodes
                print(f"Steps: {steps_collected}, Average Eval Reward: {avg_reward}")

            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()
                
            


# -----------------------------
# CLI
# -----------------------------
def main():
    # parser = argparse.ArgumentParser(
    #     description="Quick custom PPO for Gymnasium Pendulum-v1")
    # parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument("--total-steps", type=int, default=200_000)
    # parser.add_argument("--rollout-steps", type=int, default=2048)
    # parser.add_argument("--update-epochs", type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--gae-lambda", type=float, default=0.95)
    # parser.add_argument("--clip-coef", type=float, default=0.2)
    # parser.add_argument("--vf-coef", type=float, default=0.5)
    # parser.add_argument("--ent-coef", type=float, default=0.0)
    # parser.add_argument("--max-grad-norm", type=float, default=0.5)
    # parser.add_argument("--lr", type=float, default=3e-4)
    # parser.add_argument("--hidden", type=int, default=64)
    # parser.add_argument("--eval-episodes", type=int, default=5)
    # args = parser.parse_args()

    agent = DistanceAgent(
        # env_id=args.env_id,
        # seed=args.seed,
        # batch_size=args.batch_size,
        # lr=args.lr,
        # hidden=args.hidden,
        # device=args.device,
        # eval_episodes=args.eval_episodes,
    )
    agent.train()


if __name__ == "__main__":
    main()
