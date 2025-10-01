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
from dist_rl.utils import RolloutBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RecDistanceAgent:
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

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.distance_optimizer = optim.Adam(self.distance.parameters(), lr=lr)

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
        self.eval_episodes = eval_episodes
        self.tau = 0.005
        self.discount = 0.99

        self.v_gamma = v_gamma
        self.wandb_run = wandb_run

        if not dynamic_beta:
            if env_id == "LunarLanderContinuous-v3":
                self.beta = 700.0
            elif env_id == "Pendulum-v1":
                self.beta = 0.1
            elif env_id == "MountainCarContinuous-v0":
                self.beta = 0.1
            else:
                assert False, "Please set beta manually for this env!!!"
        else:
            self.beta = None  # will be set dynamically during training
        print(f'Setting beta to: {self.beta}')

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_tensor).detach().cpu().numpy()[0]
        return action

    def train_distance(self):

        for _ in range(self.update_epochs_policy):
            start_time = time.time()
            obs, next_obs, actions, rewards, dones = self.buffer.get_batch(self.batch_size)

            d_embeddings = self.distance(obs, actions)
            
            with torch.no_grad():
                next_actions = self.actor_target(next_obs)
                d_embeddings_next = self.distance_target(next_obs, next_actions)
                
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
            obs, next_obs, _, rewards, dones = self.buffer.get_batch(self.batch_size)

            obs_comp, next_obs_comp, actions_comp, rewards_comp, dones_comp = self.buffer.get_batch(self.batch_size)
            
            # Compute distance features
            actions = self.actor(obs)            
            d = self.distance(obs, actions)
            
            with torch.no_grad():
                d_comp = self.distance(obs_comp, actions_comp)                

            # calculate cosine similarity matrix
            d = nn.functional.normalize(d, p=2, dim=1)
            d_comp = nn.functional.normalize(d_comp, p=2, dim=1)

            # calculate cosine similarity matrix between all pairs in the batch
            S_all = torch.matmul(d, d_comp.T)  # (B, B)

            # print(f'Cosine similarity matrix shape: {S_all.shape}')
            # print(f'Cosine similarity matrix values: {S_all}')            

            # Apply logarithmic weighting: w_i = (log(μ + 0.5) - log(i)) / Σ(log(μ + 0.5) - log(j))
            sorted_indices = torch.argsort(
                torch.argsort(rewards_comp, descending=True))
            mu = len(rewards_comp)  # μ is the batch size
            i = sorted_indices.float() + 1  # 1-indexed ranks

            # # Calculate logarithmic weights
            numerator = torch.log(torch.tensor(mu + 0.5)) - torch.log(i)
            # # Calculate denominator as sum over all j from 1 to μ
            j_values = torch.arange(1, mu + 1, dtype=torch.float32)
            denominator = torch.sum(
                torch.log(torch.tensor(mu + 0.5)) - torch.log(j_values))
            rewards_comp = numerator / denominator
            # rewards_comp = 1 - rewards_comp  # invert weights so higher rewards get higher weights
            # print(f'reward: {rewards_comp}')

            # print(f'Logarithmic weighted reward vector: {rewards_comp}')
            rewards_comp = rewards_comp.unsqueeze(1).repeat(1, S_all.size(1))
            # print(f'reward: {rewards_comp}')

            # Policy loss: maximize cosine similarity weighted by rewards
            policy_loss = - ((S_all + torch.ones_like(S_all))
                             * rewards_comp.T).mean()
            # print(f'Policy loss: {policy_loss.item()}\n')
            
            policy_loss = - S_all.mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            # calculate the norm of the gradients, dont clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"train/policy_loss": policy_loss.item(),
                     "train/policy_grad_norm": grad_norm,
                     "time/policy_step_time": time.time() - start_time},
                    step=self.steps_collected)

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

        if self.wandb_run is not None:
            self.wandb_run.log({"eval/avg_reward": avg_reward,
                                "eval/avg_ep_length": np.mean(ep_steps)},
                               step=self.steps_collected)
        return avg_reward

    def train(self):
        self.steps_collected = 0
        env_step = 0
        ep_reward = 0

        obs, _ = self.env.reset()

        self.evaluate_policy()

        while self.steps_collected < self.total_steps:

            action = self.get_action(obs)
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = (action + noise).clip(
                self.env.action_space.low, self.env.action_space.high)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            
            ep_reward += reward

            self.buffer.add(obs, next_obs, action, reward, done)

            self.steps_collected += 1
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

                eval_reward = self.evaluate_policy()
