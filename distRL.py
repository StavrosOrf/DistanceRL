import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from loss import reward_aware_cosine_loss_exp
import random
import wandb
import time

from models import Actor, Distance, ValueNetLSTM, ValueNetTransformer
from utils import RolloutBuffer, Trajectory_ReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DistanceAgent:
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
            self.max_episode_steps = 1000

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

        if value_model_type == "Transformer":
            self.distance = ValueNetTransformer(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=hidden_size,
                seq_len=K,
            ).to(self.device)
        else:
            self.distance = ValueNetLSTM(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=hidden_size,
                seq_len=K,
            ).to(self.device)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.distance_optimizer = optim.AdamW(
            self.distance.parameters(), lr=lr)

        self.buffer = Trajectory_ReplayBuffer(
            self.obs_dim,
            self.act_dim,
            self.max_episode_steps,
            self.device,
            max_size=buffer_size
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

        self.v_gamma = v_gamma
        self.wandb_run = wandb_run

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_tensor).detach().cpu().numpy()[0]
        return action

    def train_distance(self):

        for _ in range(self.update_epochs_policy):
            start_time = time.time()
            obs, actions, rewards, _ = self.buffer.get_batch(self.batch_size,
                                                             self.K)

            embeddings = self.distance(obs, actions)

            distance_loss, info = reward_aware_cosine_loss_exp(
                embeddings=embeddings,
                utilities=rewards.sum(dim=1),  # sum over K steps
                beta=None,
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
            obs_batch, actions_batch, rewards_batch, _ = self.buffer.get_batch(self.batch_size,
                                                                               self.K)

            obs_batch_comp, actions_batch_comp, rewards_batch_comp, _ = self.buffer.get_batch(self.batch_size,
                                                                                              self.K)

            # print(f'Batch Obs shape: {obs_batch.shape}')
            # print(f'Batch Action shape: {actions_batch.shape}')

            action_pred = self.actor(obs_batch[:, -1, :])
            # print(f'Action prediction shape: {action_pred.shape}')
            # print(f'Action unsqueeze shape: {action_pred.unsqueeze(1).shape}')
            # print(f'Batch Action shape: {actions_batch[:,:-1,:].shape}')
            all_actions = torch.cat([actions_batch[:, :-1, :],
                                     action_pred.unsqueeze(1)], dim=1)
            # print(f'All actions shape: {all_actions.shape}')

            # Compute distance features
            d_batch = self.distance(obs_batch, all_actions)
            # print(f'\n\nDistance shape: {d_batch.shape}')

            with torch.no_grad():
                d_batch_comp = self.distance(
                    obs_batch_comp, actions_batch_comp)
                # print(f'Comparison Distance shape: {d_batch_comp.shape}')

            # calculate cosine similarity matrix
            d_batch = nn.functional.normalize(d_batch, p=2, dim=1)
            d_batch_comp = nn.functional.normalize(d_batch_comp, p=2, dim=1)
            # print(f'Normalized Distance shape: {d_batch.shape}')
            # print(f'Normalized Comparison Distance shape: {d_batch_comp.shape}')
            S = (d_batch * d_batch_comp).sum(dim=1,
                                             keepdim=True)  # cosine similarity
            # print(f'\n\nCosine similarity shape: {S.shape}')
            # print(f'Cosine similarity values: {S.squeeze()}')

            rewards_comp = rewards_batch_comp.sum(dim=1)  # sum over K steps
            # print(f'Comparison rewards shape: {rewards_comp.shape}')
            # normalize rewards to [0,1]
            rewards_comp = (rewards_comp - rewards_comp.min()) / \
                (rewards_comp.max() - rewards_comp.min() + 1e-8)
            # print(f'reward vector: {rewards_comp}')
            # print(f'Normalized Comparison rewards shape: {rewards_comp.shape}')

            # Policy loss: maximize cosine similarity weighted by rewards
            policy_loss = - ((S.squeeze() + 1) * rewards_comp).mean()

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

        obs, _ = self.env.reset()

        action_traj = torch.zeros(
            (self.max_episode_steps, self.act_dim)).to(self.device)
        state_traj = torch.zeros(
            (self.max_episode_steps, self.obs_dim)).to(self.device)
        done_traj = torch.zeros((self.max_episode_steps, 1)).to(self.device)
        reward_traj = torch.zeros((self.max_episode_steps, 1)).to(self.device)

        self.evaluate_policy()

        while self.steps_collected < self.total_steps:

            action = self.get_action(obs)
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = (action + noise).clip(
                self.env.action_space.low, self.env.action_space.high)

            next_obs, reward, done, truncated, _ = self.env.step(action)

            action_traj[env_step] = torch.tensor(
                action, dtype=torch.float32, device=self.device)
            state_traj[env_step] = torch.tensor(
                obs, dtype=torch.float32, device=self.device)
            done_traj[env_step] = torch.tensor(
                done, dtype=torch.float32, device=self.device)
            reward_traj[env_step] = torch.tensor(
                reward, dtype=torch.float32, device=self.device)

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
                    f"[Train: {self.steps_collected}/{self.total_steps:<5d}] Reward {reward_traj[:env_step].sum().item():10.2f}, Steps: {np.mean(env_step):6.1f}")
                
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"rollout/ep_reward": reward_traj[:env_step].sum().item(),
                         "rollout/ep_length": env_step,
                         "rollout/total_env_steps": self.steps_collected},
                        step=self.steps_collected)

                env_step = 0
                self.buffer.add(
                    state_traj, action_traj, reward_traj, done_traj
                )

                obs, _ = self.env.reset()
                action_traj = torch.zeros(
                    (self.max_episode_steps, self.act_dim)).to(self.device)
                state_traj = torch.zeros(
                    (self.max_episode_steps, self.obs_dim)).to(self.device)
                done_traj = torch.zeros(
                    (self.max_episode_steps, 1)).to(self.device)
                reward_traj = torch.zeros(
                    (self.max_episode_steps, 1)).to(self.device)

                eval_reward = self.evaluate_policy()
