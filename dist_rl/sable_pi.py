import copy
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dist_rl.models import Critic, StateEncoder


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
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        return TransitionBatch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            dones=self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, min_action: float, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)
        self.log_std_min = -5.0
        self.log_std_max = 2.0

        self.register_buffer("act_scale", torch.tensor((max_action - min_action) / 2.0, dtype=torch.float32))
        self.register_buffer("act_bias", torch.tensor((max_action + min_action) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.act_scale + self.act_bias
        log_prob = normal.log_prob(z) - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean, log_std

    def deterministic(self, obs: torch.Tensor):
        mean, _ = self.forward(obs)
        squashed = torch.tanh(mean)
        return squashed * self.act_scale + self.act_bias

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        clipped = torch.clamp((actions - self.act_bias) / (self.act_scale + 1e-6), -0.999, 0.999)
        unsquashed = torch.atanh(clipped)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(unsquashed) - torch.log(1 - clipped.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)

    def kl_divergence(self, obs: torch.Tensor, other: "SquashedGaussianActor") -> torch.Tensor:
        mean1, log_std1 = self.forward(obs)
        with torch.no_grad():
            mean2, log_std2 = other.forward(obs)
        std1 = torch.exp(log_std1)
        std2 = torch.exp(log_std2)

        var_ratio = (std1.pow(2) + 1e-6) / (std2.pow(2) + 1e-6)
        t1 = (mean2 - mean1).pow(2) / (std2.pow(2) + 1e-6)
        kl = 0.5 * (log_std2 - log_std1 + var_ratio + t1 - 1)
        return kl.sum(dim=-1, keepdim=True)


class SABLEPIAgent:
    def __init__(
        self,
        env_id,
        seed,
        total_steps=300_000,
        buffer_size=200_000,
        batch_size=256,
        lr=3e-4,
        hidden_size=256,
        device="cpu",
        wandb_run=None,
        eval_episodes=5,
        gamma=0.99,
        warmup_steps=1000,
        update_epochs=1,
        repr_updates=1,
        target_tau=0.005,
        entropy_coef=0.0,
        kernel_tau=0.5,
        kernel_topk=16,
        trust_region_eta=0.05,
        nip_alpha=0.0,
        nip_beta=10.0,
        eval_interval=10_000,
        **kwargs,
    ):
        del kwargs
        set_seed(seed)
        self.device = torch.device(device)
        self.gamma = gamma
        self.eval_episodes = eval_episodes
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_epochs = update_epochs
        self.repr_updates = repr_updates
        self.target_tau = target_tau
        self.entropy_coef = entropy_coef
        self.kernel_tau = kernel_tau
        self.kernel_topk = kernel_topk
        self.trust_region_eta = trust_region_eta
        self.nip_alpha = nip_alpha
        self.nip_beta = nip_beta
        self.eval_interval = eval_interval
        self.wandb_run = wandb_run

        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)

        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.min_action = float(self.env.action_space.low[0])
        self.max_action = float(self.env.action_space.high[0])

        self.actor = SquashedGaussianActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=hidden_size,
            min_action=self.min_action,
            max_action=self.max_action,
        ).to(self.device)
        self.old_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = Critic(self.obs_dim, self.act_dim, hidden_size=hidden_size).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.encoder = StateEncoder(self.obs_dim, hidden_size=hidden_size, embed_dim=hidden_size // 2).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, self.obs_dim, self.act_dim, self.device)

    def compute_kernel(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scores = -torch.cdist(embeddings, embeddings, p=2).pow(2) / max(self.kernel_tau, 1e-6)
            if self.kernel_topk is not None and embeddings.size(0) > self.kernel_topk:
                topk_scores, topk_idx = torch.topk(scores, self.kernel_topk, dim=1)
                mask = torch.full_like(scores, float("-inf"))
                mask.scatter_(1, topk_idx, topk_scores)
                weights = torch.softmax(mask, dim=1)
            else:
                weights = torch.softmax(scores, dim=1)
        return weights

    def energy_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cross = torch.cdist(x, y, p=2).mean()
        xx = torch.cdist(x, x, p=2).mean()
        yy = torch.cdist(y, y, p=2).mean()
        return 2 * cross - xx - yy

    def reward_predictive_loss(self, batch: TransitionBatch) -> torch.Tensor:
        perm = torch.randperm(batch.obs.size(0), device=batch.obs.device)
        obs_1, obs_2 = batch.obs, batch.obs[perm]
        actions_1, actions_2 = batch.actions, batch.actions[perm]
        rewards_1, rewards_2 = batch.rewards, batch.rewards[perm]
        next_obs_1, next_obs_2 = batch.next_obs, batch.next_obs[perm]

        z1 = self.encoder(obs_1)
        z2 = self.encoder(obs_2)
        next_z1 = self.encoder(next_obs_1)
        next_z2 = self.encoder(next_obs_2)

        reward_loss = F.l1_loss(rewards_1, rewards_2)
        dist_loss = self.energy_distance(next_z1, next_z2)
        return reward_loss + self.gamma * dist_loss

    def smoothed_values(self, obs: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(obs).detach()
        kernel = self.compute_kernel(embeddings)
        return kernel @ q_values

    def critic_update(self, batch: TransitionBatch) -> dict:
        next_actions, next_logp, _, _ = self.actor.sample(batch.next_obs)
        with torch.no_grad():
            target_q1, target_q2 = self.target_critic(batch.next_obs, next_actions)
            target_v = torch.min(target_q1, target_q2)
            if self.entropy_coef > 0:
                target_v = target_v - self.entropy_coef * next_logp
            smoothed_v = self.smoothed_values(batch.next_obs, target_v)
            target = batch.rewards + (1.0 - batch.dones) * self.gamma * smoothed_v

        current_q1, current_q2 = self.critic(batch.obs, batch.actions)
        loss_q1 = F.mse_loss(current_q1, target)
        loss_q2 = F.mse_loss(current_q2, target)
        critic_loss = loss_q1 + loss_q2

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        return {
            "critic_loss": critic_loss.item(),
            "target_q_mean": float(target.mean().item()),
        }

    def neighbor_imitation_loss(self, batch: TransitionBatch, kernel: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(self.nip_beta * adv).detach()
        weights = kernel * weights.t()
        obs_i = batch.obs.unsqueeze(1).expand(-1, batch.obs.size(0), -1)
        actions_j = batch.actions.unsqueeze(0).expand(batch.obs.size(0), -1, -1)
        log_probs = self.actor.log_prob(obs_i.reshape(-1, self.obs_dim), actions_j.reshape(-1, self.act_dim))
        log_probs = log_probs.view(batch.obs.size(0), batch.obs.size(0))
        loss = -(weights * log_probs).mean()
        return loss

    def actor_update(self, batch: TransitionBatch) -> dict:
        actions, log_prob, _, _ = self.actor.sample(batch.obs)
        q1, q2 = self.critic(batch.obs, actions)
        q = torch.min(q1, q2)
        smoothed_v = self.smoothed_values(batch.obs, q.detach())
        advantage = q - smoothed_v

        policy_loss = -(advantage.detach() * log_prob).mean()
        kl = self.actor.kl_divergence(batch.obs, self.old_actor).mean()
        loss = policy_loss + self.trust_region_eta * kl

        if self.nip_alpha > 0:
            kernel = self.compute_kernel(self.encoder(batch.obs).detach())
            with torch.no_grad():
                data_q1, data_q2 = self.critic(batch.obs, batch.actions)
                data_q = torch.min(data_q1, data_q2)
                data_v = self.smoothed_values(batch.obs, data_q)
                adv_data = data_q - data_v
            nip_loss = self.neighbor_imitation_loss(batch, kernel, adv_data)
            loss = loss + self.nip_alpha * nip_loss
        else:
            nip_loss = torch.tensor(0.0, device=self.device)

        self.actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()
        self.old_actor.load_state_dict(self.actor.state_dict())

        return {
            "policy_loss": float(policy_loss.item()),
            "kl": float(kl.item()),
            "nip_loss": float(nip_loss.item()),
        }

    def encoder_update(self, batch: TransitionBatch) -> float:
        loss = self.reward_predictive_loss(batch)
        self.encoder_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
        self.encoder_optim.step()
        return float(loss.item())

    def soft_update_targets(self):
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.mul_(1 - self.target_tau)
                target_param.data.add_(self.target_tau * param.data)

    def evaluate(self, step: int):
        returns = []
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor.deterministic(obs_tensor).cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            returns.append(total_reward)
        mean_return = np.mean(returns)
        print(f"Step {step}: Eval return {mean_return:.2f}")
        if self.wandb_run is not None:
            self.wandb_run.log({"eval/return": mean_return, "step": step})

    def train(self):
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        for step in range(1, self.total_steps + 1):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if step <= self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _, _, _ = self.actor.sample(obs_tensor)
                    action = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            if done:
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "train/episode_return": episode_reward,
                        "train/episode_length": episode_length,
                        "step": step,
                    })
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            if len(self.buffer) < self.batch_size:
                continue

            for _ in range(self.update_epochs):
                batch = self.buffer.sample(self.batch_size)
                encoder_loss = 0.0
                if self.repr_updates > 0:
                    for _ in range(self.repr_updates):
                        encoder_loss = self.encoder_update(batch)
                critic_metrics = self.critic_update(batch)
                actor_metrics = self.actor_update(batch)
                self.soft_update_targets()

                if self.wandb_run is not None:
                    log_data = {
                        "train/critic_loss": critic_metrics["critic_loss"],
                        "train/policy_loss": actor_metrics["policy_loss"],
                        "train/kl": actor_metrics["kl"],
                        "train/nip_loss": actor_metrics["nip_loss"],
                        "train/encoder_loss": encoder_loss,
                        "step": step,
                    }
                    self.wandb_run.log(log_data)

            if step % self.eval_interval == 0:
                self.evaluate(step)
