"""Training loop for the discrete Distance RL agent."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from dist_rl.representations import BetaEMA, recursive_nstep_cosine_loss_ema

from .buffers import AtariReplayBuffer
from .models import (
    AtariEncoder,
    CategoricalActor,
    DistanceTrunkDiscrete,
    TwinQDiscrete,
)
from .utils import LinearSchedule, polyak_update, set_seed


@dataclass
class AgentConfig:
    env_id: str
    seed: int
    device: torch.device
    total_steps: int = 1_000_000
    eval_episodes: int = 10
    eval_freq: int = 50_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    target_entropy_scale: float = 0.98
    updates_per_step: int = 1
    rep_gamma_shape: float = 1.0
    rep_lam: float = 0.5
    rep_huber: float = 0.2
    warmup_steps: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 250_000
    save_dir: str = "checkpoints"


class DiscreteDistAgent:
    """Distance-aware soft actor-critic variant for discrete action spaces."""

    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        config: AgentConfig,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.cfg = config
        self.device = config.device

        set_seed(config.seed)
        self.env.reset(seed=config.seed)
        self.eval_env.reset(seed=config.seed + 1)

        obs_shape = self.env.observation_space.shape
        assert len(obs_shape) == 3, "Atari observations should be (C, 84, 84)."
        self.obs_shape = obs_shape
        self.frames = obs_shape[0]

        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_dim = self.env.action_space.n

        # === Networks ===
        feature_dim = 512
        hidden_dim = 512
        self.encoder = AtariEncoder(self.frames, feature_dim).to(self.device)
        self.encoder_target = AtariEncoder(self.frames, feature_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        self.actor = CategoricalActor(feature_dim, self.action_dim, hidden_dim).to(self.device)
        self.actor_target = CategoricalActor(feature_dim, self.action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q_net = TwinQDiscrete(feature_dim, self.action_dim, hidden_dim).to(self.device)
        self.q_target = TwinQDiscrete(feature_dim, self.action_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.rep_trunk = DistanceTrunkDiscrete(feature_dim, self.action_dim).to(self.device)
        self.rep_trunk_target = DistanceTrunkDiscrete(feature_dim, self.action_dim).to(self.device)
        self.rep_trunk_target.load_state_dict(self.rep_trunk.state_dict())

        # === Optimisers ===
        self.optim_q = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q_net.parameters()), lr=config.lr
        )
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.optim_rep = torch.optim.Adam(self.rep_trunk.parameters(), lr=config.lr)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr)

        self.target_entropy = config.target_entropy_scale * math.log(self.action_dim)

        self.replay = AtariReplayBuffer(
            config.buffer_size,
            observation_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.beta_ema = BetaEMA(decay=0.995)
        self.max_grad_norm = 10.0
        self.steps = 0
        self.best_eval = -float("inf")

        self.epsilon_schedule = LinearSchedule(
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        )

        os.makedirs(config.save_dir, exist_ok=True)

        if wandb.run is not None:
            wandb.run.log_code(".")

        print(
            f"[Init] DiscreteDistAgent env={config.env_id} device={config.device} total_steps={config.total_steps}"
        )

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

    def act(self, obs: np.ndarray, eval_mode: bool = False, epsilon: float = 0.0) -> int:
        if not eval_mode and np.random.rand() < epsilon:
            return int(self.env.action_space.sample())

        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            features = self.encoder(obs_t)
            dist, logits = self.actor(features)
            if eval_mode:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        return int(action.item())

    def evaluate(self) -> float:
        returns = []
        for _ in range(self.cfg.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total = 0.0
            while not done:
                action = self.act(obs, eval_mode=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
        mean_return = float(np.mean(returns))
        if wandb.run is not None:
            wandb.log({"eval/return": mean_return, "step": self.steps})
        return mean_return

    def _update_critics(self, batch: Dict[str, torch.Tensor]) -> float:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].unsqueeze(-1)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)

        features = self.encoder(obs)
        q1, q2 = self.q_net(features)
        q1 = q1.gather(1, actions)
        q2 = q2.gather(1, actions)

        with torch.no_grad():
            next_features = self.encoder_target(next_obs)
            dist_next, logits_next = self.actor_target(next_features)
            log_probs_next = torch.log_softmax(logits_next, dim=-1)
            probs_next = log_probs_next.exp()
            next_q1, next_q2 = self.q_target(next_features)
            min_next_q = torch.min(next_q1, next_q2)
            next_values = (probs_next * (min_next_q - self.alpha * log_probs_next)).sum(dim=-1)
            targets = rewards + (1.0 - dones) * self.cfg.gamma * next_values
            targets = targets.unsqueeze(-1)

        loss_q1 = F.mse_loss(q1, targets)
        loss_q2 = F.mse_loss(q2, targets)
        loss = loss_q1 + loss_q2

        self.optim_q.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optim_q.step()

        if wandb.run is not None:
            wandb.log({"train/q_loss": float(loss.item()), "step": self.steps})
        return float(loss.item())

    def _update_actor_and_alpha(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        with torch.no_grad():
            features = self.encoder(obs)

        dist, logits = self.actor(features)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        q1, q2 = self.q_net(features)
        min_q = torch.min(q1, q2).detach()
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        entropy = -(probs * log_probs).sum(dim=-1)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        if wandb.run is not None:
            wandb.log(
                {
                    "train/actor_loss": float(actor_loss.item()),
                    "train/entropy": float(entropy.mean().item()),
                    "train/alpha": float(self.alpha),
                    "step": self.steps,
                }
            )
        return {
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    def _representation_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            features = self.encoder(obs)
            next_features = self.encoder_target(next_obs)

        z = self.rep_trunk(features, actions)

        with torch.no_grad():
            dist_next, logits_next = self.actor_target(next_features)
            next_actions = dist_next.sample()
            z_next = self.rep_trunk_target(next_features, next_actions)
            q1_t, q2_t = self.q_target(features)
            q_targ = torch.min(q1_t, q2_t).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss, info = recursive_nstep_cosine_loss_ema(
            z,
            z_next,
            dones,
            q_targ,
            discount=self.cfg.gamma,
            gamma_shape=self.cfg.rep_gamma_shape,
            lam=self.cfg.rep_lam,
            huber_delta=self.cfg.rep_huber,
            beta_ema=self.beta_ema,
        )

        self.optim_rep.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rep_trunk.parameters(), self.max_grad_norm)
        self.optim_rep.step()

        if wandb.run is not None:
            wandb.log(
                {
                    "rep/loss": float(loss.item()),
                    "rep/beta": info.get("beta_ema", 0.0),
                    "step": self.steps,
                }
            )
        return {"rep_loss": float(loss.item()), **info}

    def _update_targets(self) -> None:
        polyak_update(self.encoder_target, self.encoder, self.cfg.tau)
        polyak_update(self.q_target, self.q_net, self.cfg.tau)
        polyak_update(self.actor_target, self.actor, self.cfg.tau)
        polyak_update(self.rep_trunk_target, self.rep_trunk, self.cfg.tau)

    def save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        payload = {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q_net": self.q_net.state_dict(),
            "rep_trunk": self.rep_trunk.state_dict(),
            "alpha": self.log_alpha.detach().cpu(),
            "step": self.steps,
        }
        torch.save(payload, path)

    def train(self) -> None:
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for self.steps in range(1, self.cfg.total_steps + 1):
            epsilon = self.epsilon_schedule.value(self.steps)
            if self.steps < self.cfg.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.act(obs, eval_mode=False, epsilon=epsilon)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.replay.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += float(reward)
            episode_length += 1

            if done:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/episode_return": episode_reward,
                            "train/episode_length": episode_length,
                            "step": self.steps,
                        }
                    )
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            if self.steps >= self.cfg.warmup_steps and len(self.replay) >= self.cfg.batch_size:
                for _ in range(self.cfg.updates_per_step):
                    batch = self.replay.sample(self.cfg.batch_size)
                    self._update_critics(batch)
                    self._update_actor_and_alpha(batch)
                    self._representation_loss(batch)
                    self._update_targets()

            if self.steps % self.cfg.eval_freq == 0:
                eval_return = self.evaluate()
                if eval_return > self.best_eval:
                    self.best_eval = eval_return
                    self.save_checkpoint("best")

        # final checkpoint
        self.save_checkpoint("final")
