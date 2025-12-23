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
    total_steps: int = 10_000_000
    eval_episodes: int = 10
    eval_freq: int = 100_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    target_entropy_scale: float = 0.98
    updates_per_step: int = 1
    rep_gamma_shape: float = 2.0
    rep_lam: float = 0.5
    rep_huber: float = 0.2
    warmup_steps: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1_000_000
    policy_smoothing_eps: float = 0.2
    proposal_samples: int = 32
    kernel_softmax_temp: float = 1.0
    kernel_eps: float = 0.05
    kernel_adaptive_tau: bool = True
    learn_alpha: bool = False
    init_alpha: float = 0.01
    max_grad_norm: float = 10.0
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
        init_alpha = torch.tensor(config.init_alpha, device=self.device).clamp(min=1e-6)
        self.log_alpha = torch.nn.Parameter(init_alpha.log())
        self.log_alpha.requires_grad = config.learn_alpha
        self.alpha_opt = (
            torch.optim.Adam([self.log_alpha], lr=config.lr) if config.learn_alpha else None
        )
        self.learn_alpha = config.learn_alpha
        self.fixed_alpha = float(init_alpha.item())

        self.target_entropy = config.target_entropy_scale * math.log(self.action_dim)

        self.replay = AtariReplayBuffer(
            config.buffer_size,
            observation_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.beta_ema = BetaEMA(decay=0.995)
        self.max_grad_norm = config.max_grad_norm
        self.steps = 0
        self.best_eval = -float("inf")

        self.policy_smoothing_eps = config.policy_smoothing_eps
        self.proposal_samples = max(1, config.proposal_samples)
        self.kernel_softmax_temp = config.kernel_softmax_temp
        self.kernel_eps = config.kernel_eps
        self.kernel_adaptive_tau = config.kernel_adaptive_tau
        self.all_actions = torch.arange(
            self.action_dim, device=self.device, dtype=torch.long
        )

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
        print(f"[Init] Observation shape: {obs_shape}, Action dim: {self.action_dim}")
        print(f"[Init] Feature dim: {feature_dim}, Hidden dim: {hidden_dim}")
        print(f"[Init] Buffer size: {config.buffer_size}, Batch size: {config.batch_size}")
        print(f"[Init] Warmup steps: {config.warmup_steps}, Eval freq: {config.eval_freq}")
        print(f"[Init] Gamma: {config.gamma}, Tau: {config.tau}, LR: {config.lr}")
        print(f"[Init] Rep gamma shape: {config.rep_gamma_shape}, Rep lambda: {config.rep_lam}")
        print(
            f"[Init] Policy smoothing epsilon: {self.policy_smoothing_eps}, Proposal samples: {self.proposal_samples}"
        )
        alpha_mode = "learned" if self.learn_alpha else f"fixed={self.fixed_alpha:.4f}"
        print(f"[Init] Alpha mode: {alpha_mode}, Target entropy: {self.target_entropy:.2f}")

    @property
    def alpha(self) -> float:
        if self.learn_alpha:
            return float(self.log_alpha.exp().item())
        return self.fixed_alpha

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

    def _smoothed_action_distribution(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        eps = self.policy_smoothing_eps
        if eps <= 0.0:
            return probs, log_probs

        num_actions = logits.size(-1)
        uniform = torch.full_like(probs, 1.0 / num_actions)
        probs = (1.0 - eps) * probs + eps * uniform
        probs = probs / probs.sum(dim=-1, keepdim=True)
        log_probs = torch.log(probs + 1e-8)
        return probs, log_probs

    def _kernel_tau_instate(
        self, S_rowwise: torch.Tensor, base_temp: float
    ) -> torch.Tensor:
        rows = S_rowwise.size(0)
        device = S_rowwise.device

        tau_max = max(0.75, float(base_temp))
        tau_min = max(0.05, 0.30 * float(base_temp))
        T_sched = 200_000
        p = min(1.0, float(self.steps) / float(T_sched))
        tau_sched = tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(math.pi * p))

        tau = torch.full((rows, 1), tau_sched, device=device)
        if self.kernel_adaptive_tau:
            row_std = S_rowwise.std(dim=1, keepdim=True).clamp(min=1e-4)
            tau = tau + row_std
        return tau

    def _qhat_in_state_norm(
        self, features: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        K = max(1, self.proposal_samples)

        with torch.no_grad():
            proposal_actions = torch.multinomial(
                probs.detach(), num_samples=K, replacement=True
            )
            z_k = self.rep_trunk_target(features, proposal_actions)
            z_k = F.normalize(z_k, p=2, dim=-1)
            q1_t, q2_t = self.q_target(features)
            q_min_target = torch.min(q1_t, q2_t)
            q_proposals = q_min_target.gather(1, proposal_actions)

        actions_full = self.all_actions.unsqueeze(0).expand(features.size(0), -1)
        z_anchor = self.rep_trunk(features, actions_full)
        z_anchor = F.normalize(z_anchor, p=2, dim=-1)

        S = torch.einsum("bah,bkh->bak", z_anchor, z_k).clamp(-1.0, 1.0)
        S_flat = S.view(-1, K)
        tau_flat = self._kernel_tau_instate(
            S_flat, base_temp=self.kernel_softmax_temp
        )
        W = torch.softmax(S_flat / tau_flat, dim=-1)
        if self.kernel_eps > 0.0:
            W = (1.0 - self.kernel_eps) * W + self.kernel_eps / K
        W = W.view(features.size(0), self.action_dim, K)

        q_prop = q_proposals.unsqueeze(1)
        q_bar = (W * q_prop).sum(dim=-1, keepdim=True).detach()
        q_tilde = q_prop - q_bar
        Qhat = (W * q_tilde).sum(dim=-1)

        if wandb.run is not None:
            wandb.log(
                {
                    "kernel_instate/mean_tau": float(tau_flat.mean().item()),
                    "kernel_instate/qhat_abs_mean": float(Qhat.abs().mean().item()),
                },
                step=self.steps,
            )

        return Qhat

    def evaluate(self) -> float:
        print(f"\n[Eval] Step {self.steps}: Starting evaluation ({self.cfg.eval_episodes} episodes)...")
        returns = []
        for ep_idx in range(self.cfg.eval_episodes):
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
        std_return = float(np.std(returns))
        min_return = float(np.min(returns))
        max_return = float(np.max(returns))
        print(f"[Eval] Mean: {mean_return:.2f} ± {std_return:.2f} (Min: {min_return:.2f}, Max: {max_return:.2f})")
        if wandb.run is not None:
            wandb.log({"eval/avg_reward": mean_return}, step=self.steps)
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
            _, logits_next = self.actor_target(next_features)
            probs_next, log_probs_next = self._smoothed_action_distribution(logits_next)
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
            wandb.log({"train/q_loss": float(loss.item())},
                      step=self.steps)
        return float(loss.item())

    def _update_actor_and_alpha(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        with torch.no_grad():
            features = self.encoder(obs)

        _, logits = self.actor(features)
        probs, log_probs = self._smoothed_action_distribution(logits)
        qhat = self._qhat_in_state_norm(features, probs)
        alpha_val = self.alpha
        actor_loss = (probs * (alpha_val * log_probs - qhat)).sum(dim=-1).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        entropy = -(probs * log_probs).sum(dim=-1)
        if self.learn_alpha:
            alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            if self.alpha_opt is None:
                raise RuntimeError("Alpha optimizer is not initialized despite learn_alpha=True")
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        if wandb.run is not None:
            wandb.log(
                {
                    "train/actor_loss": float(actor_loss.item()),
                    "train/entropy": float(entropy.mean().item()),
                    "train/alpha": float(self.alpha),
                    "train/qhat_mean": float(qhat.mean().item()),
                }, step=self.steps,
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
            _, logits_next = self.actor_target(next_features)
            probs_next, _ = self._smoothed_action_distribution(logits_next)
            proposal_actions = torch.multinomial(
                probs_next, num_samples=self.proposal_samples, replacement=True
            )
            z_candidates = self.rep_trunk_target(next_features, proposal_actions)
            q1_next, q2_next = self.q_target(next_features)
            q_min_next = torch.min(q1_next, q2_next)
            q_candidates = q_min_next.gather(1, proposal_actions)
            best_idx = torch.argmax(q_candidates, dim=1)
            batch_indices = torch.arange(proposal_actions.size(0), device=self.device)
            next_actions = proposal_actions[batch_indices, best_idx]
            z_next = z_candidates[batch_indices, best_idx]
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
                }, step=self.steps
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
        print(f"[Save] Checkpoint saved: {tag} at step {self.steps}")

    def train(self) -> None:
        print(f"\n[Train] Starting training loop...")
        print(f"[Train] Warmup phase: steps 1-{self.cfg.warmup_steps}")
        print(f"[Train] Training phase: steps {self.cfg.warmup_steps+1}-{self.cfg.total_steps}\n")
        
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        self.steps = 0
        
        while self.steps <= self.cfg.total_steps:
            self.steps += 1
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
                episode_count += 1
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/episode_return": episode_reward,
                            "train/episode_length": episode_length,
                        }, step=self.steps
                    )
                # Print every episode during warmup, every 10 episodes during training
                if self.steps < self.cfg.warmup_steps or episode_count % 10 == 0:
                    phase = "Warmup" if self.steps < self.cfg.warmup_steps else "Train"
                    print(f"[{phase}] Step {self.steps:7d} | Episode {episode_count:4d} | Return: {episode_reward:7.2f} | Length: {episode_length:4d} | Buffer: {len(self.replay):7d}")
                
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            if self.steps >= self.cfg.warmup_steps and len(self.replay) >= self.cfg.batch_size:
                # Print when starting training phase
                if self.steps == self.cfg.warmup_steps:
                    print(f"\n[Train] Warmup complete! Starting gradient updates...")
                    print(f"[Train] Buffer size: {len(self.replay)}, Alpha: {self.alpha:.4f}\n")
                
                for _ in range(self.cfg.updates_per_step):
                    batch = self.replay.sample(self.cfg.batch_size)
                    self._update_critics(batch)
                    self._update_actor_and_alpha(batch)
                    self._representation_loss(batch)
                    self._update_targets()

            if self.steps % self.cfg.eval_freq == 0:
                eval_return = self.evaluate()
                if eval_return > self.best_eval:
                    print(f"[Eval] New best return: {eval_return:.2f} (previous: {self.best_eval:.2f})")
                    self.best_eval = eval_return
                    self.save_checkpoint("best")
                else:
                    print(f"[Eval] Current best remains: {self.best_eval:.2f}")
                print()  # Empty line for readability

        # final checkpoint
        print(f"\n[Train] Training complete! Total steps: {self.cfg.total_steps}")
        print(f"[Train] Best eval return: {self.best_eval:.2f}")
        self.save_checkpoint("final")
