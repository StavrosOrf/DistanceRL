
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from dist_rl.utils import RolloutBuffer, RunningMeanStd, polyak_update


# ---------------------------------------------------------------------
# Metric utils used by the original JAX DBC implementation:
# - l1: elementwise L1 sum over last dim (reward diffs are scalar anyway)
# - l2: L2 norm over last dim
# - cosine distance: angular distance (arccos cos-sim), used in DBC+MICo mode
# ---------------------------------------------------------------------

def _l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().sum(dim=-1)  # (B,)

def _l2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(((x - y).pow(2)).sum(dim=-1) + eps)  # (B,)

def _angular_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_norm = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
    y_norm = y / (y.norm(p=2, dim=-1, keepdim=True) + eps)
    cos = (x_norm * y_norm).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos)  # (B,)


# ---------------------------------------------------------------------
# DBC modules
# Mirrors the original JAX `SACConvNetwork` + `encoder_network_def` split:
# - Encoder produces actor_z and critic_z (updated ONLY by bisimulation loss)
# - SAC networks consume z's (no gradient to encoder)
# - Reward model consumes actor_z (no encoder gradient)
# - Dynamics model consumes actor_z and action, outputs Gaussian (mu,sigma)
# ---------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_proj = nn.Linear(hidden, z_dim)
        self.critic_proj = nn.Linear(hidden, z_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.actor_proj(h), self.critic_proj(h)


class Actor(nn.Module):
    def __init__(self, z_dim: int, act_dim: int, hidden: int,
                 log_std_bounds: Tuple[float, float] = (-5.0, 2.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        mu = self.mu(h)
        log_std = self.log_std(h)
        low, high = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = low + 0.5 * (high - low) * (log_std + 1.0)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std = self.forward(z)
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        logp = (-0.5 * ((pre_tanh - mu) / std).pow(2) - torch.log(std) -
                0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, z_dim: int, act_dim: int, hidden: int):
        super().__init__()
        in_dim = z_dim + act_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, a], dim=-1)
        return self.q1(x), self.q2(x)


class RewardModel(nn.Module):
    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DynamicsModel(nn.Module):
    """
    Outputs a diagonal Gaussian N(mu, sigma) over next latent z.
    This matches the original JAX implementation which uses mu and sigma and
    includes both in the target distance.
    """
    def __init__(self, z_dim: int, act_dim: int, hidden: int, min_sigma: float = 1e-4):
        super().__init__()
        self.min_sigma = float(min_sigma)
        in_dim = z_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.sigma = nn.Linear(hidden, z_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, a], dim=-1)
        h = self.net(x)
        mu = self.mu(h)
        sigma = F.softplus(self.sigma(h)) + self.min_sigma
        return mu, sigma

    def sample(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.forward(z, a)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma


@dataclass
class DBCInfo:
    sac_critic_loss: float
    sac_policy_loss: float
    alpha_loss: float
    reward_loss: float
    dynamics_loss: float
    bisim_loss: float
    alpha: float


class _DBCBase:
    def __init__(self,
                 env_id: str,
                 seed: int,
                 device,
                 total_steps: int,
                 eval_episodes: int,
                 eval_freq: int,
                 buffer_size: int,
                 batch_size: int,
                 hidden_size: int,
                 gamma: float,
                 tau: float,
                 lr: float,
                 expl_sigma: float = 0.1,
                 target_entropy_scale: float = 1.0,
                 z_dim: int = 50,
                 use_mico: bool = False,
                 warmup_steps: int = 10_000,
                 normalize_obs: int = 1,
                 exp_prefix: str = "exp",
                 save_dir: str = "./runs",
                 log_to_wandb: bool = False,
                 **_unused):
        self.device = device
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)

        self.env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 1)

        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])

        self.low = torch.as_tensor(self.env.action_space.low, device=self.device, dtype=torch.float32)
        self.high = torch.as_tensor(self.env.action_space.high, device=self.device, dtype=torch.float32)

        self.total_steps = int(total_steps)
        self.eval_episodes = int(eval_episodes)
        self.eval_freq = int(eval_freq)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.lr = float(lr)
        self.noise_std = float(expl_sigma)
        self.warmup_steps = int(warmup_steps)

        self.normalize_obs = True if int(normalize_obs) != 0 else False
        self.obs_rms = RunningMeanStd(self.obs_dim, device=self.device)

        self.use_mico = bool(use_mico)

        # Entropy (SAC part)
        self.target_entropy = -float(target_entropy_scale) * float(self.act_dim)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # Networks
        self.encoder = Encoder(self.obs_dim, z_dim=z_dim, hidden=hidden_size).to(self.device)
        self.encoder_t = Encoder(self.obs_dim, z_dim=z_dim, hidden=hidden_size).to(self.device)
        self.encoder_t.load_state_dict(self.encoder.state_dict())

        self.actor = Actor(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)
        self.critic = Critic(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)

        self.actor_t = Actor(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)
        self.critic_t = Critic(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.reward_model = RewardModel(z_dim=z_dim, hidden=hidden_size).to(self.device)
        self.dynamics_model = self._make_dynamics(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)

        # Optimizers (matches JAX split optimizers)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.net_opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr)
        self.dynamics_opt = torch.optim.Adam(self.dynamics_model.parameters(), lr=self.lr)

        # Replay
        self.replay = RolloutBuffer(buffer_size, self.obs_dim, self.act_dim, device=self.device)

        self.steps = 0
        self.best_eval = -float("inf")

        if log_to_wandb and wandb.run is not None:
            wandb.run.log_code(".")

    # Implemented by subclasses (stochastic vs deterministic dynamics)
    def _make_dynamics(self, z_dim: int, act_dim: int, hidden: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def _sample_batch(self):
        obs, next_obs, act, rew, done = self.replay.sample(self.batch_size)
        if self.normalize_obs:
            obs = self.obs_rms.normalize(obs)
            next_obs = self.obs_rms.normalize(next_obs)
        return obs, act, next_obs, rew, done

    @torch.no_grad()
    def evaluate(self) -> float:
        total = 0.0
        for _ in range(self.eval_episodes):
            o, _ = self.eval_env.reset()
            done = False
            trunc = False
            ep_ret = 0.0
            while not (done or trunc):
                obs = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
                if self.normalize_obs:
                    obs = self.obs_rms.normalize(obs)
                actor_z, _ = self.encoder(obs)
                # deterministic
                mu, _ = self.actor(actor_z)
                a = torch.tanh(mu)
                a_env = ((a + 1) / 2) * (self.high - self.low) + self.low
                o, r, done, trunc, _ = self.eval_env.step(a_env.squeeze(0).cpu().numpy())
                ep_ret += float(r)
            total += ep_ret
        return total / self.eval_episodes

    # ---------------- SAC loss on detached encoder outputs ----------------
    def _sac_update(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor,
                    rew: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rew = rew.view(-1, 1)
        done = done.view(-1, 1)
        with torch.no_grad():
            next_actor_z, next_critic_z = self.encoder_t(next_obs)
            next_a, next_logp, _ = self.actor_t.sample(next_actor_z)
            q1_t, q2_t = self.critic_t(next_critic_z, next_a)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            backup = rew + (1.0 - done) * self.gamma * q_t

        # encoder is NOT updated by SAC (matches original)
        with torch.no_grad():
            actor_z, critic_z = self.encoder(obs)
        q1, q2 = self.critic(critic_z, act)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        with torch.no_grad():
            actor_z_pi, critic_z_pi = self.encoder(obs)
        a_pi, logp_pi, _ = self.actor.sample(actor_z_pi)
        q1_pi, q2_pi = self.critic(critic_z_pi, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * logp_pi - q_pi).mean()

        entropy_diffs = (-logp_pi - self.target_entropy)
        alpha_loss = (self.log_alpha * entropy_diffs.detach()).mean()

        self.net_opt.zero_grad(set_to_none=True)
        (0.5 * critic_loss + policy_loss).backward()
        self.net_opt.step()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        return critic_loss.detach(), policy_loss.detach(), alpha_loss.detach()

    # ---------------- reward + dynamics losses (no encoder gradient) ----------------
    def _dynamics_update(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor,
                         rew: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rew = rew.view(-1, 1)
        with torch.no_grad():
            actor_z, _ = self.encoder(obs)            # fixed_encoded_states in JAX
            next_actor_z_t, _ = self.encoder_t(next_obs)  # encoded_next_states in JAX

        pred_r = self.reward_model(actor_z)
        reward_loss = F.mse_loss(pred_r, rew)

        pred_sample, _, _ = self._dynamics_sample(actor_z, act)
        dynamics_loss = F.mse_loss(pred_sample, next_actor_z_t)

        self.reward_opt.zero_grad(set_to_none=True)
        reward_loss.backward()
        self.reward_opt.step()

        self.dynamics_opt.zero_grad(set_to_none=True)
        dynamics_loss.backward()
        self.dynamics_opt.step()

        return reward_loss.detach(), dynamics_loss.detach()

    # implemented by subclasses
    def _dynamics_sample(self, actor_z: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    # ---------------- bisimulation loss (encoder-only update) ----------------
    def _bisim_update(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor,
                      rew: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        idx = torch.arange(B, device=self.device)
        shuffled = torch.randperm(B, device=self.device)  # JAX uses RNG for DBC shuffling
        # In the original DBC code, shuffled_idx comes from a PRNG key split.

        _, critic_z = self.encoder(obs)
        shuffled_z = critic_z[shuffled]

        if self.use_mico:
            # online_dist = (||z||²+||z'||²)/2 + 0.1 * angular_distance(z,z')
            norm_avg = 0.5 * (critic_z.pow(2).sum(dim=-1) + shuffled_z.pow(2).sum(dim=-1))
            online_dist = norm_avg + 0.1 * _angular_distance(critic_z, shuffled_z)
            with torch.no_grad():
                _, next_critic_z = self.encoder(next_obs)
                reward_diffs = (rew.squeeze(-1) - rew[shuffled].squeeze(-1)).abs()
                next_state_dist = _angular_distance(next_critic_z, next_critic_z[shuffled])
                target_dist = reward_diffs + self.gamma * next_state_dist
        else:
            online_dist = _l1(critic_z, shuffled_z)
            with torch.no_grad():
                actor_z, _ = self.encoder(obs)  # fixed_encoded_states
                # predicted_dynamics is computed with fixed encoder output, and
                # treated as a constant when updating the encoder.
                mu, sigma = self._dynamics_forward_no_grad(actor_z, act)
                reward_diffs = (rew.squeeze(-1) - rew[shuffled].squeeze(-1)).abs()
                mu_diffs = _l2(mu, mu[shuffled])
                sigma_diffs = _l2(sigma, sigma[shuffled])
                target_dist = reward_diffs + self.gamma * (mu_diffs + sigma_diffs)

        target_dist = target_dist.detach()

        if self.use_mico:
            bisim_loss = F.huber_loss(online_dist, target_dist, reduction="mean", delta=1.0)
        else:
            bisim_loss = (online_dist - target_dist).pow(2).mean()

        self.encoder_opt.zero_grad(set_to_none=True)
        bisim_loss.backward()
        self.encoder_opt.step()

        return bisim_loss.detach()

    @torch.no_grad()
    def _dynamics_forward_no_grad(self, actor_z: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sigma = self.dynamics_model(actor_z, act)
        return mu, sigma

    def _target_update(self):
        polyak_update(self.actor, self.actor_t, self.tau)
        polyak_update(self.critic, self.critic_t, self.tau)
        polyak_update(self.encoder, self.encoder_t, self.tau)

    def train(self):
        o, _ = self.env.reset()

        while self.steps < self.total_steps:
            obs_t = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
            if self.normalize_obs:
                self.obs_rms.update(obs_t)
                obs_norm = self.obs_rms.normalize(obs_t)
            else:
                obs_norm = obs_t

            if self.steps < self.warmup_steps:
                a_env = self.env.action_space.sample()
                a = torch.as_tensor(a_env, device=self.device, dtype=torch.float32).unsqueeze(0)
                a = 2 * (a - self.low) / (self.high - self.low) - 1
                a_env_np = a_env
            else:
                with torch.no_grad():
                    actor_z, _ = self.encoder(obs_norm)
                    a, _, _ = self.actor.sample(actor_z)
                    if self.noise_std > 0:
                        a = (a + self.noise_std * torch.randn_like(a)).clamp(-1.0, 1.0)
                    a_env = ((a + 1) / 2) * (self.high - self.low) + self.low
                a_env_np = a_env.squeeze(0).cpu().numpy()

            o2, r, done, trunc, _ = self.env.step(a_env_np)
            done_f = float(done or trunc)

            obs2_t = torch.as_tensor(o2, device=self.device, dtype=torch.float32).unsqueeze(0)
            # Store transition with obs, next_obs, normalized action, scalar reward/done
            self.replay.add(
                obs_t.squeeze(0),
                obs2_t.squeeze(0),
                a.squeeze(0),
                float(r),
                float(done_f),
            )
            self.steps += 1

            # episode reset
            if done or trunc:
                o, _ = self.env.reset()
            else:
                o = o2

            if self.steps >= max(self.warmup_steps, self.batch_size):
                obs_b, act_b, next_obs_b, rew_b, done_b = self._sample_batch()

                sac_critic_loss, sac_policy_loss, alpha_loss = self._sac_update(
                    obs_b, act_b, next_obs_b, rew_b, done_b
                )
                reward_loss, dynamics_loss = self._dynamics_update(
                    obs_b, act_b, next_obs_b, rew_b
                )
                bisim_loss = self._bisim_update(
                    obs_b, act_b, next_obs_b, rew_b
                )

                self._target_update()

                if wandb.run is not None:
                    wandb.log({
                        "train/sac_critic_loss": float(sac_critic_loss.item()),
                        "train/sac_policy_loss": float(sac_policy_loss.item()),
                        "train/alpha_loss": float(alpha_loss.item()),
                        "train/reward_loss": float(reward_loss.item()),
                        "train/dynamics_loss": float(dynamics_loss.item()),
                        "train/bisim_loss": float(bisim_loss.item()),
                        "train/alpha": float(self.alpha.item()),
                    }, step=self.steps)

            if self.steps % self.eval_freq == 0:
                avg = self.evaluate()
                print(f"[Eval] step={self.steps} avg_return={avg:.2f}")
                if wandb.run is not None:
                    wandb.log({"eval/return": avg}, step=self.steps)


class DBCAgent(_DBCBase):
    """Stochastic DBC (mu+sigma) as in the provided `dbc_agent.py.py` source."""
    def _make_dynamics(self, z_dim: int, act_dim: int, hidden: int) -> nn.Module:
        return DynamicsModel(z_dim=z_dim, act_dim=act_dim, hidden=hidden)

    def _dynamics_sample(self, actor_z: torch.Tensor, act: torch.Tensor):
        return self.dynamics_model.sample(actor_z, act)


class DBCDeterministicAgent(_DBCBase):
    """
    Deterministic DBC variant: sigma is zero and the dynamics sample is mu.
    This mirrors the common "DBC-Det" ablation (no stochasticity in dynamics).
    """
    def _make_dynamics(self, z_dim: int, act_dim: int, hidden: int) -> nn.Module:
        # Reuse DynamicsModel but ignore sigma head in sampling
        return DynamicsModel(z_dim=z_dim, act_dim=act_dim, hidden=hidden)

    def _dynamics_sample(self, actor_z: torch.Tensor, act: torch.Tensor):
        mu, sigma = self.dynamics_model(actor_z, act)
        sample = mu  # deterministic
        # sigma kept for target distance structure; callers may ignore it
        return sample, mu, torch.zeros_like(sigma)

    @torch.no_grad()
    def _dynamics_forward_no_grad(self, actor_z: torch.Tensor, act: torch.Tensor):
        mu, _sigma = self.dynamics_model(actor_z, act)
        return mu, torch.zeros_like(mu)

