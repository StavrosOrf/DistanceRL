
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from dist_rl.utils import RolloutBuffer, RunningMeanStd, polyak_update


# ---------------------------------------------------------------------
# MICo metric utilities (matches the NeurIPS'21 MICo paper and the
# original JAX MetricSACAgent logic: U_ω(x,y) = (||φ(x)||²+||φ(y)||²)/2
# + β * θ(φ(x),φ(y)), with θ the angular distance (arccos cos-sim).
# Pairing (x,y) is done by a deterministic batch roll, as in the JAX
# implementation which does not use extra RNG keys for shuffling.
# ---------------------------------------------------------------------

def _angular_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_norm = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
    y_norm = y / (y.norm(p=2, dim=-1, keepdim=True) + eps)
    cos = (x_norm * y_norm).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos)  # (B,)

def _representation_distances(phi_x: torch.Tensor,
                              phi_y: torch.Tensor,
                              beta: float = 0.1) -> torch.Tensor:
    # U_ω(x,y) per-sample, shape (B,)
    norm_avg = 0.5 * (phi_x.pow(2).sum(dim=-1) + phi_y.pow(2).sum(dim=-1))
    return norm_avg + beta * _angular_distance(phi_x, phi_y)

def _target_distances(phi_next: torch.Tensor,
                      rewards: torch.Tensor,
                      cumulative_gamma: float,
                      beta: float = 0.1) -> torch.Tensor:
    # Target: |r_x-r_y| + γ U_{\bar{ω}}(x',y')
    B = phi_next.shape[0]
    idx = torch.arange(B, device=phi_next.device)
    shuffled = torch.roll(idx, shifts=-1, dims=0)
    r_diff = (rewards.squeeze(-1) - rewards[shuffled].squeeze(-1)).abs()
    u_next = _representation_distances(phi_next, phi_next[shuffled], beta=beta)
    return r_diff + cumulative_gamma * u_next


# ---------------------------------------------------------------------
# Networks (vector observations)
# The structure mirrors SACConvNetwork used by the original JAX code:
# - Encoder produces (actor_z, critic_z)
# - Actor consumes actor_z
# - Critic consumes critic_z and action, and returns (q1,q2, critic_z)
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
        # logπ(a|s) with tanh correction
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


@dataclass
class MICoLossInfo:
    critic_loss: float
    policy_loss: float
    alpha_loss: float
    mico_loss: float
    total_loss: float


class MICoAgent:
    """
    PyTorch port of the original JAX `MetricSACAgent` (MICo + SAC) logic.

    Key points that match the source:
    - MICo uses Huber loss between online U_ω(x,y) and target |r_x-r_y|+γ U_{\\bar{ω}}(x',y')
    - Online pairing uses deterministic batch roll (no RNG needed)
    - The 'y' side in U_ω(x,y) uses a frozen (detached) representation (target_r)
    - Total loss: (1-mico_weight)*SAC_loss + mico_weight*MICo_loss
      where SAC_loss = 0.5*critic_loss + 1.0*policy_loss + 1.0*alpha_loss
      and alpha_loss uses stop-gradient on entropy differences
    """

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
                 mico_weight: float = 1e-5,
                 mico_beta: float = 0.1,
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

        # MICo params
        self.mico_weight = float(mico_weight)
        self.mico_beta = float(mico_beta)

        # Entropy
        self.target_entropy = -float(target_entropy_scale) * float(self.act_dim)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # Networks
        self.encoder = Encoder(self.obs_dim, z_dim=z_dim, hidden=hidden_size).to(self.device)
        self.actor = Actor(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)
        self.critic = Critic(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)

        # Target networks (entire network in the JAX implementation)
        self.encoder_t = Encoder(self.obs_dim, z_dim=z_dim, hidden=hidden_size).to(self.device)
        self.actor_t = Actor(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)
        self.critic_t = Critic(z_dim=z_dim, act_dim=self.act_dim, hidden=hidden_size).to(self.device)
        self.encoder_t.load_state_dict(self.encoder.state_dict())
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.net_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )

        # Replay buffer
        self.replay = RolloutBuffer(buffer_size, self.obs_dim, self.act_dim, device=self.device)

        self.steps = 0
        self.best_eval = -float("inf")

        if log_to_wandb and wandb.run is not None:
            wandb.run.log_code(".")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def _maybe_norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_obs:
            return obs
        self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    @torch.no_grad()
    def _act(self, obs: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        obs = obs.to(self.device)
        if self.normalize_obs:
            obs = self.obs_rms.normalize(obs)
        actor_z, _ = self.encoder(obs)
        if stochastic:
            a, _, _ = self.actor.sample(actor_z)
        else:
            mu, _ = self.actor(actor_z)
            a = torch.tanh(mu)
        return a

    @torch.no_grad()
    def evaluate(self) -> float:
        total = 0.0
        lengths = []
        for _ in range(self.eval_episodes):
            o, _ = self.eval_env.reset()
            done = False
            trunc = False
            ep_ret = 0.0
            L = 0
            while not (done or trunc):
                obs = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
                a = self._act(obs, stochastic=False)
                a_env = ((a + 1) / 2) * (self.high - self.low) + self.low
                o, r, done, trunc, _ = self.eval_env.step(a_env.squeeze(0).cpu().numpy())
                ep_ret += float(r)
                L += 1
            total += ep_ret
            lengths.append(L)
        avg = total / self.eval_episodes
        if avg > self.best_eval:
            self.best_eval = avg
        return avg

    def _sample_batch(self):
        obs, next_obs, act, rew, done = self.replay.sample(self.batch_size)
        if self.normalize_obs:
            obs = self.obs_rms.normalize(obs)
            next_obs = self.obs_rms.normalize(next_obs)
        return obs, act, next_obs, rew, done

    def _update(self) -> MICoLossInfo:
        obs, act, next_obs, rew, done = self._sample_batch()
        rew = rew.view(-1, 1)
        done = done.view(-1, 1)
        B = obs.shape[0]

        # ---- Critic target (target network) ----
        with torch.no_grad():
            next_actor_z, next_critic_z = self.encoder_t(next_obs)
            next_a, next_logp, _ = self.actor_t.sample(next_actor_z)
            q1_t, q2_t = self.critic_t(next_critic_z, next_a)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            backup = rew + (1.0 - done) * self.gamma * q_t

        # ---- Critic loss (online) ----
        actor_z, critic_z = self.encoder(obs)
        q1, q2 = self.critic(critic_z, act)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        # ---- Policy loss (online actor, frozen critic features & weights) ----
        a_pi, logp_pi, _ = self.actor.sample(actor_z)
        # freeze critic side exactly like JAX: treat critic network as constant.
        with torch.no_grad():
            # critic_z_frozen corresponds to encoding.critic_z computed under frozen params
            _, critic_z_frozen = self.encoder(obs)
        q1_pi, q2_pi = self.critic(critic_z_frozen, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * logp_pi - q_pi).mean()

        # ---- Alpha loss (stop-grad entropy diffs) ----
        entropy_diffs = (-logp_pi - self.target_entropy)
        alpha_loss = (self.log_alpha * entropy_diffs.detach()).mean()

        # ---- MICo loss ----
        # representations = critic_z from critic_online(state, action)
        # target_r = critic_z from frozen_critic_online(state, sampled_action) (stop-grad)
        with torch.no_grad():
            _, target_r = self.encoder(obs)
        # Pair (x,y) via deterministic roll (as in metric_utils)
        idx = torch.arange(B, device=self.device)
        shuf = torch.roll(idx, shifts=-1, dims=0)
        online_dist = _representation_distances(critic_z, target_r[shuf], beta=self.mico_beta)

        with torch.no_grad():
            _, target_next_r = self.encoder_t(next_obs)
            target_dist = _target_distances(target_next_r, rew, cumulative_gamma=self.gamma, beta=self.mico_beta)

        mico_loss = F.huber_loss(online_dist, target_dist, reduction="mean", delta=1.0)

        # ---- Combined loss (SAC terms exclude alpha; alpha optimized separately) ----
        sac_loss = 0.5 * critic_loss + 1.0 * policy_loss
        total_loss = (1.0 - self.mico_weight) * sac_loss + self.mico_weight * mico_loss

        self.net_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.net_opt.step()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # target update (EMA)
        polyak_update(self.encoder, self.encoder_t, self.tau)
        polyak_update(self.actor, self.actor_t, self.tau)
        polyak_update(self.critic, self.critic_t, self.tau)

        return MICoLossInfo(
            critic_loss=float(critic_loss.item()),
            policy_loss=float(policy_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            mico_loss=float(mico_loss.item()),
            total_loss=float(total_loss.item()),
        )

    def train(self):
        o, _ = self.env.reset()
        ep_ret = 0.0
        ep_len = 0

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
                # map env action to [-1,1] space for storage/critic
                a = 2 * (a - self.low) / (self.high - self.low) - 1
            else:
                a = self._act(obs_norm, stochastic=True)
                if self.noise_std > 0:
                    a = (a + self.noise_std * torch.randn_like(a)).clamp(-1.0, 1.0)
                a_env = ((a + 1) / 2) * (self.high - self.low) + self.low
                a_env = a_env.squeeze(0).cpu().numpy()

            o2, r, done, trunc, _ = self.env.step(a_env)
            done_f = float(done or trunc)

            obs2_t = torch.as_tensor(o2, device=self.device, dtype=torch.float32).unsqueeze(0)
            self.replay.add(
                obs_t.squeeze(0),
                obs2_t.squeeze(0),
                a.squeeze(0),
                float(r),
                float(done_f),
            )

            ep_ret += float(r)
            ep_len += 1
            self.steps += 1

            if done or trunc:
                o, _ = self.env.reset()
                ep_ret = 0.0
                ep_len = 0
            else:
                o = o2

            # updates
            if self.steps >= max(self.warmup_steps, self.batch_size):
                info = self._update()
                if wandb.run is not None:
                    wandb.log({
                        "train/critic_loss": info.critic_loss,
                        "train/policy_loss": info.policy_loss,
                        "train/alpha_loss": info.alpha_loss,
                        "train/mico_loss": info.mico_loss,
                        "train/total_loss": info.total_loss,
                        "train/alpha": float(self.alpha.item()),
                    }, step=self.steps)

            # eval
            if self.steps % self.eval_freq == 0:
                avg = self.evaluate()
                print(f"[Eval] step={self.steps} avg_return={avg:.2f}")
                if wandb.run is not None:
                    wandb.log({"eval/return": avg}, step=self.steps)
