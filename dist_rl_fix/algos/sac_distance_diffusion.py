# dist_rl_fix/algos/sac_distance.py  (updated)
from typing import Optional
import gymnasium as gym
import torch
import wandb
import math
import torch.nn.functional as F

from dist_rl_fix.models.networks import DistanceTrunk, GaussianActor, TwinQ
from dist_rl.utils import RolloutBuffer, differentiable_topk
from dist_rl_fix.representations import recursive_nstep_cosine_loss_ema, BetaEMA
from dist_rl_fix.utils import (RunningMeanStd,
                               polyak_update)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateCond(nn.Module):
    def __init__(self, obs_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 2*out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*out_dim, out_dim)
        )

    def forward(self, obs):  # (B,D) -> (B,out_dim)
        return self.net(obs)


class LatentDenoiser(nn.Module):
    """
    g_psi(s_emb, z_noisy) -> z_hat  (both on the unit sphere)
    """

    def __init__(self, z_dim: int, s_dim: int, hidden: int = 2*256):
        super().__init__()
        in_dim = z_dim + (s_dim if s_dim > 0 else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, z_noisy, s_emb=None):
        if s_emb is not None:
            if s_emb.dim() == 2 and z_noisy.dim() == 3:
                s_emb = s_emb.unsqueeze(1).expand(
                    z_noisy.size(0), z_noisy.size(1), s_emb.size(-1))
            x = torch.cat([z_noisy, s_emb], dim=-1)
        else:
            x = z_noisy
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class SACDistanceDiffusionAgent:
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
                 K: int,
                 expl_sigma: float,
                 updates_per_step: int,
                 kernel_aux_weight: float,
                 kernel_temp: float,
                 kernel_cand: int,
                 kernel_state_k: int,
                 kernel_adaptive_tau: bool,
                 target_entropy_scale: float,
                 rep_loss_weight: float,
                 rep_gamma_shape: float,
                 rep_lam: float,
                 rep_huber: float,
                 alpha: Optional[float],
                 save_dir: str,
                 **kwargs):

        # === extra knobs (safe defaults) ===
        self.target_entropy_scale = target_entropy_scale
        self.updates_per_step = updates_per_step
        # optional kernel auxiliary to actor (improves early training)
        self.kernel_aux_weight = kernel_aux_weight
        self.kernel_temp = kernel_temp
        self.kernel_cand = kernel_cand
        self.kernel_state_k = kernel_state_k
        self.kernel_adaptive_tau = kernel_adaptive_tau

        self.K = K  # for in-state Qhat
        self.noise_std = expl_sigma

        self.device = device
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        self.env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 1)

        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])
        self.low = torch.as_tensor(
            self.env.action_space.low, device=self.device, dtype=torch.float32)
        self.high = torch.as_tensor(
            self.env.action_space.high, device=self.device, dtype=torch.float32)

        self.total_steps = total_steps
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.save_dir = save_dir

        # === Modules ===
        self.actor = GaussianActor(
            self.obs_dim, self.act_dim, hidden=hidden_size).to(self.device)

        # (A) Critics with separate encoders (no sharing with rep trunk)
        self.qnet = TwinQ(self.obs_dim, self.act_dim,
                          hidden=hidden_size).to(self.device)
        self.q_targ = TwinQ(self.obs_dim, self.act_dim,
                            hidden=hidden_size).to(self.device)
        self.q_targ.load_state_dict(self.qnet.state_dict())

        # (B) Representation trunk (and target) used ONLY for rep-loss / kernel similarity
        self.rep_trunk = DistanceTrunk(
            self.obs_dim, self.act_dim, hidden=hidden_size).to(self.device)
        self.rep_trunk_targ = DistanceTrunk(
            self.obs_dim, self.act_dim, hidden=hidden_size).to(self.device)
        self.rep_trunk_targ.load_state_dict(self.rep_trunk.state_dict())

        # ---- Infer latent dim H once (probe rep_trunk) ----
        with torch.no_grad():
            _o = torch.zeros(1, self.obs_dim, device=self.device)
            _a = torch.zeros(1, self.act_dim, device=self.device)
            _z = self.rep_trunk(_o, _a)
            self.z_dim = int(_z.shape[1])

        # ---- Denoiser + conditioner ----
        self.s_cond_dim = min(64, max(16, self.obs_dim // 2))  # small, robust
        self.state_cond = StateCond(
            self.obs_dim, self.s_cond_dim).to(self.device)
        self.denoiser = LatentDenoiser(
            self.z_dim, self.s_cond_dim, hidden=2*hidden_size).to(self.device)

        self.optim_dnsr = torch.optim.Adam(
            list(self.denoiser.parameters()) +
            list(self.state_cond.parameters()),
            lr=3e-4
        )
        self.denoiser_sigma = expl_sigma  # latent noise std (fixed, robust)
        self.denoiser_tau_q = 1.0    # temperature for Q-weights to train g_psi
        # denoiser updates per learner step (cheap)
        self.denoiser_steps = 1
        self.max_dnsr_grad_norm = 5.0

        # === Optimizers ===
        self.optim_actor = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)
        self.optim_q = torch.optim.Adam(
            self.qnet.parameters(), lr=3e-4, weight_decay=1e-4)
        self.optim_rep = torch.optim.Adam(
            self.rep_trunk.parameters(), lr=self.lr)

        # === Temperature alpha ===

        self.target_entropy = - \
            self.target_entropy_scale * float(self.act_dim)
        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self._alpha_fixed = False

        # Replay & normalization
        self.replay = RolloutBuffer(
            buffer_size, self.obs_dim, self.act_dim, device=self.device)
        self.obs_rms = RunningMeanStd(self.obs_dim, device=self.device)

        # Representation loss
        self.rep_loss_weight = rep_loss_weight
        self.rep_gamma_shape = rep_gamma_shape
        self.rep_lam = rep_lam
        self.rep_huber = rep_huber
        self.beta_ema = BetaEMA(decay=0.995)

        # self.alpha_min = 0.02                    # floor to keep some exploration
        self.max_grad_norm = 5
        self.steps = 0
        self.warmup_steps = 5000
        self.best_eval = -float('inf')
        self._printed_warmup_notice = False

        if wandb.run is not None:
            wandb.run.log_code(".")

        print("[Init] SACDistanceAgent setup complete")
        print(
            f"[Init] env={env_id}, device={device}, total_steps={total_steps}, batch_size={batch_size}, buffer_size={buffer_size}")
        print(
            f"[Init] lr={lr}, gamma={gamma}, tau={tau}, rep_loss_weight={rep_loss_weight}, target_entropy={self.target_entropy:.2f}")
        print(f'Kernel aux weight: {self.kernel_aux_weight}, kernel temp: {self.kernel_temp}, kernel cand: {self.kernel_cand}, kernel state k: {self.kernel_state_k}, kernel adaptive tau: {self.kernel_adaptive_tau}')

    @property
    def alpha(self):
        return self.log_alpha.exp().item() if not self._alpha_fixed else self._alpha

    @alpha.setter
    def alpha(self, v):
        self._alpha = v

    # ---------- helpers ----------
    def _env_to_action(self, a):
        a = torch.as_tensor(a, device=self.device, dtype=torch.float32)
        return 2 * (a - self.low) / (self.high - self.low) - 1

    def _sample_action(self, obs_t: torch.Tensor, eval_mode=False):
        if eval_mode:
            mu, _ = self.actor.forward(obs_t)
            return torch.tanh(mu)
        a, _, _ = self.actor.sample(obs_t)
        return a

    def evaluate(self):
        total = 0.0
        lengths = []
        for ep in range(self.eval_episodes):
            o, _ = self.eval_env.reset()
            done = False
            trunc = False
            L = 0
            while not (done or trunc):
                ot = torch.as_tensor(o, device=self.device,
                                     dtype=torch.float32).unsqueeze(0)
                otn = self.obs_rms.normalize(ot)
                a = self._sample_action(otn, eval_mode=True).clamp(-1, 1)
                a_env = ((a + 1) / 2) * (self.high - self.low) + self.low
                o, r, done, trunc, _ = self.eval_env.step(
                    a_env.squeeze(0).detach().cpu().numpy())
                total += r
                L += 1
            lengths.append(L)
        avg = total / self.eval_episodes
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0

        if self.best_eval < avg:
            self.best_eval = avg
            print(
                f"[Eval] New best! avg_return={avg:.2f}, avg_length={avg_len:.1f}")
            self._save("best")

        print(f"[Eval] avg_return={avg:.2f}, avg_length={avg_len:.1f}")
        if wandb.run is not None:
            wandb.log({"eval/avg_reward": avg, "eval/avg_len": avg_len,
                      "step": self.steps}, step=self.steps)
        return avg

    # ---------- critic / actor / rep losses ----------
    def _td_targets(self, r, d, next_obs):
        with torch.no_grad():
            next_obs_n = self.obs_rms.normalize(next_obs)
            a2, logp2, _ = self.actor.sample(next_obs_n)
            q1t, q2t = self.q_targ(next_obs_n, a2)
            qt = torch.min(q1t, q2t)
            alpha = self.alpha if self._alpha_fixed else self.log_alpha.exp()

            target = r.unsqueeze(-1) + self.gamma * \
                (1 - d.unsqueeze(-1)) * (qt - alpha * logp2)
        return target

    def _q_loss(self, obs, act_env, r, next_obs, d):
        obs_n = self.obs_rms.normalize(obs)
        act = self._env_to_action(act_env)
        q1, q2 = self.qnet(obs_n, act)
        target = self._td_targets(r, d, next_obs)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        return loss_q, {"train/q_loss_raw": float(loss_q.item())}
    
    def _rep_loss(self, obs, act_env, next_obs, done):
        # Use rep trunk and TARGET rep trunk (stop-grad on next)
        obs_n = self.obs_rms.normalize(obs)
        next_obs_n = self.obs_rms.normalize(next_obs)

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():  # stop-grad target
            a2, _, _ = self.actor.sample(next_obs_n)
            # add noise to a2
            a2 += torch.randn_like(a2) * 0.2
            a2 = a2.clamp(-1, 1)
            z_next = self.rep_trunk_targ(next_obs_n, a2)

            q1t, q2t = self.q_targ(obs_n, act)
            q_targ = torch.min(q1t, q2t).squeeze(-1)  # (B,)

        loss, info = recursive_nstep_cosine_loss_ema(
            z, z_next, done, q_targ,
            discount=self.gamma,
            gamma_shape=self.rep_gamma_shape,
            lam=self.rep_lam,
            huber_delta=self.rep_huber,
            beta_ema=self.beta_ema
        )
        return loss, info

    def _denoiser_step(self, obs, K: int = 64, beta: float = 0.10):
        """
        Train g_psi to denoise *top-tail* (CVaR) latents toward high-Q latents.
        - beta: tail level (0.1 keeps top 10%)
        """
        eps = 1e-8
        device = self.device
        B = obs.size(0)

        with torch.no_grad():
            obs_n = self.obs_rms.normalize(obs)                      # (B,D)
            obs_rep = obs_n.repeat_interleave(K, dim=0)              # (B*K,D)

            a_k, _, _ = self.actor.sample(obs_rep)                   # (B*K,A)
            a_k = (a_k + 0.10 * torch.randn_like(a_k)).clamp(-1, 1)

            z_k = F.normalize(self.rep_trunk_targ(obs_rep, a_k), p=2, dim=1)
            z_k = z_k.view(B, K, -1)                                 # (B,K,H)

            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K)                      # (B,K)

            # ---- CVaR tail weights (stop-grad) ----
            # threshold u: top-beta tail
            u = torch.quantile(qk, q=1.0 - beta, dim=1,
                               keepdim=True)     # (B,1)
            # (B,K)
            v = torch.relu(qk - u)
            # small smoothing so we never get all zeros
            v = v + 1e-6
            # (B,K)
            w = v / v.sum(dim=1, keepdim=True)

            # noisy latents
            z_noisy = z_k + self.denoiser_sigma * torch.randn_like(z_k)
            z_noisy = F.normalize(z_noisy, p=2, dim=-1)

        obs_n = self.obs_rms.normalize(obs)
        s_emb = self.state_cond(obs_n)

        # (B,K,H)
        z_hat = self.denoiser(z_noisy, s_emb)
        cos = (z_hat * z_k).sum(dim=-1).clamp(-1.0,
                                              1.0)                  # (B,K)

        # minimize (1 - cosine) weighted by tail weights
        loss = (w * (1.0 - cos)).sum(dim=1).mean()

        self.optim_dnsr.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.denoiser.parameters()) +
            list(self.state_cond.parameters()),
            self.max_dnsr_grad_norm
        )
        self.optim_dnsr.step()

        if wandb.run is not None:
            wandb.log({"ldp/denoiser_loss": float(loss.item()),
                       "ldp/sigma": float(self.denoiser_sigma),
                       "ldp/cvar_beta": float(beta)}, step=self.steps)

    def _actor_loss_ldp(self,
                        obs,
                        K: int = 64,
                        pull_weight: float = 1.0,
                        beta_cvar: float = 0.10,      # CVaR level for both denoiser and jump
                        jump_thresh: float = 0.03,    # if 1 - cos < thresh, use a "jump"
                        # (sigma_max, sigma_min, T_anneal)
                        sigma_schedule: tuple = (0.30, 0.15, 200_000)
                        ):
        """
        Latent Diffusion Pull (aggressive):
        - trains denoiser on CVaR tail (top beta)
        - actor pull weight is amplified by the current misalignment (1 - cos)
        - when the denoiser pull is too small, jump to a CVaR barycenter in latent space
        """
        # ---- anneal sigma high -> low (bigger steps early, precise later) ----
        sig_max, sig_min, T = sigma_schedule
        p = min(1.0, float(self.steps) / float(T))
        # cosine anneal
        self.denoiser_sigma = sig_min + 0.5 * \
            (sig_max - sig_min) * (1.0 + math.cos(math.pi * p))

        # ---- a couple of denoiser steps on CVaR tail ----
        for _ in range(self.denoiser_steps):
            self._denoiser_step(obs, K=K, beta=beta_cvar)

        obs_n = self.obs_rms.normalize(obs)                          # (B,D)
        a_anchor, logp, _ = self.actor.sample(obs_n)
        z_i = F.normalize(self.rep_trunk(
            obs_n, a_anchor), p=2, dim=1)   # (B,H)

        # ---- denoised target of the anchor (stop-grad) ----
        with torch.no_grad():
            s_emb = self.state_cond(obs_n)
            z_noisy = F.normalize(
                z_i + self.denoiser_sigma * torch.randn_like(z_i), p=2, dim=-1)
            z_hat = self.denoiser(
                z_noisy, s_emb)                        # (B,H)

        # cosine gap and adaptive pull strength
        cos_i = (z_i * z_hat).sum(dim=-1).clamp(-1.0, 1.0)               # (B,)
        gap = (1.0 - cos_i).detach()                                     # (B,)
        # amplify pull when almost aligned (prevents tiny gradients)
        # pull_weight_eff in [pull_weight, ~2.5 * pull_weight]
        pull_weight_eff = pull_weight * \
            (1.0 + 1.5 * (gap / (gap.mean() + 1e-6))).clamp(1.0, 2.5).mean()

        # base pull loss
        pull_loss = pull_weight_eff * (1.0 - cos_i).mean()

        # ---- fallback: CVaR barycenter "jump" if pull is vanishing ----
        # condition: mean gap below threshold
        if gap.mean().item() < jump_thresh:
            with torch.no_grad():
                # reuse proposals to compute a CVaR latent barycenter
                B = obs_n.size(0)
                obs_rep = obs_n.repeat_interleave(K, dim=0)
                a_k, _, _ = self.actor.sample(obs_rep)
                a_k = (a_k + 0.10 * torch.randn_like(a_k)).clamp(-1, 1)

                z_k = F.normalize(self.rep_trunk_targ(
                    obs_rep, a_k), p=2, dim=1).view(B, K, -1)
                q1k, q2k = self.q_targ(obs_rep, a_k)
                qk = torch.min(q1k, q2k).view(B, K)

                u = torch.quantile(qk, q=1.0 - beta_cvar,
                                   dim=1, keepdim=True)  # (B,1)
                # (B,K)
                v = torch.relu(qk - u) + 1e-6
                v = v / v.sum(dim=1, keepdim=True)
                z_cvar = F.normalize(torch.einsum(
                    'bk,bkd->bd', v, z_k), p=2, dim=1)  # (B,H)

            # add a jump loss
            cos_jump = (z_i * z_cvar).sum(dim=-1).clamp(-1.0, 1.0)
            # scale jump to be comparable to pull
            jump_loss = pull_weight * (1.0 - cos_jump).mean()
            pull_loss = pull_loss + 0.5 * jump_loss  # blend to keep stability

            if wandb.run is not None:
                wandb.log({"ldp/jump_triggered": 1.0,
                           "ldp/jump_loss": float(jump_loss.item())}, step=self.steps)

        # ---- SAC entropy and alpha update ----
        alpha = self.log_alpha.exp()
        entropy_loss = (alpha * logp).mean()
        alpha_loss = -(self.log_alpha *
                       (logp + self.target_entropy).detach()).mean()

        actor_loss = entropy_loss + pull_loss

        if wandb.run is not None:
            wandb.log({
                "ldp/pull_loss": float(pull_loss.item()),
                "ldp/cos_anchor_hat": float(cos_i.mean().item()),
                "ldp/sigma": float(self.denoiser_sigma),
                "ldp/pull_weight_eff": float(pull_weight_eff.item() if torch.is_tensor(pull_weight_eff) else pull_weight_eff),
                "ldp/gap_mean": float(gap.mean().item())
            }, step=self.steps)

        return actor_loss, alpha_loss

    # ---------- training ----------

    def train(self):
        print(
            f"[Train] Starting SAC training for {self.total_steps} steps (eval every {self.eval_freq})")
        o, _ = self.env.reset(seed=None)
        ep_r = 0.0
        ep_len = 0

        while self.steps < self.total_steps:
            ot = torch.as_tensor(o, device=self.device,
                                 dtype=torch.float32).unsqueeze(0)
            self.obs_rms.update(ot)
            otn = self.obs_rms.normalize(ot)

            with torch.no_grad():
                if self.steps < self.warmup_steps:
                    a_env = self.env.action_space.sample()
                else:
                    a = self._sample_action(otn).clamp(-1, 1)
                    a_env = ((a + 1) / 2) * (self.high - self.low) + self.low
                    a_env = a_env.squeeze(0).cpu().numpy()

            o2, r, done, trunc, _ = self.env.step(a_env)
            self.replay.add(o, o2, a_env, r, done or trunc)

            ep_r += r
            ep_len += 1
            self.steps += 1
            o = o2

            if not self._printed_warmup_notice and self.steps >= self.warmup_steps:
                print(
                    f"[Train] Warmup complete at step {self.steps}, switching to policy actions")
                self._printed_warmup_notice = True

            if done or trunc:
                print(
                    f"[Train] Episode: return={ep_r:.2f}, length={ep_len}, total_steps={self.steps}")
                if wandb.run is not None:
                    wandb.log({"rollout/ep_reward": ep_r, "rollout/ep_len": ep_len,
                              "step": self.steps}, step=self.steps)
                ep_r = 0.0
                ep_len = 0
                o, _ = self.env.reset()

            # updates
            if self.steps < max(self.warmup_steps, self.batch_size):
                continue

            for _ in range(self.updates_per_step):
                obs, next_obs, act_env, rew, done_b = self.replay.get_batch(
                    self.batch_size)

                # ---- Q update (separate encoders) ----
                loss_q, qinfo = self._q_loss(
                    obs, act_env, rew, next_obs, done_b)
                self.optim_q.zero_grad()
                loss_q.backward()
                q_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.qnet.parameters(), self.max_grad_norm)
                self.optim_q.step()
                if wandb.run is not None:
                    wandb.log({**qinfo, "train/q_loss": float(loss_q.item()),
                               "train/q_grad_norm": float(q_grad_norm)}, step=self.steps)

                # ---- Representation loss (with target rep trunk) ----                
                rep_loss, rep_info = self._rep_loss(
                    obs, act_env, next_obs, done_b)
                self.optim_rep.zero_grad()
                rep_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.rep_trunk.parameters(), self.max_grad_norm)
                self.optim_rep.step()
                rep_logs = {f"rep/{k}": v for k, v in rep_info.items()}
                rep_logs.update(
                    {"rep/loss": float(rep_loss.item()), "step": self.steps})
                if wandb.run is not None:
                    wandb.log(rep_logs, step=self.steps)

                actor_loss, alpha_loss = self._actor_loss_ldp(
                    obs,
                    K=self.K,
                    pull_weight=1.5, # 1.0
                    beta_cvar=0.05, # 0.10
                    jump_thresh=0.03,
                    sigma_schedule=(0.30, 0.15, 200_000)
                )

                self.optim_actor.zero_grad()
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.max_grad_norm)
                self.optim_actor.step()

                logs = {
                    "train/actor_loss": float(actor_loss.item()),
                    "train/actor_grad_norm": float(actor_grad_norm),
                    "step": self.steps}
                if alpha_loss is not None:
                    self.alpha_opt.zero_grad()
                    alpha_loss.backward()
                    # monitor alpha value grad norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [self.log_alpha], self.max_grad_norm)
                    self.alpha_opt.step()
                    logs["train/alpha"] = float(self.log_alpha.exp().item())
                    logs["train/alpha_loss"] = float(alpha_loss.item())
                    # monitor alpha grad norm
                    logs["train/alpha_grad_norm"] = float(grad_norm)
                else:
                    logs["train/alpha"] = float(self.alpha)
                if wandb.run is not None:
                    wandb.log(logs, step=self.steps)

                # targets
                polyak_update(self.q_targ, self.qnet, self.tau)
                polyak_update(self.rep_trunk_targ, self.rep_trunk, self.tau)

            if (self.steps % self.eval_freq) == 0:
                print(f"[Train] Evaluation at step {self.steps}")
                self.evaluate()

    def _save(self, name: str):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"{name}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "qnet": self.qnet.state_dict(),
            "rep_trunk": self.rep_trunk.state_dict(),
            "normalization": self.obs_rms.state_dict(),
            "steps": self.steps
        }, path)

        print(f"[Save] Model saved to {path}")
