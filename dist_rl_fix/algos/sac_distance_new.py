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


class SACDistanceAgentNew:
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

    def _new_actor_alpha_loss(self, obs):
        
        instate = False  # choose in-state Qhat
        
        if instate:
            obs_n = self.obs_rms.normalize(obs)

            a, logp, _ = self.actor.sample(obs_n)

            # sample candidates
            obs_c, obs_c_next, act_env_c, ret_c, done_c = self.replay.get_batch(
                self.kernel_cand)
            obs_c_n = self.obs_rms.normalize(obs_c)
            act_c = self._env_to_action(act_env_c)

            # kernel similarities in rep space
            z_i = F.normalize(self.rep_trunk(obs_n, a), p=2,
                              dim=1)                 # (B,H)

            with torch.no_grad():
                z_c = F.normalize(
                    self.rep_trunk_targ(obs_c_n, act_c), p=2, dim=1)  # (B_c,H)

            S_full = z_i @ z_c.T                        # (B, B_c) cosine sim

            # Compute per-sample k based on cosine similarity threshold
            cos_sim_threshold = 0.85
            num_above_threshold = (
                S_full > cos_sim_threshold).sum(dim=1)  # [B]
            # print(f'num_above_threshold: {num_above_threshold}')
            k_per_sample = torch.clamp(
                num_above_threshold, min=5, max=self.kernel_state_k)  # [B]
            # print(f'k_per_sample: {k_per_sample}')

            S_masked, top_vals, top_idx = differentiable_topk(
                S_full, k_per_sample)

            # take top_vals without infinite values
            top_vals = top_vals[torch.isfinite(top_vals)]

            # adaptive tau per row
            # if self.kernel_adaptive_tau:
            W = torch.softmax(S_masked, dim=1)

            # targets: critics' Q rather than returns (much better bias)
            with torch.no_grad():
                qc1, qc2 = self.q_targ(obs_c_n, act_c)
                qc = torch.min(qc1, qc2).unsqueeze(0)

            # (B,1)
            Qhat = (W.unsqueeze(-1) * qc).sum(dim=1)
            # print(f'Qhat mean: {Qhat.mean().item()}')

        # in-state Qhat
        else:
            Qhat, logp = self._qhat_in_state(obs,
                                             K=32,
                                             noise_std=0.1,
                                             softmax_temp=1.0,
                                             eps=0.05)
            
            #Normalized in-state Qhat
            # Qhat, logp = self._qhat_in_state_norm(obs,
            #                                  K=32,
            #                                  noise_std=0.1,
            #                                  softmax_temp=1.0,
            #                                  eps=0.05)

        alpha = self.log_alpha.exp()

        entropy_loss = (alpha * logp).mean()

        actor_loss = entropy_loss - Qhat.mean()
        alpha_loss = -(self.log_alpha *
                       (logp + self.target_entropy).detach()).mean()

        logs = {
                # "kernel/top_state_sim_mean": float(top_vals.mean().item()),
                "kernel/aux_term": float(-Qhat.mean().item()),
                "train/actor_entropy_loss": float(entropy_loss.item())}

        if wandb.run is not None:
            wandb.log(logs, step=self.steps)

        return actor_loss, alpha_loss

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

    def _qhat_in_state(self, obs, K: int = 64, noise_std: float = 0.1,
                       softmax_temp: float = 1.0, eps: float = 0.05):

        obs_n = self.obs_rms.normalize(obs)                     # (B,D)
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)             # (B*K,D)

        # propose K actions per current state
        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)              # (B*K,A)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)

            z_k = F.normalize(self.rep_trunk_targ(
                obs_rep, a_k), p=2, dim=1)  # (B*K,H)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)              # (B,K,1)

        # anchor at current policy action
        a_anchor, logp, _ = self.actor.sample(obs_n)               # (B,A)
        z_i = F.normalize(self.rep_trunk(obs_n, a_anchor),
                          p=2, dim=1)         # (B,H)

        # cosine sims to K proposals for the *same* state
        z_k_view = z_k.view(B, K, -1)                           # (B,K,H)
        S = torch.einsum('bd,bkd->bk', z_i, z_k_view)           # (B,K)

        W = torch.softmax(S / softmax_temp, dim=1)              # (B,K)
        if eps > 0.0:                                           # keep gradients alive
            W = (1 - eps) * W + eps / K

        Qhat = (W.unsqueeze(-1) * qk).sum(dim=1)                # (B,1)
        return Qhat, logp

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                       softmax_temp: float = 1.0, eps: float = 0.05):

        obs_n = self.obs_rms.normalize(obs)                     # (B,D)
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)             # (B*K,D)

        # --- propose K actions per current state (stop-grad path for proposals) ---
        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)              # (B*K,A)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)

            z_k = F.normalize(self.rep_trunk_targ(
                obs_rep, a_k), p=2, dim=1)  # (B*K,H)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            # (B,K,1)   (stop-grad)
            qk = torch.min(q1k, q2k).view(B, K, 1)

        # --- anchor at current policy action (this branch carries gradients) ---
        a_anchor, logp, _ = self.actor.sample(obs_n)            # (B,A)
        z_i = F.normalize(self.rep_trunk(obs_n, a_anchor), p=2, dim=1)  # (B,H)

        # --- cosine sims to K proposals for the same state ---
        z_k_view = z_k.view(B, K, -1)                           # (B,K,H)
        S = torch.einsum('bd,bkd->bk', z_i, z_k_view).clamp(-1.0, 1.0)  # (B,K)

        # --- kernel bandwidth τ: cosine-anneal + optional adaptive from row std ---
        tau_row = self._kernel_tau_instate(
            S, base_temp=softmax_temp)    # (B,1)

        # --- softmax weights with τ (plus tiny ε smoothing for stability) ---
        W = torch.softmax(S / tau_row, dim=1)                   # (B,K)
        if eps > 0.0:
            W = (1.0 - eps) * W + eps / K

        # --- advantage-centering: q̃_k = q_k - \bar q  (baseline is stop-grad) ---
        q_bar = (W.unsqueeze(-1) * qk).sum(dim=1,
                                           keepdim=True).detach()  # (B,1,1)
        # (B,K,1)
        q_tilde = qk - q_bar

        # --- centered readout ---
        Qhat = (W.unsqueeze(-1) * q_tilde).sum(dim=1)            # (B,1)

        # (optional) diagnostics
        if wandb.run is not None:
            wandb.log({
                "kernel_instate/tau_mean": float(tau_row.mean().item()),
                "kernel_instate/top_sim_mean": float(S.max(dim=1).values.mean().item()),
                "kernel_instate/qbar_mean": float(q_bar.mean().item()),
                "kernel_instate/qtilde_abs_mean": float(q_tilde.abs().mean().item()),
            }, step=self.steps)

        return Qhat, logp

    def _kernel_tau_instate(self, S_rowwise: torch.Tensor, base_temp: float) -> torch.Tensor:
        """
        S_rowwise: (B, K) cosine sims for a single state across K proposals.
        Returns τ as (B,1). Uses cosine anneal τ_max->τ_min, plus optional adaptive bump from row std.
        """
        B = S_rowwise.size(0)
        device = S_rowwise.device

        # schedule window & bounds
        T_sched = 200_000
        # start wider than config
        tau_max = max(0.75, float(base_temp))
        tau_min = max(0.05, 0.30 * float(base_temp))   # end sharper

        p = min(1.0, float(self.steps) / float(T_sched))
        # cosine anneal: p=0 -> tau_max, p=1 -> tau_min
        tau_sched = tau_min + 0.5 * \
            (tau_max - tau_min) * (1.0 + math.cos(math.pi * p))

        if getattr(self, "kernel_adaptive_tau", False):
            row_std = S_rowwise.std(dim=1, keepdim=True).clamp(min=1e-4)
            tau_row = torch.full(
                (B, 1), tau_sched, device=device) + row_std  # c=1.0
        else:
            tau_row = torch.full((B, 1), tau_sched, device=device)

        return tau_row  # (B,1)

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
                if self.rep_loss_weight > 0.0 and self.kernel_aux_weight > 0.0:
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

                actor_loss, alpha_loss = self._new_actor_alpha_loss(obs)

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
