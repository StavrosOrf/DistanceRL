# dist_rl_fix/algos/sac_distance.py  (updated)
from typing import Optional
import gymnasium as gym
import torch
import wandb
import math
import torch.nn.functional as F

from dist_rl_fix.models.networks import DistanceTrunk, GaussianActor, TwinQ
from dist_rl.utils import RolloutBuffer
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
                 K: int,
                 expl_sigma: float,
                 target_entropy_scale: float,
                 updates_per_step: int,
                 kernel_adaptive_tau: int,
                 rep_gamma_shape: float,
                 rep_lam: float,
                 rep_huber: float,
                 normalize_obs: int,
                 warmup_steps: int,
                 alpha: Optional[float],
                 save_dir: str,
                 **kwargs):

        self.updates_per_step = updates_per_step
        self.kernel_adaptive_tau = True if kernel_adaptive_tau != 0 else False
        self.target_entropy_scale = target_entropy_scale

        self.K = K  # for in-state Qhat
        self.noise_std = expl_sigma

        self.normalize_obs = True if normalize_obs != 0 else False

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
        self.rep_gamma_shape = rep_gamma_shape
        self.rep_lam = rep_lam
        self.rep_huber = rep_huber
        self.beta_ema = BetaEMA(decay=0.995)

        # self.alpha_min = 0.02                    # floor to keep some exploration
        self.max_grad_norm = 5
        self.steps = 0
        self.warmup_steps = warmup_steps
        self.best_eval = -float('inf')
        self._printed_warmup_notice = False

        if wandb.run is not None:
            wandb.run.log_code(".")

        print("[Init] SACDistanceAgent setup complete")
        print(
            f"[Init] env={env_id}, device={device}, total_steps={total_steps}, batch_size={batch_size}, buffer_size={buffer_size}")        
        print(f'Normalize obs: {self.normalize_obs}')

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
                if self.normalize_obs:
                    otn = self.obs_rms.normalize(ot)
                else:
                    otn = ot
                    
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
            if self.normalize_obs:
                next_obs_n = self.obs_rms.normalize(next_obs)
            else:
                next_obs_n = next_obs
                
            a2, logp2, _ = self.actor.sample(next_obs_n)
            q1t, q2t = self.q_targ(next_obs_n, a2)
            qt = torch.min(q1t, q2t)
            alpha = self.alpha if self._alpha_fixed else self.log_alpha.exp()

            target = r.unsqueeze(-1) + self.gamma * \
                (1 - d.unsqueeze(-1)) * (qt - alpha * logp2)
        return target

    def _q_loss(self, obs, act_env, r, next_obs, d):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        act = self._env_to_action(act_env)
        q1, q2 = self.qnet(obs_n, act)
        target = self._td_targets(r, d, next_obs)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        return loss_q, {"train/q_loss_raw": float(loss_q.item())}

    def _new_actor_alpha_loss(self, obs):
     
        # Qhat, logp = self._qhat_in_state(obs,
        #                                  K=32,
        #                                  noise_std=0.1,
        #                                  softmax_temp=1.0,
        #                                  eps=0.05)

        # Normalized in-state Qhat
        Qhat, logp = self._qhat_in_state_norm(obs,
                                                K=self.K,
                                                noise_std=self.noise_std,
                                                softmax_temp=1.0,
                                                eps=0.05)

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
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
            next_obs_n = self.obs_rms.normalize(next_obs)
        else:
            obs_n = obs
            next_obs_n = next_obs

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():  # stop-grad target
            a2, _, _ = self.actor.sample(next_obs_n)
            # add noise to a2
            a2 += torch.randn_like(a2) * self.noise_std # used 0.2
            a2 = a2.clamp(-1, 1)
            a2 = ((a2 + 1) / 2) * (self.high - self.low) + self.low
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

        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)                     # (B,D)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)             # (B*K,D)

        # propose K actions per current state
        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)              # (B*K,A)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low

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

        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)                     # (B,D)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)             # (B*K,D)

        # --- propose K actions per current state (stop-grad path for proposals) ---
        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)              # (B*K,A)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low

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

    def _qhat_in_state_fast(self,
                            obs,
                            K: int = 64,
                            noise_std: float = 0.10,
                            base_temp: float = 1.0,
                            eps_w: float = 1e-3,
                            positive_tail: bool = True,
                            multi_tau: bool = True,
                            keff_target_frac: float = 0.4,   # target effective support ~ 0.4*Ktot
                            keff_min_frac: float = 0.2,      # clamps for safety
                            keff_max_frac: float = 0.6,
                            cem_elites: int = 8,             # CEM-lite elites per state
                            cem_new: int = 16,               # new samples per state
                            m_anchors: int = 1,              # multi-anchor variance reduction
                            bary_lambda: float = 0.2):       # tiny barycentric attraction
        """
        In-state Q̂ with stability + acceleration:
        - Proposals: K from π + small noise, then CEM-lite augments with 'cem_new'
        - Weights: softmax over cosine sims, adaptive τ (keff tracking), multi-τ average
        - Advantage: centered, optionally positive-tail only (ReLU), tiny clipping
        - Barycentric boost: -λ * < z_i , z_bar >  (z_bar stop-grad), improves step size
        - Optional M anchors averaged for variance reduction

        Returns:
        Qhat_mean: (B,1)
        logp_mean: (B,)  (mean over anchors)
        bary_loss: scalar (already with the correct sign to be added to actor loss)
        """
        device = self.device
        obs_n = self.obs_rms.normalize(obs)                       # (B,D)
        B = obs_n.size(0)

        # ---------- proposals (stop-grad path) ----------
        with torch.no_grad():
            # initial K samples from current policy
            obs_rep0 = obs_n.repeat_interleave(K, dim=0)          # (B*K,D)
            a0, _, _ = self.actor.sample(obs_rep0)                # (B*K,A)
            if noise_std > 0:
                a0 = (a0 + noise_std * torch.randn_like(a0)).clamp(-1, 1)
                a0 = ((a0 + 1) / 2) * (self.high - self.low) + self.low
                
            A0 = a0.view(B, K, self.act_dim)                      # (B,K,A)
            K0 = K

            # evaluate critic on initial proposals
            q1_0, q2_0 = self.q_targ(obs_rep0, a0)                # (B*K,1)
            qk0 = torch.min(q1_0, q2_0).view(B, K0)               # (B,K)

            # ---- CEM-lite refinement (cheap, stop-grad) ----
            m = min(cem_elites, K0) if cem_elites > 0 else 0
            if m > 0 and cem_new > 0:
                elite_idx = torch.topk(
                    qk0, k=m, dim=1).indices                       # (B,m)
                elite = torch.gather(
                    A0, 1, elite_idx.unsqueeze(-1).expand(-1, -1, self.act_dim))  # (B,m,A)
                # (B,1,A)
                mu = elite.mean(dim=1, keepdim=True)
                std = elite.std(dim=1, keepdim=True).clamp_min(
                    1e-3)                  # (B,1,A)
                A_cem = (mu + std * torch.randn(B, cem_new,
                         self.act_dim, device=device)).tanh()   # (B,cem_new,A)
                # (B,Ktot,A)
                A_all = torch.cat([A0, A_cem], dim=1)
            else:
                A_all = A0

            Ktot = A_all.size(1)
            obs_rep = obs_n.repeat_interleave(
                Ktot, dim=0)                             # (B*Ktot,D)
            A_all_flat = A_all.reshape(
                B * Ktot, self.act_dim)                         # (B*Ktot,A)

            # critic targets on augmented set
            # (B*Ktot,1)
            q1, q2 = self.q_targ(obs_rep, A_all_flat)
            # (B,Ktot,1)
            qk = torch.min(q1, q2).view(B, Ktot, 1)

            # representation targets of proposals
            z_k = F.normalize(self.rep_trunk_targ(
                obs_rep, A_all_flat), p=2, dim=1)    # (B*Ktot,H)
            # (B,Ktot,H)
            z_k = z_k.view(B, Ktot, -1)

        # ---------- adaptive τ via K_eff of similarity weights ----------
        # we adapt τ from S only (independent of q), using a persistent tracker
        if not hasattr(self, "_tau_instate"):
            self._tau_instate = torch.tensor(
                float(max(0.75, base_temp)), device=device)

        # choose an "anchor for τ adaptation": use current policy once
        a_tau, _, _ = self.actor.sample(
            obs_n)                                         # (B,A)
        z_tau = F.normalize(self.rep_trunk(obs_n, a_tau),
                            p=2, dim=1)                  # (B,H)

        # cosine similarities S (B,Ktot)
        S_tau = torch.einsum('bd,bkd->bk', z_tau, z_k).clamp(-1.0, 1.0)

        # multiplicative update to keep K_eff ≈ keff_target_frac*Ktot (within [keff_min, keff_max])
        eps = 1e-8

        def keff_from_tau(tau_scalar: torch.Tensor):
            Wtmp = torch.softmax(S_tau / (tau_scalar + eps),
                                 dim=1)                    # (B,Ktot)
            # (B,)
            keff = 1.0 / (Wtmp.pow(2).sum(dim=1) + eps)
            return Wtmp, keff

        Wtmp, keff = keff_from_tau(self._tau_instate)
        tgt = float(max(keff_min_frac * Ktot,
                    min(keff_max_frac * Ktot, keff_target_frac * Ktot)))
        err = (keff.mean().item() - tgt) / max(tgt, 1.0)
        # small, stable correction; clamp τ to reasonable range
        self._tau_instate = (self._tau_instate *
                             math.exp(0.35 * (-err))).clamp(0.1, 5.0)

        # ---------- multi-anchor pass (M times) ----------
        Qhat_list = []
        logp_list = []
        bary_terms = []

        for _ in range(max(1, m_anchors)):
            # anchor (this branch carries gradients)
            a_anchor, logp, _ = self.actor.sample(
                obs_n)                                # (B,A)
            z_i = F.normalize(self.rep_trunk(obs_n, a_anchor),
                              p=2, dim=1)              # (B,H)
            # (B,)
            logp_list.append(logp)

            # cosine sims S for this anchor
            S = torch.einsum('bd,bkd->bk', z_i, z_k).clamp(-1.0,
                                                           1.0)                   # (B,Ktot)

            # weights W with τ; optional multi-τ averaging
            if multi_tau:
                taus = [self._tau_instate, 2.0 *
                        self._tau_instate, 4.0 * self._tau_instate]
                W_stack = [torch.softmax(S / (t + eps), dim=1)
                           for t in taus]           # list of (B,Ktot)
                W = torch.stack(W_stack, dim=0).mean(
                    dim=0)                             # (B,Ktot)
            else:
                W = torch.softmax(S / (self._tau_instate + eps),
                                  dim=1)                 # (B,Ktot)

            # tiny ε-smoothing to avoid zero-weights
            W = (1.0 - eps_w) * W + eps_w / Ktot

            # robust centering baseline
            with torch.no_grad():
                # (B,Ktot)
                qk_flat = qk.squeeze(-1)
                # mean + median baseline (robust to outliers)
                # (B,1)
                q_mean = (W * qk_flat).sum(dim=1, keepdim=True)
                q_med = qk_flat.median(
                    dim=1, keepdim=True).values                     # (B,1)
                # (B,1)
                q_bar = 0.5 * (q_mean + q_med)
            # centered (B,Ktot,1)
            # (B,Ktot,1)
            q_tilde = qk - q_bar.unsqueeze(-1)

            if positive_tail:
                # keep only positive tail
                q_tilde = torch.relu(q_tilde)

            # mild clipping to stabilize huge early spikes
            q_tilde = q_tilde.clamp(min=0.0, max=10.0)

            # centered readout
            # (B,1)
            Qhat = (W.unsqueeze(-1) * q_tilde).sum(dim=1)
            Qhat_list.append(Qhat)

            # barycentric attraction: -λ * < z_i , z_bar >  (z_bar stop-grad)
            with torch.no_grad():
                # (B,H)
                z_bar = (W.unsqueeze(-1) * z_k).sum(dim=1)
            # scalar
            bary = - bary_lambda * (z_i * z_bar).sum(dim=1).mean()
            bary_terms.append(bary)

        # aggregate anchors
        Qhat_mean = torch.stack(Qhat_list, dim=0).mean(
            dim=0)                           # (B,1)
        logp_mean = torch.stack(logp_list, dim=0).mean(
            dim=0)                           # (B,)
        bary_loss = torch.stack(bary_terms, dim=0).mean(
        )                               # scalar

        # diagnostics
        if wandb.run is not None:
            keff_now = (1.0 / (W.pow(2).sum(dim=1) + eps)).mean().item()
            wandb.log({
                "instate_fast/tau": float(self._tau_instate.item()),
                "instate_fast/keff": float(keff_now),
                "instate_fast/top_sim": float(S.max(dim=1).values.mean().item()),
            }, step=self.steps)

        return Qhat_mean, logp_mean, bary_loss

    def _actor_loss_instate_fast(self, obs,
                                 K: int = 64,
                                 noise_std: float = 0.10,
                                 base_temp: float = 1.0,
                                 positive_tail: bool = True,
                                 multi_tau: bool = True,
                                 cem_elites: int = 8,
                                 cem_new: int = 16,
                                 m_anchors: int = 1,
                                 bary_lambda: float = 0.2):

        Qhat, logp, bary_loss = self._qhat_in_state_fast(
            obs,
            K=K,
            noise_std=noise_std,
            base_temp=base_temp,
            positive_tail=positive_tail,
            multi_tau=multi_tau,
            cem_elites=cem_elites,
            cem_new=cem_new,
            m_anchors=m_anchors,
            bary_lambda=bary_lambda,
        )

        # entropy term (optionally clamp α early for stability)
        alpha = self.log_alpha.exp()
        # alpha = torch.clamp(alpha, min=0.03)  # optional floor during first ~200k steps

        entropy_loss = (alpha * logp).mean()
        actor_loss = entropy_loss - Qhat.mean() + bary_loss

        # standard alpha update
        alpha_loss = -(self.log_alpha *
                       (logp + self.target_entropy).detach()).mean()

        if wandb.run is not None:
            wandb.log({
                "instate_fast/Qhat": float(Qhat.mean().item()),
                "instate_fast/entropy_loss": float(entropy_loss.item()),
                "instate_fast/bary_loss": float(bary_loss.item()),
                "train/alpha": float(alpha.item()),
            }, step=self.steps)

        return actor_loss, alpha_loss

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

        if self.kernel_adaptive_tau:
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
            
            if self.normalize_obs:
                self.obs_rms.update(ot)
                otn = self.obs_rms.normalize(ot)
            else:
                otn = ot

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
