# dist_rl_fix/algos/sac_distance.py  (updated)
from typing import Optional
import gymnasium as gym
import torch, wandb
import torch.nn.functional as F

from dist_rl_fix.models.networks import DistanceTrunk, GaussianActor, TwinQ
from dist_rl_fix.buffers.replay import ReplayBuffer
from dist_rl_fix.representations import recursive_nstep_cosine_loss_ema, BetaEMA
from dist_rl_fix.utils import RunningMeanStd, polyak_update

class SACDistanceAgent:
    def __init__(self,
                 env_id: str,
                 seed: int,
                 device,
                 total_steps: int,
                 eval_episodes: int,
                 eval_freq: int,
                 buffer_size: int,
                 batch_size: int,
                 hidden: int,
                 gamma: float,
                 tau: float,
                 lr: float,
                 n_step: int,
                 rep_loss_weight: float,
                 rep_gamma_shape: float,
                 rep_lam: float,
                 rep_huber: float,
                 alpha: Optional[float],
                 save_dir: str,
                 **kwargs):

        # === extra knobs (safe defaults) ===
        self.target_entropy_scale = float(kwargs.get("target_entropy_scale", 0.7))  # SAC entropy target scaling
        self.updates_per_step = int(kwargs.get("updates_per_step", 1))
        self.alpha_cql = float(kwargs.get("alpha_cql", 0.0))  # small conservative Q term (0 to disable)

        # optional kernel auxiliary to actor (improves early training)
        self.kernel_aux_weight = float(kwargs.get("kernel_aux_weight", 0.0))
        self.kernel_temp = float(kwargs.get("kernel_temp", 0.5))
        self.kernel_cand = int(kwargs.get("kernel_cand", 2048))
        self.kernel_state_k = int(kwargs.get("kernel_state_k", 64))
        self.kernel_adaptive_tau = bool(kwargs.get("kernel_adaptive_tau", True))

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

        self.total_steps = total_steps
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.save_dir = save_dir

        # === Modules ===
        self.actor = GaussianActor(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)

        # (A) Critics with separate encoders (no sharing with rep trunk)
        self.qnet = TwinQ(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
        self.q_targ = TwinQ(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
        self.q_targ.load_state_dict(self.qnet.state_dict())

        # (B) Representation trunk (and target) used ONLY for rep-loss / kernel similarity
        self.rep_trunk = DistanceTrunk(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
        self.rep_trunk_targ = DistanceTrunk(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
        self.rep_trunk_targ.load_state_dict(self.rep_trunk.state_dict())

        # === Optimizers ===
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_q = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.optim_rep = torch.optim.Adam(self.rep_trunk.parameters(), lr=self.lr)

        # === Temperature alpha ===
        if alpha is None:
            # scaled target entropy (better for MuJoCo)
            self.target_entropy = -self.target_entropy_scale * float(self.act_dim)
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)
            self._alpha_fixed = False
        else:
            self._alpha = float(alpha)
            self._alpha_fixed = True
            self.target_entropy = -self.target_entropy_scale * float(self.act_dim)

        # Replay & normalization
        self.replay = ReplayBuffer(buffer_size, self.obs_dim, self.act_dim, device=self.device, gamma=self.gamma, n_step=n_step)
        self.obs_rms = RunningMeanStd(self.obs_dim, device=self.device)

        # Representation loss
        self.rep_loss_weight = rep_loss_weight
        self.rep_gamma_shape = rep_gamma_shape
        self.rep_lam = rep_lam
        self.rep_huber = rep_huber
        self.beta_ema = BetaEMA(decay=0.995)

        self.steps = 0
        self.warmup_steps = 5000
        self._printed_warmup_notice = False

        print("[Init] SACDistanceAgent setup complete")
        print(f"[Init] env={env_id}, device={device}, total_steps={total_steps}, batch_size={batch_size}, buffer_size={buffer_size}")
        print(f"[Init] n_step={n_step}, lr={lr}, gamma={gamma}, tau={tau}, rep_loss_weight={rep_loss_weight}, target_entropy={self.target_entropy:.2f}")

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
        total = 0.0; lengths = []
        for ep in range(self.eval_episodes):
            o, _ = self.eval_env.reset()
            done = False; trunc = False; L = 0
            while not (done or trunc):
                ot = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
                otn = self.obs_rms.normalize(ot)
                a = self._sample_action(otn, eval_mode=True).clamp(-1, 1)
                a_env = ((a + 1) / 2) * (self.high - self.low) + self.low
                o, r, done, trunc, _ = self.eval_env.step(a_env.squeeze(0).detach().cpu().numpy())
                total += r; L += 1
            lengths.append(L)
        avg = total / self.eval_episodes
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        print(f"[Eval] avg_return={avg:.2f}, avg_length={avg_len:.1f}")
        if wandb.run is not None:
            wandb.log({"eval/avg_reward": avg, "eval/avg_len": avg_len, "step": self.steps}, step=self.steps)
        return avg

    # ---------- critic / actor / rep losses ----------
    def _td_targets(self, r, d, next_obs):
        with torch.no_grad():
            next_obs_n = self.obs_rms.normalize(next_obs)
            a2, logp2, _ = self.actor.sample(next_obs_n)
            q1t, q2t = self.q_targ(next_obs_n, a2)
            qt = torch.min(q1t, q2t)
            alpha = self.alpha if self._alpha_fixed else self.log_alpha.exp()
            target = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * (qt - alpha * logp2)
        return target

    def _cql_regularizer(self, obs_n, q1, q2, policy, n_samples=10):
        """Small CQL term to reduce over-optimism (optional)."""
        if self.alpha_cql <= 0.0:
            return 0.0
        with torch.no_grad():
            B = obs_n.size(0)
            obs_rep = obs_n.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            a_samp, _, _ = policy.sample(obs_rep)
        q1_pi, q2_pi = self.qnet(obs_rep, a_samp)
        q_pi = torch.min(q1_pi, q2_pi).reshape(B, n_samples, 1).mean(dim=1)  # (B,1)
        q_data = torch.min(q1, q2)  # (B,1)
        # E_pi[Q] - E_D[Q]
        return self.alpha_cql * (q_pi.mean() - q_data.mean())

    def _q_loss(self, obs, act_env, r, next_obs, d):
        obs_n = self.obs_rms.normalize(obs)
        act = self._env_to_action(act_env)  # critics expect [-1,1]
        q1, q2 = self.qnet(obs_n, act)
        target = self._td_targets(r, d, next_obs)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # optional CQL
        cql = self._cql_regularizer(obs_n, q1, q2, self.actor)
        return loss_q + cql, {"train/q_loss_raw": float(loss_q.item()), "train/cql_reg": float(cql) if isinstance(cql, float) else float(cql.item())}

    def _actor_alpha_loss(self, obs):
        obs_n = self.obs_rms.normalize(obs)
        a, logp, _ = self.actor.sample(obs_n)
        q1, q2 = self.qnet(obs_n, a)
        q = torch.min(q1, q2)
        if self._alpha_fixed:
            alpha = self.alpha
            actor_loss = (alpha * logp - q).mean()
            alpha_loss = None
        else:
            alpha = self.log_alpha.exp()
            actor_loss = (alpha * logp - q).mean()
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        return actor_loss, alpha_loss

    def _rep_loss(self, obs, act_env, next_obs, done, nreturn):
        # Use rep trunk and TARGET rep trunk (stop-grad on next)
        obs_n = self.obs_rms.normalize(obs)
        next_obs_n = self.obs_rms.normalize(next_obs)

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():  # stop-grad target
            a2, _, _ = self.actor.sample(next_obs_n)
            z_next = self.rep_trunk_targ(next_obs_n, a2)

        loss, info = recursive_nstep_cosine_loss_ema(
            z, z_next, done, nreturn, discount=self.gamma, n=20,
            gamma_shape=self.rep_gamma_shape, lam=self.rep_lam, huber_delta=self.rep_huber,
            beta_ema=self.beta_ema
        )
        return loss, info

    # ---------- optional kernel auxiliary for actor ----------
    def _kernel_aux_term(self, obs):
        """Compute -E[Q_hat_kernel] with critics' Q as targets. Uses rep trunk; adaptive tau."""
        if self.kernel_aux_weight <= 0.0:
            return 0.0, {}

        obs_n = self.obs_rms.normalize(obs)
        a, _, _ = self.actor.sample(obs_n)

        # sample candidates
        obs_c, act_env_c, rtg_c, nret_c, _ = self.replay.sample_candidates(self.kernel_cand)
        obs_c_n = self.obs_rms.normalize(obs_c)
        act_c = self._env_to_action(act_env_c)

        # state-near top-K
        s_i = F.normalize(obs_n, p=2, dim=1)
        s_c = F.normalize(obs_c_n, p=2, dim=1)
        S_state = s_i @ s_c.T
        K = min(self.kernel_state_k, S_state.size(1))
        top_vals, top_idx = torch.topk(S_state, k=K, dim=1, largest=True)  # (B,K)

        # gather candidates
        B = obs.size(0)
        obs_c_k = torch.gather(obs_c_n.unsqueeze(0).expand(B, -1, -1), 1, top_idx.unsqueeze(-1).expand(-1, -1, obs_c_n.size(1)))
        act_c_k = torch.gather(act_c.unsqueeze(0).expand(B, -1, -1), 1, top_idx.unsqueeze(-1).expand(-1, -1, act_c.size(1)))

        # kernel similarities in rep space
        z_i = F.normalize(self.rep_trunk(obs_n, a), p=2, dim=1)                 # (B,H)
        z_c = self.rep_trunk(obs_c_k.reshape(B*K, -1), act_c_k.reshape(B*K, -1))
        z_c = F.normalize(z_c, p=2, dim=1).reshape(B, K, -1)                    # (B,K,H)
        S = (z_i.unsqueeze(1) * z_c).sum(-1)                                     # (B,K)

        # adaptive tau per row
        if self.kernel_adaptive_tau:
            sstd = S.std(dim=1, keepdim=True) + 1e-6
            tau = self.kernel_temp * sstd
            W = torch.softmax(S / tau, dim=1)
        else:
            W = torch.softmax(S / max(1e-6, self.kernel_temp), dim=1)

        # targets are critics' Q(s_c, a_c) (NOT returns)
        q1c, q2c = self.qnet(obs_c_k.reshape(B*K, -1), act_c_k.reshape(B*K, -1))
        qc = torch.min(q1c, q2c).reshape(B, K, 1)                                # (B,K,1)

        Qhat = (W.unsqueeze(-1) * qc).sum(dim=1)                                  # (B,1)
        aux = -Qhat.mean()  # we add to actor loss with kernel_aux_weight
        logs = {"kernel/top_state_sim_mean": float(top_vals.mean().item()),
                "kernel/aux_term": float(aux.item())}
        return aux, logs

    # ---------- training ----------
    def train_sac(self):
        print(f"[Train] Starting SAC training for {self.total_steps} steps (eval every {self.eval_freq})")
        o, _ = self.env.reset(seed=None)
        ep_r = 0.0; ep_len = 0

        while self.steps < self.total_steps:
            ot = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
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
            ep_r += r; ep_len += 1
            self.steps += 1
            o = o2

            if not self._printed_warmup_notice and self.steps >= self.warmup_steps:
                print(f"[Train] Warmup complete at step {self.steps}, switching to policy actions")
                self._printed_warmup_notice = True

            if done or trunc:
                print(f"[Train] Episode: return={ep_r:.2f}, length={ep_len}, total_steps={self.steps}")
                if wandb.run is not None:
                    wandb.log({"rollout/ep_reward": ep_r, "rollout/ep_len": ep_len, "step": self.steps}, step=self.steps)
                ep_r = 0.0; ep_len = 0
                o, _ = self.env.reset()

            # updates
            if self.steps < max(self.warmup_steps, self.batch_size):
                continue

            for _ in range(self.updates_per_step):
                obs, act_env, rew, next_obs, done_b, rtg, nret, _ = self.replay.sample(self.batch_size)

                # ---- Q update (separate encoders) ----
                loss_q, qinfo = self._q_loss(obs, act_env, rew, next_obs, done_b)
                self.optim_q.zero_grad(); loss_q.backward()
                torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
                self.optim_q.step()
                if wandb.run is not None:
                    wandb.log({**qinfo, "train/q_loss": float(loss_q.item()), "step": self.steps}, step=self.steps)

                # ---- Representation loss (with target rep trunk) ----
                if self.rep_loss_weight > 0.0:
                    rep_loss, rep_info = self._rep_loss(obs, act_env, next_obs, done_b, nret)
                    self.optim_rep.zero_grad()
                    (self.rep_loss_weight * rep_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.rep_trunk.parameters(), 10.0)
                    self.optim_rep.step()
                    rep_logs = {f"rep/{k}": v for k, v in rep_info.items()}
                    rep_logs.update({"rep/loss": float(rep_loss.item()), "step": self.steps})
                    if wandb.run is not None:
                        wandb.log(rep_logs, step=self.steps)

                # ---- Actor + alpha (plus optional kernel auxiliary) ----
                actor_loss, alpha_loss = self._actor_alpha_loss(obs)

                # kernel auxiliary shaping
                if self.kernel_aux_weight > 0.0:
                    aux, aux_logs = self._kernel_aux_term(obs)
                    actor_loss = actor_loss + self.kernel_aux_weight * aux
                    if wandb.run is not None:
                        wandb.log(aux_logs, step=self.steps)

                self.optim_actor.zero_grad(); actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.optim_actor.step()

                logs = {"train/actor_loss": float(actor_loss.item()), "step": self.steps}
                if alpha_loss is not None:
                    self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
                    logs["train/alpha"] = float(self.log_alpha.exp().item())
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
                # self._save("ckpt")

    def _save(self, name: str):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"{name}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "qnet": self.qnet.state_dict(),
            "rep_trunk": self.rep_trunk.state_dict(),
            "steps": self.steps
        }, path)
        if wandb.run is not None:
            wandb.save(path)
        print(f"[Save] Model saved to {path}")