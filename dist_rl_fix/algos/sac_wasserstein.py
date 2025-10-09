# dist_rl_fix/algos/sac_wasserstein.py
from typing import Optional
import math
import gymnasium as gym
import torch, wandb
import torch.nn.functional as F

from dist_rl_fix.models.networks import DistanceTrunkWs, GaussianActor, TwinQ
from dist_rl_fix.buffers.replay import ReplayBuffer
from dist_rl_fix.representations import instate_advantage_rep_loss, sinkhorn_divergence
from dist_rl_fix.utils import RunningMeanStd, polyak_update

class SACWassersteinAgent:
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
                 rep_margin_scale: float,
                 rep_temp: float,
                 rep_huber: float,
                 alpha: Optional[float],
                 save_dir: str,
                 # SAC / training extras
                 target_entropy_scale: float = 0.8,
                 target_entropy_scale_final: float = 0.3,
                 entropy_anneal_steps: int = 300_000,
                 updates_per_step: int = 2,
                 alpha_cql: float = 0.0,
                 # OT (Wasserstein) extras
                 ot_eta: float = 0.1,
                 ot_eps: float = 0.05,
                 ot_iters: int = 10,
                 ot_K: int = 16,
                 ot_Kt: int = 32,
                 ot_std_scale: float = 1.5,
                 ot_topk_target: bool = True,
                 **kwargs):

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

        # Modules
        self.actor = GaussianActor(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
        self.qnet = TwinQ(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
        self.q_targ = TwinQ(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
        self.q_targ.load_state_dict(self.qnet.state_dict())

        self.rep_trunk = DistanceTrunkWs(self.obs_dim,
                                       self.act_dim,
                                       hidden=hidden,
                                       out_dim=hidden).to(self.device)

        # Optims
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_q = torch.optim.Adam(self.qnet.parameters(), lr=self.lr, weight_decay=1e-4)
        self.optim_rep = torch.optim.Adam(self.rep_trunk.parameters(), lr=self.lr)

        # Temperature alpha
        self._alpha_fixed = alpha is not None
        if self._alpha_fixed:
            self._alpha = float(alpha)
        else:
            # target entropy anneals from scale -> scale_final
            self.target_entropy_scale = target_entropy_scale
            self.target_entropy_scale_final = target_entropy_scale_final
            self.entropy_anneal_steps = max(1, int(entropy_anneal_steps))
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # Replay & normalization
        self.replay = ReplayBuffer(buffer_size, self.obs_dim, self.act_dim, device=self.device, gamma=self.gamma, n_step=n_step)
        self.obs_rms = RunningMeanStd(self.obs_dim, device=self.device)

        # Rep loss setup
        self.rep_loss_weight = rep_loss_weight
        self.rep_margin_scale = rep_margin_scale
        self.rep_temp = rep_temp
        self.rep_huber = rep_huber

        # OT config
        self.ot_eta_start = ot_eta
        self.ot_eta_final = 0.00001
        self.ot_eta_anneal = int(0.5 * total_steps)  # anneal over first half
        self.ot_eps = ot_eps
        self.ot_iters = ot_iters
        self.ot_K = ot_K
        self.ot_Kt = ot_Kt
        self.ot_std_scale = ot_std_scale
        self.ot_topk_target = ot_topk_target

        # Training control
        self.steps = 0
        self.warmup_steps = 5_000
        self.updates_per_step = updates_per_step
        self.alpha_cql = alpha_cql

        self._printed_warmup_notice = False
        print("[Init] SACWassersteinAgent ready")
        print(f"[Init] SAC+OT: ot_eta={ot_eta}, eps={ot_eps}, iters={ot_iters}, K={ot_K}, Kt={ot_Kt}, std_scale={ot_std_scale}, topk_target={ot_topk_target}")

    @property
    def alpha(self):
        return self.log_alpha.exp().item() if not self._alpha_fixed else self._alpha

    @alpha.setter
    def alpha(self, v):
        self._alpha = v

    @property
    def ot_eta(self):
        """Linear anneal from ot_eta_start to ot_eta_final over ot_eta_anneal steps."""
        frac = min(1.0, self.steps / max(1, self.ot_eta_anneal))
        return (1 - frac) * self.ot_eta_start + frac * self.ot_eta_final

    # ---------- helpers ----------
    def _env_to_action(self, a):
        a = torch.as_tensor(a, device=self.device, dtype=torch.float32)
        return 2 * (a - self.low) / (self.high - self.low) - 1

    def _act_to_env(self, a):
        return ((a + 1) / 2) * (self.high - self.low) + self.low

    def _sample_action(self, obs_t: torch.Tensor, eval_mode=False):
        if eval_mode:
            mu, _ = self.actor.forward(obs_t)
            return torch.tanh(mu)
        a, _, _ = self.actor.sample(obs_t)
        return a

    def _target_entropy(self):
        if self._alpha_fixed:
            return -0.8 * float(self.act_dim)
        # cosine/linear anneal; here simple linear in steps
        frac = min(1.0, self.steps / float(self.entropy_anneal_steps))
        scale = (1 - frac) * self.target_entropy_scale + frac * self.target_entropy_scale_final
        return -scale * float(self.act_dim)

    # ---------- eval ----------
    def evaluate(self):
        total = 0.0; lengths = []
        for ep in range(self.eval_episodes):
            o, _ = self.eval_env.reset()
            done = False; trunc = False; L = 0
            while not (done or trunc):
                ot = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
                otn = self.obs_rms.normalize(ot)
                a = self._sample_action(otn, eval_mode=True).clamp(-1, 1)
                a_env = self._act_to_env(a)
                o, r, done, trunc, _ = self.eval_env.step(a_env.squeeze(0).detach().cpu().numpy())
                total += r; L += 1
            lengths.append(L)
        avg = total / self.eval_episodes
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        print(f"[Eval] avg_return={avg:.2f}, avg_length={avg_len:.1f}")
        if wandb.run is not None:
            wandb.log({"eval/avg_reward": avg, "eval/avg_len": avg_len, "step": self.steps}, step=self.steps)
        return avg

    # ---------- SAC core ----------
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
        if self.alpha_cql <= 0.0:
            return 0.0
        with torch.no_grad():
            B = obs_n.size(0)
            obs_rep = obs_n.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            a_samp, _, _ = policy.sample(obs_rep)
        q1_pi, q2_pi = self.qnet(obs_rep, a_samp)
        q_pi = torch.min(q1_pi, q2_pi).reshape(B, n_samples, 1).mean(dim=1)  # (B,1)
        q_data = torch.min(q1, q2)
        return self.alpha_cql * (q_pi.mean() - q_data.mean())

    def _q_loss(self, obs, act_env, r, next_obs, d):
        obs_n = self.obs_rms.normalize(obs)
        act = self._env_to_action(act_env)
        q1, q2 = self.qnet(obs_n, act)
        target = self._td_targets(r, d, next_obs)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        cql = self._cql_regularizer(obs_n, q1, q2, self.actor)
        return loss_q + (0.0 if isinstance(cql, float) else cql), {
            "train/q_loss_raw": float(loss_q.item()),
            "train/cql_reg": float(cql) if isinstance(cql, float) else float(cql.item())
        }

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
            self.target_entropy = self._target_entropy()
            alpha = self.log_alpha.exp()
            actor_loss = (alpha * logp - q).mean()
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        return actor_loss, alpha_loss

    # ---------- In-state advantage-aware rep loss ----------
    def _rep_loss(self, obs):
        if self.rep_loss_weight <= 0.0:
            return 0.0, {}

        B = obs.size(0)
        obs_n = self.obs_rms.normalize(obs)

        # sample K actions per state from policy (slightly widened to explore metric)
        K = 8
        mu, std = self.actor.forward(obs_n)         # (B, A)
        std_wide = std * 1.2
        eps = torch.randn(B, K, mu.size(-1), device=obs.device, dtype=obs.dtype)
        pre_tanh = mu.unsqueeze(1) + std_wide.unsqueeze(1) * eps
        a_k = torch.tanh(pre_tanh).reshape(B*K, -1) # (B*K, A)

        # Q(s, a_k)
        obs_rep = obs_n.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)
        q1k, q2k = self.qnet(obs_rep, a_k)
        qk = torch.min(q1k, q2k).reshape(B, K, 1)   # (B,K,1)

        # embeddings
        z_all = self.rep_trunk(obs_rep, a_k).reshape(B, K, -1)     # (B,K,D)
        # anchor = best action by Q per state
        with torch.no_grad():
            idx_pos = torch.argmax(qk.squeeze(-1), dim=1)          # (B,)
        z_anchor = z_all[torch.arange(B, device=obs.device), idx_pos]  # (B, D)

        loss, info = instate_advantage_rep_loss(
            z_anchor=z_anchor, z_all=z_all, q_all=qk,
            margin_scale=self.rep_margin_scale, temp=self.rep_temp, huber_delta=self.rep_huber
        )
        return loss, info

    # ---------- OT policy improvement term ----------
    def _ot_actor_term(self, obs):
        """
        Build two empirical measures in z-space at each state:
          - mu_s: K policy samples
          - nu_s: Kt target proposals softmaxed by Q/alpha (Boltzmann)
        Compute batch Sinkhorn divergence and return mean (to add to actor loss).
        """
        if self.ot_eta <= 0.0:
            return 0.0, {}

        B = obs.size(0)
        obs_n = self.obs_rms.normalize(obs)

        # (1) Policy samples: K actions from current policy
        K = self.ot_K
        a_pol, _, _ = self.actor.sample(obs_n)                 # (B,A)
        # widen with extra Gaussian noise to diversify K samples
        mu, std = self.actor.forward(obs_n)
        std_pol = std * 1.0
        eps = torch.randn(B, K, mu.size(-1), device=obs.device, dtype=obs.dtype)
        a_pol_K = torch.tanh(mu.unsqueeze(1) + std_pol.unsqueeze(1) * eps).reshape(B*K, -1)  # (B*K, A)

        # embeddings for policy samples
        obs_rep = obs_n.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)
        z_pol = self.rep_trunk(obs_rep, a_pol_K).reshape(B, K, -1)             # (B,K,D)
        a_weights = torch.full((B, K), 1.0 / K, device=obs.device, dtype=obs.dtype)

        # (2) Target proposals: Kt wider proposals, select top-Kt by Q or all softmaxed
        Kt = self.ot_Kt
        mu, std = self.actor.forward(obs_n)
        std_targ = std * self.ot_std_scale
        eps_t = torch.randn(B, Kt, mu.size(-1), device=obs.device, dtype=obs.dtype)
        a_targ = torch.tanh(mu.unsqueeze(1) + std_targ.unsqueeze(1) * eps_t).reshape(B*Kt, -1)

        obs_targ = obs_n.unsqueeze(1).expand(B, Kt, -1).reshape(B*Kt, -1)
        q1t, q2t = self.qnet(obs_targ, a_targ)
        qt = torch.min(q1t, q2t).reshape(B, Kt)                                     # (B,Kt)
        # Top-Kt selection optional (we can just softmax all Kt)
        if self.ot_topk_target:
            # take the same Kt; we already have exactly Kt; could reduce to top half if desired
            pass

        # Boltzmann weights by Q/alpha
        with torch.no_grad():
            alpha = self.alpha if self._alpha_fixed else self.log_alpha.exp().item()
        w_targ = torch.softmax(qt / max(alpha, 1e-6), dim=1)                        # (B,Kt)

        z_targ = self.rep_trunk(obs_targ, a_targ).reshape(B, Kt, -1)                # (B,Kt,D)

        # (3) Sinkhorn divergence over the batch
        S = sinkhorn_divergence(
            X=z_pol, Y=z_targ, a=a_weights, b=w_targ,
            epsilon=self.ot_eps, n_iters=self.ot_iters
        )  # (B,)
        loss = S.mean()
        logs = {
            "ot/sinkhorn": float(loss.item()),
            "ot/alpha_used": float(alpha),
            "ot/q_targ_mean": float(qt.mean().item()),
        }
        return loss, logs

    # ---------- training ----------
    def train(self):
        print(f"[Train] Starting SAC+OT training for {self.total_steps} steps (eval every {self.eval_freq})")
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
                    a_env = self._act_to_env(a)
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

            if self.steps < max(self.warmup_steps, self.batch_size):
                continue

            for _ in range(self.updates_per_step):
                obs, act_env, rew, next_obs, done_b, rtg, nret, _ = self.replay.sample(self.batch_size)

                # --- Q update ---
                loss_q, qinfo = self._q_loss(obs, act_env, rew, next_obs, done_b)
                self.optim_q.zero_grad(); loss_q.backward()
                torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
                self.optim_q.step()
                if wandb.run is not None:
                    wandb.log({**qinfo, "train/q_loss": float(loss_q.item()), "step": self.steps}, step=self.steps)

                # --- In-state rep loss ---
                if self.rep_loss_weight > 0.0:
                    rep_loss, rep_info = self._rep_loss(obs)
                    self.optim_rep.zero_grad()
                    (self.rep_loss_weight * rep_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.rep_trunk.parameters(), 10.0)
                    self.optim_rep.step()
                    if wandb.run is not None:
                        wandb.log({**rep_info, "rep/loss": float(rep_loss.item()), "step": self.steps}, step=self.steps)

                # --- Actor + alpha (SAC) ---
                actor_loss, alpha_loss = self._actor_alpha_loss(obs)

                # --- OT term (Wasserstein) ---
                if self.ot_eta > 0.0:
                    ot_loss, ot_logs = self._ot_actor_term(obs)
                    actor_loss = actor_loss + self.ot_eta * ot_loss
                    if wandb.run is not None:
                        wandb.log(ot_logs, step=self.steps)

                self.optim_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.optim_actor.step()

                logs = {"train/actor_loss": float(actor_loss.item()), "step": self.steps}
                if alpha_loss is not None:
                    self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
                    logs["train/alpha"] = float(self.log_alpha.exp().item())
                    logs["train/target_entropy"] = float(self.target_entropy)
                else:
                    logs["train/alpha"] = float(self.alpha)
                if wandb.run is not None:
                    wandb.log(logs, step=self.steps)

                # --- Target updates ---
                polyak_update(self.q_targ, self.qnet, self.tau)

            if (self.steps % self.eval_freq) == 0:
                print(f"[Train] Evaluation at step {self.steps}")
                self.evaluate()
                self._save("ckpt")

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
