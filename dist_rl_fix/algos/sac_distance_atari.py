from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn.functional as F

try:
    import wandb
except Exception:  # pragma: no cover
    class Dummy:
        def __getattr__(self, k): return lambda *a, **kw: None
    wandb = Dummy()

from dist_rl_fix.atari_utils.atari_wrappers import make_atari_env
from dist_rl_fix.atari_utils.atari_buffer import ReplayBufferAtari
from dist_rl_fix.utils import polyak_update
from dist_rl_fix.representations import (
    recursive_nstep_cosine_loss_ema, BetaEMA,
)
from dist_rl_fix.atari_utils.atari_networks import (
    VisualDistanceTrunkIMPALA,
    VisualDistanceTrunkNature,
    CategoricalActorVisualIMPALA,
    CategoricalActorVisualNature,
    DiscreteTwinQVisualIMPALA,
    DiscreteTwinQVisualNature,
)

# ---- DrQ-style random shift augmentation ----
@torch.no_grad()
def random_shift(imgs: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """imgs: [B,C,H,W] in [0,1]; returns augmented images with random translations."""
    b, c, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    # Sample integer shifts
    ys = torch.randint(-pad, pad + 1, size=(b,), device=imgs.device)
    xs = torch.randint(-pad, pad + 1, size=(b,), device=imgs.device)
    # build base grid
    base_y = torch.linspace(-1.0, 1.0, steps=h, device=imgs.device)
    base_x = torch.linspace(-1.0, 1.0, steps=w, device=imgs.device)
    grid_y = base_y.view(1, h, 1).expand(b, h, w)
    grid_x = base_x.view(1, 1, w).expand(b, h, w)
    # convert pixel shifts ~> normalized coords
    shift_y = ys.float() * (2.0 / (h + 2 * pad))
    shift_x = xs.float() * (2.0 / (w + 2 * pad))
    grid = torch.stack((grid_x - shift_x.view(-1, 1, 1), grid_y - shift_y.view(-1, 1, 1)), dim=-1)
    out = F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out

class SACDistanceAtari:
    """
    Distance‑regularized discrete SAC for Atari (visual observations).
    Components:
      - Encoder: NatureCNN or IMPALA (IMPALA recommended)
      - Actor: categorical π(a|s), expectation-form actor loss
      - Critic: twin dueling Q, double Q, soft target updates
      - Rep. loss: recursive n‑step cosine on z(s,a) from a dedicated trunk
      - Optional NoisyNets on heads
      - DrQ random shift augmentation (set aug=True)
    """
    def __init__(self,
                 env_id: str,
                 seed: int,
                 device: str,
                 total_steps: int = 5_000_000,
                 eval_episodes: int = 10,
                 eval_freq: int = 50_000,
                 buffer_size: int = 1_000_000,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 lr: float = 3e-4,
                 aug: bool = True,
                 use_impala: bool = True,
                 use_noisy: bool = True,
                 rep_loss_weight: float = 0.1,
                 rep_gamma_shape: float = 1.0,
                 rep_lam: float = 0.5,
                 rep_huber: float = 0.2,
                 target_entropy_scale: float = 1.0,  # target H = -scale * log(A)
                 warmup_steps: int = 50_000,
                 save_dir: str = "./checkpoints",
                 **kwargs):
        self.device = torch.device(device)
        self.env = make_atari_env(env_id, seed=seed)
        self.eval_env = make_atari_env(env_id, seed=seed+1)
        self.n_actions = int(self.env.action_space.n)
        c, h, w = self.env.observation_space.shape
        self.obs_shape = (c, h, w)

        self.total_steps = int(total_steps)
        self.eval_episodes = int(eval_episodes)
        self.eval_freq = int(eval_freq)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.lr = float(lr)
        self.aug = bool(aug)
        self.use_impala = bool(use_impala)
        self.save_dir = save_dir
        self.warmup_steps = int(warmup_steps)

        # ---- modules ----
        if self.use_impala:
            self.actor = CategoricalActorVisualIMPALA(c, self.n_actions, use_noisy=use_noisy).to(self.device)
            self.qnet = DiscreteTwinQVisualIMPALA(c, self.n_actions, use_noisy=use_noisy).to(self.device)
            self.q_targ = DiscreteTwinQVisualIMPALA(c, self.n_actions, use_noisy=use_noisy).to(self.device)
            self.q_targ.load_state_dict(self.qnet.state_dict())
            self.rep_trunk = VisualDistanceTrunkIMPALA(c, self.n_actions).to(self.device)
            self.rep_trunk_targ = VisualDistanceTrunkIMPALA(c, self.n_actions).to(self.device)
        else:
            self.actor = CategoricalActorVisualNature(c, self.n_actions, use_noisy=use_noisy).to(self.device)
            self.qnet = DiscreteTwinQVisualNature(c, self.n_actions, use_noisy=use_noisy).to(self.device)
            self.q_targ = DiscreteTwinQVisualNature(c, self.n_actions, use_noisy=use_noisy).to(self.device)
            self.q_targ.load_state_dict(self.qnet.state_dict())
            self.rep_trunk = VisualDistanceTrunkNature(c, self.n_actions).to(self.device)
            self.rep_trunk_targ = VisualDistanceTrunkNature(c, self.n_actions).to(self.device)
        self.rep_trunk_targ.load_state_dict(self.rep_trunk.state_dict())

        # ---- optims ----
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_q = torch.optim.Adam(self.qnet.parameters(), lr=self.lr, weight_decay=1e-4)
        self.optim_rep = torch.optim.Adam(self.rep_trunk.parameters(), lr=self.lr)

        # ---- temperature (entropy) ----
        self.target_entropy = - target_entropy_scale * math.log(self.n_actions)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # ---- replay ----
        self.replay = ReplayBufferAtari(buffer_size, self.obs_shape, device=self.device.type)

        # ---- rep loss params ----
        self.rep_loss_weight = float(rep_loss_weight)
        self.rep_gamma_shape = float(rep_gamma_shape)
        self.rep_lam = float(rep_lam)
        self.rep_huber = float(rep_huber)
        self.beta_ema = BetaEMA(decay=0.995)

        self.steps = 0
        self.best_eval = -1e9
        if getattr(wandb, "run", None) is not None:
            try:
                wandb.run.log_code(".")
            except Exception:
                pass
        print(f"[Init] Atari agent on {env_id} | actions={self.n_actions} obs={self.obs_shape} impala={self.use_impala}")

    # ---- utilities ----
    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def _policy_pi_logpi(self, logits: torch.Tensor):
        log_pi = torch.log_softmax(logits, dim=-1)
        return log_pi.exp(), log_pi

    @torch.no_grad()
    def _act(self, obs: torch.Tensor, eval_mode: bool = False):
        logits = self.actor(obs)
        if eval_mode:
            return torch.argmax(logits, dim=-1)
        pi, _ = self._policy_pi_logpi(logits)
        return torch.distributions.Categorical(probs=pi).sample()

    @torch.no_grad()
    def evaluate(self):
        total = 0.0
        duration = []
        
        for _ in range(self.eval_episodes):
            o, _ = self.eval_env.reset()
            done = False; trunc = False
            while not (done or trunc):
                ot = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
                a = self._act(ot, eval_mode=True).item()
                o, r, done, trunc, _ = self.eval_env.step(a)
                total += r
        avg = total / self.eval_episodes
        if avg > self.best_eval:
            self.best_eval = avg
            self._save("best")
        if getattr(wandb, "run", None) is not None:
            wandb.log({"eval/avg_reward": avg, "step": self.steps}, step=self.steps)
        print(f"[Eval] avg_return={avg:.2f} ep duration={self.eval_env.spec.max_episode_steps} steps={self.steps}")
        return avg

    # ---- losses ----
    def _q_loss(self, obs, acts, rews, next_obs, dones):
        if self.aug:
            obs = random_shift(obs)
            next_obs = random_shift(next_obs)
        q1, q2 = self.qnet(obs)                      # [B,A]
        q1_a = q1.gather(1, acts.view(-1, 1)).squeeze(1)
        q2_a = q2.gather(1, acts.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            logits_next = self.actor(next_obs)
            pi_next, log_pi_next = self._policy_pi_logpi(logits_next)
            q1t, q2t = self.q_targ(next_obs)
            q_min = torch.min(q1t, q2t)              # [B,A]
            v_next = (pi_next * (q_min - self.alpha * log_pi_next)).sum(dim=1)
            target = rews + (1.0 - dones) * self.gamma * v_next
        loss_q = F.mse_loss(q1_a, target) + F.mse_loss(q2_a, target)
        info = {"train/q_loss_raw": float(loss_q.item())}
        return loss_q, info

    def _actor_alpha_loss(self, obs):
        if self.aug:
            obs = random_shift(obs)
        logits = self.actor(obs)                     # [B,A]
        pi, log_pi = self._policy_pi_logpi(logits)
        with torch.no_grad():
            q1, q2 = self.qnet(obs)
            q_min = torch.min(q1, q2)
        actor_loss = (pi * (self.alpha.detach() * log_pi - q_min)).sum(dim=1).mean()
        # Standard alpha update: encourage entropy ~ target_entropy
        entropy = -(pi * log_pi).sum(dim=1).mean()
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        ent = float(entropy.item())
        return actor_loss, alpha_loss, {"train/entropy": ent}

    def _rep_loss(self, obs, acts, next_obs, dones):
        if self.aug:
            obs_aug = random_shift(obs)
            next_aug = random_shift(next_obs)
        else:
            obs_aug, next_aug = obs, next_obs
        # z(s,a) using current rep trunk; z_next(s', a'_targ) using target rep trunk with policy sample
        z = self.rep_trunk(obs_aug, acts)  # (B,D)
        with torch.no_grad():
            logits_next = self.actor(next_aug)
            pi_next, _ = self._policy_pi_logpi(logits_next)
            a_next = torch.distributions.Categorical(probs=pi_next).sample()
            z_next = self.rep_trunk_targ(next_aug, a_next)
            # reward‑aware signal via target critics
            q1, q2 = self.q_targ(obs_aug)
            q_targ = torch.min(q1, q2).gather(1, acts.view(-1,1)).squeeze(1)  # (B,)
        loss, info = recursive_nstep_cosine_loss_ema(
            z, z_next, dones, q_targ,
            discount=self.gamma,
            gamma_shape=self.rep_gamma_shape,
            lam=self.rep_lam,
            huber_delta=self.rep_huber,
            beta_ema=self.beta_ema,
        )
        return loss, info

    # ---- training ----
    def train(self):
        print(f"[Train] Starting Atari training for {self.total_steps} steps (eval every {self.eval_freq}).")
        o, _ = self.env.reset()
        ep_ret = 0.0
        while self.steps < self.total_steps:
            # acting
            ot = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
            if self.steps < self.warmup_steps:
                a = self.env.action_space.sample()
            else:
                a = self._act(ot, eval_mode=False).item()
            o2, r, done, trunc, _ = self.env.step(a)
            self.replay.add(o, o2, a, r, done or trunc)
            ep_ret += r
            self.steps += 1
            o = o2

            if done or trunc:
                if getattr(wandb, "run", None) is not None:
                    wandb.log({"rollout/ep_reward": ep_ret, "step": self.steps}, step=self.steps)
                ep_ret = 0.0
                o, _ = self.env.reset()

            # updates
            if self.steps < max(self.warmup_steps, self.batch_size):
                if (self.steps % self.eval_freq) == 0:
                    self.evaluate()
                continue

            for _ in range(1):  # 1 gradient step per env step
                obs, next_obs, acts, rews, dones = self.replay.sample(self.batch_size)

                # Q update
                loss_q, qinfo = self._q_loss(obs, acts, rews, next_obs, dones)
                self.optim_q.zero_grad(set_to_none=True)
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
                self.optim_q.step()

                # Rep update
                if self.rep_loss_weight > 0:
                    self.optim_rep.zero_grad(set_to_none=True)
                    rep_loss, rep_info = self._rep_loss(obs, acts, next_obs, dones)
                    (rep_loss * self.rep_loss_weight).backward()
                    torch.nn.utils.clip_grad_norm_(self.rep_trunk.parameters(), 10.0)
                    self.optim_rep.step()
                    if getattr(wandb, "run", None) is not None:
                        wandb.log({"rep/loss": float(rep_loss.item()), **{f"rep/{k}": v for k, v in rep_info.items()}, "step": self.steps}, step=self.steps)

                # Actor + alpha
                actor_loss, alpha_loss, aux = self._actor_alpha_loss(obs)
                self.optim_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.optim_actor.step()

                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()

                if getattr(wandb, "run", None) is not None:
                    wandb.log({"train/actor_loss": float(actor_loss.item()),
                               "train/alpha": float(self.alpha.item()),
                               "train/alpha_loss": float(alpha_loss.item()),
                               **aux, **qinfo,
                               "step": self.steps}, step=self.steps)

                # targets
                polyak_update(self.q_targ, self.qnet, self.tau)
                polyak_update(self.rep_trunk_targ, self.rep_trunk, self.tau)

            if (self.steps % self.eval_freq) == 0:
                self.evaluate()

    def _save(self, name: str):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"{name}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "qnet": self.qnet.state_dict(),
            "rep_trunk": self.rep_trunk.state_dict(),
            "steps": self.steps,
        }, path)
        print(f"[Save] Saved to {path}")