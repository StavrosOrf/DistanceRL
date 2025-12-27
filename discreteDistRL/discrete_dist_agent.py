"""Training loop for the discrete Distance RL agent."""
from __future__ import annotations

import math
import os
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical

from dist_rl.representations import BetaEMA, recursive_nstep_cosine_loss_ema

from .buffers import AtariReplayBuffer
from .models import (
    AtariEncoder,
    CategoricalActorNet,
    TwinQDiscreteNet,
    DistanceTrunkDiscreteNet,
)
from .utils import polyak_update, set_seed, RunningMeanStd
from .wrappers import make_atari_env


class DiscreteDistAgent:
    """Distance-aware soft actor-critic variant for discrete action spaces."""

    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        total_steps: int = 10_000_000,
        eval_episodes: int = 10,
        eval_freq: int = 100_000,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        target_entropy_scale: float = 0.98,
        updates_per_step: int = 1,
        rep_gamma_shape: float = 2.0,
        rep_lam: float = 0.5,
        rep_huber: float = 0.2,
        warmup_steps: int = 50_000,
        normalize_obs: bool = False,
        K: int = 32,
        kernel_softmax_temp: float = 1.0,
        kernel_eps: float = 0.05,
        kernel_adaptive_tau: bool = True,
        center_qhat: bool = True,
        proposal_mode: str = "multinomial",
        proposal_topk: int = 0,
        proposal_eps: float = 0.0,
        use_one_hot_actions: bool = False,
        learn_alpha: bool = True,
        init_alpha: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 5.0,
        save_dir: str = "checkpoints",
        hidden_size: int = 512,
        feature_dim: int = 512,
        verbose: bool = False,
        shared_encoder: bool = False,
        **kwargs
    ) -> None:

        env = make_atari_env(env_id, seed, sticky=True, clip_rewards=False)
        eval_env = make_atari_env(
            env_id, seed + 1, sticky=True, clip_rewards=False)

        self.env = env
        self.eval_env = eval_env
        self.device = device

        # Core hyperparameters stored directly.
        self.total_steps = total_steps
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.target_entropy_scale = target_entropy_scale
        self.updates_per_step = updates_per_step
        self.rep_gamma_shape = rep_gamma_shape
        self.rep_lam = rep_lam
        self.rep_huber = rep_huber
        self.warmup_steps = warmup_steps
        self.normalize_obs = False
        self.center_qhat = center_qhat
        self.proposal_mode = proposal_mode
        self.proposal_topk = proposal_topk
        self.proposal_eps = proposal_eps
        self.use_one_hot_actions = use_one_hot_actions
        self.learn_alpha = learn_alpha
        self.init_alpha = init_alpha
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.verbose = verbose

        set_seed(seed)
        self.env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 1)

        obs_shape = self.env.observation_space.shape
        assert len(obs_shape) == 3, "Atari observations should be (C, 84, 84)."
        self.obs_shape = obs_shape
        self.frames = obs_shape[0]

        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_dim = self.env.action_space.n

        # Toggle observation normalization to mirror continuous agent behavior.
        self.normalize_obs = bool(normalize_obs)

        # Optional shared encoder for actor/critic/representation trunks.
        self.shared_encoder = bool(shared_encoder)
        encoder_shared = None
        if self.shared_encoder:
            encoder_shared = AtariEncoder(self.frames, feature_dim)
            encoder_shared = encoder_shared.to(self.device)
        self.encoder_shared = encoder_shared

        # === Networks ===
        hidden_dim = hidden_size
        self.actor = CategoricalActorNet(
            self.frames, self.action_dim, feature_dim, hidden_dim,
            encoder=self.encoder_shared,
        ).to(self.device)

        self.q_net = TwinQDiscreteNet(
            self.frames, self.action_dim, feature_dim, hidden_dim,
            encoder=self.encoder_shared,
        ).to(self.device)
        self.q_target = TwinQDiscreteNet(
            self.frames, self.action_dim, feature_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.rep_trunk = DistanceTrunkDiscreteNet(
            self.frames, self.action_dim, feature_dim, hidden_dim,
            use_one_hot_actions=use_one_hot_actions,
            verbose=verbose,
            encoder=self.encoder_shared,
        ).to(self.device)
        self.rep_trunk_target = DistanceTrunkDiscreteNet(
            self.frames, self.action_dim, feature_dim, hidden_dim,
            use_one_hot_actions=use_one_hot_actions,
            verbose=verbose,
        ).to(self.device)
        self.rep_trunk_target.load_state_dict(self.rep_trunk.state_dict())

        # === Optimisers ===
        self.optim_shared_encoder = None
        if self.shared_encoder:
            shared_params = list(self.encoder_shared.parameters())
            shared_param_ids = {id(p) for p in shared_params}

            def exclude_shared(params):
                return [p for p in params if id(p) not in shared_param_ids]

            self.optim_shared_encoder = torch.optim.Adam(
                shared_params, lr=lr, weight_decay=1e-4)
            self.optim_q = torch.optim.Adam(
                exclude_shared(self.q_net.parameters()), lr=lr, weight_decay=1e-4)
            self.optim_actor = torch.optim.Adam(
                exclude_shared(self.actor.parameters()), lr=lr)
            self.optim_rep = torch.optim.Adam(
                exclude_shared(self.rep_trunk.parameters()), lr=lr)
            self.shared_encoder_params = shared_params
        else:
            self.optim_q = torch.optim.Adam(
                self.q_net.parameters(), lr=lr, weight_decay=1e-4)
            self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.optim_rep = torch.optim.Adam(
                self.rep_trunk.parameters(), lr=lr)
            self.shared_encoder_params = []

        # SAC-style temperature: start at alpha=1.0 (log_alpha=0) unless overridden.
        init_alpha_t = torch.tensor(init_alpha, device=self.device)
        self.learn_alpha = learn_alpha
        self.log_alpha = torch.nn.Parameter(init_alpha_t.log())
        self.log_alpha.requires_grad = learn_alpha
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=lr) if learn_alpha else None
        self.fixed_alpha = None if learn_alpha else float(init_alpha_t.item())

        self.target_entropy = target_entropy_scale * \
            math.log(float(self.action_dim))

        self.replay = AtariReplayBuffer(
            buffer_size,
            observation_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.beta_ema = BetaEMA(decay=0.995)
        self.obs_rms = RunningMeanStd(self.obs_shape, device=self.device)
        self.max_grad_norm = max_grad_norm
        self.steps = 0
        self.best_eval = -float("inf")

        # Respect lightweight wandb flag passed through main.py; default False when absent.
        self.lightweight_wandb = bool(kwargs.get("lightweight_wandb", True))
        print(f"[Init] lightweight_wandb={self.lightweight_wandb}")

        self.K = max(1, K)
        self.kernel_softmax_temp = kernel_softmax_temp
        self.kernel_eps = kernel_eps
        self.kernel_adaptive_tau = kernel_adaptive_tau
        self.center_qhat = center_qhat
        self.proposal_mode = proposal_mode
        self.proposal_topk = proposal_topk
        self.proposal_eps = proposal_eps
        self.all_actions = torch.arange(
            self.action_dim, device=self.device, dtype=torch.long
        )

        os.makedirs(save_dir, exist_ok=True)

        if wandb.run is not None and not self.lightweight_wandb:
            wandb.run.log_code(".")

        # --- kernel/pi mixing schedule (fast early learning) ---
        self.kernel_pi_mix_init = float(kwargs.get(
            "kernel_pi_mix_init", 1.0))   # start as SAC
        self.kernel_pi_mix_final = float(kwargs.get(
            "kernel_pi_mix_final", 0.0))  # end as pure kernel
        self.kernel_pi_mix_steps = int(
            kwargs.get("kernel_pi_mix_steps", 300_000))

        # --- target entropy anneal (prevents staying near-uniform forever) ---
        # Use log(|A|) scale (not |A|).
        self.target_entropy_max = float(
            target_entropy_scale) * math.log(float(self.action_dim))
        self.target_entropy_min = float(kwargs.get(
            "target_entropy_min_scale", 0.25)) * math.log(float(self.action_dim))
        self.entropy_anneal_steps = int(
            kwargs.get("entropy_anneal_steps", 300_000))

        print(
            f"[Init] DiscreteDistAgent env={env_id} device={device} total_steps={total_steps}"
        )
        print(
            f"[Init] Observation shape: {obs_shape}, Action dim: {self.action_dim}")
        print(f"[Init] Feature dim: {feature_dim}, Hidden dim: {hidden_dim}")
        print(f"[Init] Buffer size: {buffer_size}, Batch size: {batch_size}")
        print(f"[Init] Warmup steps: {warmup_steps}, Eval freq: {eval_freq}")
        print(f"[Init] Gamma: {gamma}, Tau: {tau}, LR: {lr}")
        print(
            f"[Init] Rep gamma shape: {rep_gamma_shape}, Rep lambda: {rep_lam}")
        print(f"[Init] Proposal samples: {self.K}")
        alpha_mode = "learned" if self.learn_alpha else f"fixed={self.fixed_alpha:.4f}"
        print(
            f"[Init] updates_per_step: {updates_per_step}, Entropy coef: {entropy_coef}")
        print(
            f"[Init] kernel_softmax_temp: {kernel_softmax_temp}, kernel_eps: {kernel_eps}, kernel_adaptive_tau: {kernel_adaptive_tau}, center_qhat: {center_qhat}")
        print(
            f"[Init] proposal_mode: {proposal_mode}, proposal_topk: {proposal_topk}, proposal_eps: {proposal_eps}, use_one_hot_actions: {use_one_hot_actions}")
        print(
            f"[Init] Alpha mode: {alpha_mode}, Target entropy: {self.target_entropy:.2f}")
        print(f"[Init] Normalize obs: {self.normalize_obs}")
        print(f"[Init] Shared encoder: {self.shared_encoder}")

    def _lin_anneal(self, start: float, end: float, duration: int) -> float:
        if duration <= 0:
            return end
        p = min(1.0, float(self.steps) / float(duration))
        return start + p * (end - start)

    def _kernel_pi_mix_coef(self) -> float:
        # lambda_t: 1.0 -> 0.0 (default)
        return self._lin_anneal(self.kernel_pi_mix_init, self.kernel_pi_mix_final, self.kernel_pi_mix_steps)

    def _target_entropy_now(self) -> float:
        return self._lin_anneal(self.target_entropy_max, self.target_entropy_min, self.entropy_anneal_steps)

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

    def _normalize_obs(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self._maybe_normalize_obs(obs_t, update_stats=True)

    def _maybe_normalize_obs(self, obs_t: torch.Tensor, update_stats: bool = False) -> torch.Tensor:
        if not self.normalize_obs:
            return obs_t
        if update_stats:
            with torch.no_grad():
                self.obs_rms.update(obs_t)
        return self.obs_rms.normalize(obs_t)

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        obs_t = self._obs_to_tensor(obs)
        obs_t = self._maybe_normalize_obs(obs_t, update_stats=not eval_mode)
        with torch.no_grad():
            dist, logits = self.actor(obs_t)
            if eval_mode:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        return int(action.item())

    def _kernel_tau_instate(
        self, S_rowwise: torch.Tensor, base_temp: float
    ) -> torch.Tensor:
        rows = S_rowwise.size(0)
        device = S_rowwise.device

        tau_max = max(0.75, float(base_temp))
        tau_min = max(0.05, 0.30 * float(base_temp))
        T_sched = 200_000
        p = min(1.0, float(self.steps) / float(T_sched))
        tau_sched = tau_min + 0.5 * \
            (tau_max - tau_min) * (1.0 + math.cos(math.pi * p))

        tau = torch.full((rows, 1), tau_sched, device=device)
        if self.kernel_adaptive_tau:
            row_std = S_rowwise.std(dim=1, keepdim=True).clamp(min=1e-4)
            tau = tau + row_std
        return tau

    def _qhat_in_state_norm(
        self,
        obs: torch.Tensor,
        K: int | None = None,
        noise_std: float = 0.0,
        softmax_temp: float = 1.0,
        eps: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        In-state kernel readout + expected log-prob for SAC-style alpha update.

        Returns:
        Qhat: (B,1)
        logp: (B,1) where logp = E_a~pi [log pi(a|s)] = sum_a pi(a|s) log pi(a|s)
        """
        B = obs.size(0)
        A = self.action_dim

        # normalize obs consistently for actor/critic/rep
        obs_n = self._maybe_normalize_obs(obs, update_stats=False)

        # enumerate all discrete actions once (B,A,A)
        actions_full = torch.nn.functional.one_hot(
            self.all_actions, num_classes=A).float()  # (A,A)
        actions_full = actions_full.unsqueeze(0).expand(
            B, -1, -1)                           # (B,A,A)

        # compute z(s,a) + Q(s,a) without critic grads flowing into actor
        with torch.no_grad():
            z_all = self.rep_trunk_target(
                obs_n, actions_full)         # (B,A,H)
            z_all = torch.nn.functional.normalize(
                z_all, p=2, dim=-1)  # (B,A,H)

            # (B,A) (or q_target if you prefer)
            q1_all, q2_all = self.q_net(obs_n)
            q_min = torch.min(q1_all, q2_all)                          # (B,A)
            # (B,A,1)
            qk = q_min.unsqueeze(-1)

        # current policy (grad path)
        _, logits = self.actor(obs_n)                                  # (B,A)
        log_probs = torch.log_softmax(logits, dim=-1)                  # (B,A)
        probs = log_probs.exp()                                        # (B,A)

        # anchor: expectation AFTER nonlinearity in representation space
        z_anchor = torch.sum(probs.unsqueeze(-1) * z_all, dim=1)        # (B,H)
        z_anchor = torch.nn.functional.normalize(z_anchor, p=2, dim=-1)

        # similarities + kernel weights
        S = torch.einsum("bh,bah->ba", z_anchor,
                         z_all).clamp(-1.0, 1.0)   # (B,A)

        tau = self._kernel_tau_instate(
            S, base_temp=softmax_temp)          # (B,1)
        # (B,A)
        W = torch.softmax(S / tau, dim=-1)
        if eps > 0.0:
            W = (1.0 - eps) * W + eps / float(A)

        # --- NEW: mix kernel weights with pi weights early ---
        lam = self._kernel_pi_mix_coef()                                   # scalar
        if lam > 0.0:
            W_mix = (1.0 - lam) * W + lam * \
                probs                           # (B,A)
        else:
            W_mix = W

        # optional baseline centering uses mixed weights
        if self.center_qhat:
            q_bar = (W_mix.unsqueeze(-1) * qk).sum(dim=1,
                                                   keepdim=True).detach()  # (B,1,1)
            q_tilde = qk - q_bar
        else:
            q_tilde = qk

        # (B,1)
        Qhat = (W_mix.unsqueeze(-1) * q_tilde).sum(dim=1)

        # expected log-prob (=-entropy)
        logp = (probs * log_probs).sum(dim=-1,
                                       keepdim=True)                # (B,1)

        # diagnostics: effective number of actions under W_mix
        if wandb.run is not None and not self.lightweight_wandb:
            with torch.no_grad():
                W_ent = -(W_mix * (W_mix + 1e-8).log()
                          ).sum(dim=-1)         # (B,)
                neff_W = torch.exp(W_ent).mean().item()
                pi_ent = -(probs * (probs + 1e-8).log()
                           ).sum(dim=-1).mean().item()

            wandb.log(
                {
                    "kernel_instate/mean_tau": float(tau.mean().item()),
                    "kernel_instate/qhat_abs_mean": float(Qhat.abs().mean().item()),
                    "kernel_instate/pi_mix_lambda": float(lam),
                    "kernel_instate/neff_Wmix": float(neff_W),
                    "kernel_instate/pi_entropy": float(pi_ent),
                },
                step=self.steps,
            )

        return Qhat, logp

    def evaluate(self) -> float:
        print(
            f"\n[Eval] Step {self.steps}: Starting evaluation ({self.eval_episodes} episodes)...")
        returns = []
        for ep_idx in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total = 0.0
            while not done:
                action = self.act(obs, eval_mode=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(
                    action)
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        min_return = float(np.min(returns))
        max_return = float(np.max(returns))
        print(
            f"[Eval] Mean: {mean_return:.2f} ± {std_return:.2f} (Min: {min_return:.2f}, Max: {max_return:.2f})")
        if wandb.run is not None:
            wandb.log({"eval/avg_reward": mean_return}, step=self.steps)
        return mean_return

    def _update_critics(self, batch: Dict[str, torch.Tensor]) -> float:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        obs_n = self._maybe_normalize_obs(obs, update_stats=False)
        next_obs_n = self._maybe_normalize_obs(next_obs, update_stats=False)
        actions = batch["actions"].unsqueeze(-1)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)

        q1, q2 = self.q_net(obs_n)
        q1 = q1.gather(1, actions)
        q2 = q2.gather(1, actions)

        with torch.no_grad():
            _, logits_next = self.actor(next_obs_n)
            log_probs_next = torch.log_softmax(logits_next, dim=-1)
            probs_next = log_probs_next.exp()
            next_q1, next_q2 = self.q_target(next_obs_n)
            min_next_q = torch.min(next_q1, next_q2)
            next_values = (probs_next * (min_next_q -
                           self.alpha * log_probs_next)).sum(dim=-1)
            targets = rewards + (1.0 - dones) * self.gamma * next_values
            targets = targets.unsqueeze(-1)

        loss_q1 = F.mse_loss(q1, targets)
        loss_q2 = F.mse_loss(q2, targets)
        loss = loss_q1 + loss_q2

        self.optim_q.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.max_grad_norm)
        self.optim_q.step()

        if wandb.run is not None and not self.lightweight_wandb:
            wandb.log({"train/q_loss": float(loss.item())},
                      step=self.steps)
        return float(loss.item())

    def _update_actor_and_alpha(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)

        Qhat, logp = self._qhat_in_state_norm(
            obs,
            K=self.K,
            softmax_temp=self.kernel_softmax_temp,
            eps=self.kernel_eps,
        )

        # Use learnable alpha when enabled; detach for actor to avoid coupling its grads.
        alpha = self.log_alpha.exp() if self.learn_alpha else torch.tensor(self.fixed_alpha, device=self.device)
        alpha_detached = alpha.detach()

        # actor: minimize alpha * E[log pi] - Qhat
        actor_loss = (alpha_detached * logp - Qhat).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        # alpha: minimize alpha * (H(pi) - H_target)
        entropy = (-logp).detach()                            # (B,1)
        target_entropy = self._target_entropy_now()            # scalar

        alpha_loss_item = 0.0
        if self.learn_alpha:
            target_entropy_t = torch.tensor(
                target_entropy, device=self.device, dtype=entropy.dtype
            )
            alpha_loss = (alpha * (entropy - target_entropy_t)).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.alpha_opt.step()
            alpha_loss_item = float(alpha_loss.item())

        if wandb.run is not None and not self.lightweight_wandb:
            wandb.log(
                {
                    "train/actor_loss": float(actor_loss.item()),
                    "train/entropy": float(entropy.mean().item()),
                    "train/target_entropy": float(target_entropy),
                    "train/alpha": float(self.alpha),
                    "train/alpha_loss": float(alpha_loss_item),
                    "train/qhat_mean": float(Qhat.mean().item()),
                },
                step=self.steps,
            )

        return {"actor_loss": float(actor_loss.item()), "alpha_loss": float(alpha_loss_item)}

    def _representation_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        obs = self._maybe_normalize_obs(obs, update_stats=False)
        next_obs = self._maybe_normalize_obs(next_obs, update_stats=False)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        if self.verbose:
            print("\n\n[Representation Loss]")
        z = self.rep_trunk(obs, actions)

        with torch.no_grad():
            # --- policy at next state ---
            _, logits_next = self.actor(
                next_obs)                        # (B, A)
            log_probs_next = torch.log_softmax(
                logits_next, dim=-1)      # (B, A)
            probs_next = log_probs_next.exp()                            # (B, A)

            B = next_obs.size(0)
            A = self.action_dim

            # --- compute z(next_obs, a) for all actions a ---
            # actions_full: (B, A, A) one-hot for each action
            actions_full = torch.nn.functional.one_hot(
                self.all_actions.unsqueeze(0).expand(B, -1), num_classes=A
            ).float().to(next_obs.device)

            z_next_all = self.rep_trunk_target(
                next_obs, actions_full)   # (B, A, feature_dim)

            # --- expected next embedding z_pi(next_obs) = sum_a pi(a|s') z(s',a) ---
            z_next = torch.sum(probs_next.unsqueeze(-1) *
                               z_next_all, dim=1)  # (B, feature_dim)

            # --- expected next value ---
            q1_next, q2_next = self.q_target(
                next_obs)                   # (B, A)
            q_min_next = torch.min(
                q1_next, q2_next)                     # (B, A)

            # Option 1: plain expectation (usually good)
            # v_next = torch.sum(probs_next * q_min_next, dim=1, keepdim=True)  # (B, 1)

            # Option 2: soft expectation (often better aligned with your SAC-style critic)
            alpha = self.log_alpha.exp()
            v_next = torch.sum(probs_next * (q_min_next - alpha * log_probs_next),
                               dim=1, keepdim=True)  # (B, 1)

        # bootstrap target utility
        q_targ = rewards.unsqueeze(-1) + (1.0 -
                                          dones.unsqueeze(-1)) * self.gamma * v_next

        loss, info = recursive_nstep_cosine_loss_ema(
            z,
            z_next,
            dones,
            q_targ,
            discount=self.gamma,
            gamma_shape=self.rep_gamma_shape,
            lam=self.rep_lam,
            huber_delta=self.rep_huber,
            beta_ema=self.beta_ema,
        )

        self.optim_rep.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.rep_trunk.parameters(), self.max_grad_norm)
        self.optim_rep.step()

        if wandb.run is not None and not self.lightweight_wandb:
            rep_logs = {f"rep/{k}": v for k, v in info.items()}
            rep_logs.update(
                {"rep/loss": float(loss.item()), "step": self.steps})
            if wandb.run is not None:
                wandb.log(rep_logs, step=self.steps)
        return {"rep_loss": float(loss.item()), **info}

    def _update_targets(self) -> None:
        polyak_update(self.q_target, self.q_net, self.tau)
        polyak_update(self.rep_trunk_target, self.rep_trunk, self.tau)

    def save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.save_dir, f"{tag}.pt")
        payload = {
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
        print(f"[Train] Warmup phase: steps 1-{self.warmup_steps}")
        print(
            f"[Train] Training phase: steps {self.warmup_steps+1}-{self.total_steps}\n")

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        self.steps = 0

        while self.steps <= self.total_steps:
            self.steps += 1
            if self.steps < self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.act(obs, eval_mode=False)

            next_obs, reward, terminated, truncated, info = self.env.step(
                action)
            done = terminated or truncated
            self.replay.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += float(reward)
            episode_length += 1

            # Update step
            if self.steps > self.warmup_steps:
                for _ in range(self.updates_per_step):
                    if self.optim_shared_encoder is not None:
                        self.optim_shared_encoder.zero_grad(set_to_none=True)
                    batch = self.replay.sample(self.batch_size)
                    self._update_critics(batch)
                    self._update_actor_and_alpha(batch)
                    self._representation_loss(batch)
                    if self.optim_shared_encoder is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.shared_encoder_params, self.max_grad_norm)
                        self.optim_shared_encoder.step()
                    self._update_targets()

            if done:
                episode_count += 1
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/episode_return": episode_reward,
                            "train/episode_length": episode_length,
                        }, step=self.steps
                    )

                phase = "Warmup" if self.steps < self.warmup_steps else "Train"
                print(f"[{phase}] Step {self.steps:7d} | Episode {episode_count:4d} | Return: {episode_reward:7.2f} | Length: {episode_length:4d} | Buffer: {len(self.replay):7d}")

                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

                # exit(0)
            if self.steps % self.eval_freq == 0 and self.steps >= self.warmup_steps:
                eval_return = self.evaluate()
                if eval_return > self.best_eval:
                    print(
                        f"[Eval] New best return: {eval_return:.2f} (previous: {self.best_eval:.2f})")
                    self.best_eval = eval_return
                    self.save_checkpoint("best")
                else:
                    print(f"[Eval] Current best remains: {self.best_eval:.2f}")
                print()

        # final checkpoint
        print(
            f"\n[Train] Training complete! Total steps: {self.total_steps}")
        print(f"[Train] Best eval return: {self.best_eval:.2f}")
        self.save_checkpoint("final")
