import torch
import torch.nn.functional as F
import wandb

from dist_rl.dist_agent import DistAgent
from dist_rl.models import DistanceTrunk
from dist_rl.representations import recursive_nstep_cosine_loss_ema
from dist_rl.utils import polyak_update


def _zero_rep_loss(device: torch.device):
    """Utility to return a dummy rep loss that still supports backward()."""
    return torch.zeros(1, device=device, requires_grad=True), {"disabled": 1.0}


class DistAblationA1RandomEncoder(DistAgent):
    """A1: Random fixed encoder; rep loss disabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for p in self.rep_trunk.parameters():
            p.requires_grad = False
        for p in self.rep_trunk_targ.parameters():
            p.requires_grad = False

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        return _zero_rep_loss(self.device)


class DistAblationA2ActorOnlyEncoder(DistAgent):
    """A2: Encoder trained only through actor/critic pathways; no rep loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Let actor optimizer also update the rep trunk.
        self.optim_actor = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.rep_trunk.parameters()), lr=3e-4
        )

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        return _zero_rep_loss(self.device)


class DistAblationA3NoTemporalMix(DistAgent):
    """A3: Remove temporal mixing (lambda = 0)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rep_lam = 0.0


class DistAblationA4NoBetaScaling(DistAgent):
    """A4: Remove quantile-EMA scaling; use fixed scale (default 1.0)."""

    def __init__(self, *args, **kwargs):
        self.rep_fixed_scale = kwargs.pop("rep_fixed_scale", 1.0)
        super().__init__(*args, **kwargs)

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
            next_obs_n = self.obs_rms.normalize(next_obs)
        else:
            obs_n = obs
            next_obs_n = next_obs

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():
            a2, _, _ = self.actor.sample(next_obs_n)
            a2 += torch.randn_like(a2) * self.noise_std
            a2 = a2.clamp(-1, 1)
            a2 = ((a2 + 1) / 2) * (self.high - self.low) + self.low
            z_next = self.rep_trunk_targ(next_obs_n, a2)

            q1t, q2t = self.q_targ(obs_n, act)
            q_targ = torch.min(q1t, q2t).squeeze(-1)
        
        q_targ = rew + self.gamma * (1.0 - done) * q_targ

        # Fixed-scale variant of the recursive loss: replace beta EMA with constant.
        z_norm = F.normalize(z, p=2, dim=1)
        z_next_norm = F.normalize(z_next, p=2, dim=1)

        S = z_norm @ z_norm.T
        S_next = z_next_norm @ z_next_norm.T

        u = q_targ.view(-1, 1)
        G = (u - u.T).abs()
        beta = torch.tensor(self.rep_fixed_scale, device=self.device).clamp(min=1e-6)
        Delta = (G / beta).clamp(0.0, 1.0)
        T = 1.0 - 2.0 * (Delta ** float(self.rep_gamma_shape))

        alive = (1.0 - done.view(-1, 1)).to(S.dtype)
        Y = (1.0 - self.rep_lam) * T + self.rep_lam * alive * (self.gamma * S_next)

        mask = torch.ones_like(S, dtype=torch.bool)
        mask.fill_diagonal_(False)
        err = (S - Y)[mask]
        loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=self.rep_huber, reduction="mean")

        info = {
            "beta_fixed": float(beta),
            "mean_gap": float(G[mask].mean()),
            "mean_targets": float(Y[mask].mean()),
            "mean_cos": float(S[mask].mean()),
        }
        return loss, info


class DistAblationA5GammaFixed(DistAgent):
    """A5: Fix curvature shaping gamma_shape to 1.0."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rep_gamma_shape = 1.0


class DistAblationB1UniformKernel(DistAgent):
    """B1: Uniform weights over proposals (no cosine kernel influence)."""

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                            softmax_temp: float = 1.0, eps: float = 0.05):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)

        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low
            z_k = F.normalize(self.rep_trunk_targ(obs_rep, a_k), p=2, dim=1)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)

        a_anchor, logp, _ = self.actor.sample(obs_n)
        z_i = F.normalize(self.rep_trunk(obs_n, a_anchor), p=2, dim=1)
        z_k_view = z_k.view(B, K, -1)
        _ = torch.einsum('bd,bkd->bk', z_i, z_k_view)  # keep symmetry with base path

        W = torch.full((B, K), 1.0 / K, device=self.device)
        # q_bar = (W.unsqueeze(-1) * qk).sum(dim=1, keepdim=True).detach()
        q_tilde = qk
        Qhat = (W.unsqueeze(-1) * q_tilde).sum(dim=1)
        return Qhat, logp


class DistAblationB2EuclideanSim(DistAgent):
    """B2: Use Euclidean distance instead of cosine similarity."""

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                            softmax_temp: float = 1.0, eps: float = 0.05):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)

        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low
            z_k = self.rep_trunk_targ(obs_rep, a_k)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)

        a_anchor, logp, _ = self.actor.sample(obs_n)
        z_i = self.rep_trunk(obs_n, a_anchor)
        z_k_view = z_k.view(B, K, -1)
        # Negative Euclidean distance as similarity
        S = -((z_i.unsqueeze(1) - z_k_view).pow(2).sum(dim=-1).sqrt())

        tau_row = self._kernel_tau_instate(S, base_temp=softmax_temp)
        W = torch.softmax(S / tau_row, dim=1)
        if eps > 0.0:
            W = (1.0 - eps) * W + eps / K

        q_bar = (W.unsqueeze(-1) * qk).sum(dim=1, keepdim=True).detach()
        q_tilde = qk - q_bar
        Qhat = (W.unsqueeze(-1) * q_tilde).sum(dim=1)
        return Qhat, logp

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        # Use negative L2 distance for both current and target embeddings.
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
            next_obs_n = self.obs_rms.normalize(next_obs)
        else:
            obs_n = obs
            next_obs_n = next_obs

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():
            a2, _, _ = self.actor.sample(next_obs_n)
            a2 += torch.randn_like(a2) * self.noise_std
            a2 = a2.clamp(-1, 1)
            a2 = ((a2 + 1) / 2) * (self.high - self.low) + self.low
            z_next = self.rep_trunk_targ(next_obs_n, a2)

            q1t, q2t = self.q_targ(obs_n, act)
            q_targ = torch.min(q1t, q2t).squeeze(-1)

        # Pairwise negative Euclidean similarities
        S = -torch.cdist(z, z, p=2)
        S_next = -torch.cdist(z_next, z_next, p=2)

        u = q_targ.view(-1, 1)
        G = (u - u.T).abs()
        with torch.no_grad():
            beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
            beta = z.new_tensor(self.beta_ema.update(beta_batch) if self.beta_ema is not None else beta_batch)
        Delta = (G / beta).clamp(0.0, 1.0)
        T = 1.0 - 2.0 * (Delta ** float(self.rep_gamma_shape))

        alive = (1.0 - done.view(-1, 1)).to(S.dtype)
        Y = (1.0 - self.rep_lam) * T + self.rep_lam * alive * (self.gamma * S_next)

        mask = torch.ones_like(S, dtype=torch.bool)
        mask.fill_diagonal_(False)
        err = (S - Y)[mask]
        loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=self.rep_huber, reduction="mean")

        info = {
            "beta_batch": float(beta_batch),
            "beta_ema": float(beta),
            "mean_gap": float(G[mask].mean()),
            "mean_targets": float(Y[mask].mean()),
            "mean_sim": float(S[mask].mean()),
        }
        return loss, info


class DistAblationB6PoincareSim(DistAgent):
    """B6: Poincare-ball distance (negative) as similarity for kernel and rep loss."""

    def _to_ball(self, z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # Map to open unit ball to keep distance well-defined.
        return torch.tanh(z) * (1 - eps)

    def _poincare_sim(self, z_ref: torch.Tensor, z_set: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # z_ref: (B,D), z_set: (B,K,D) -> sims: (B,K)
        x = self._to_ball(z_ref, eps=eps).unsqueeze(1)
        y = self._to_ball(z_set, eps=eps)
        diff_sq = ((x - y) ** 2).sum(dim=-1)  # (B,K)
        x2 = (x * x).sum(dim=-1)              # (B,1)
        y2 = (y * y).sum(dim=-1)              # (B,K)
        denom = (1 - x2) * (1 - y2) + eps
        arg = 1.0 + 2.0 * diff_sq / denom
        dist = torch.acosh(arg.clamp(min=1.0 + eps))
        return -dist

    def _pairwise_poincare(self, z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        z_ball = self._to_ball(z, eps=eps)
        diff_sq = torch.cdist(z_ball, z_ball, p=2).pow(2)
        z2 = (z_ball * z_ball).sum(dim=-1, keepdim=True)
        denom = (1 - z2) @ (1 - z2).T + eps
        arg = 1.0 + 2.0 * diff_sq / denom
        dist = torch.acosh(arg.clamp(min=1.0 + eps))
        return -dist

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                            softmax_temp: float = 1.0, eps: float = 0.05):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)

        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low
            z_k = self.rep_trunk_targ(obs_rep, a_k).view(B, K, -1)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)

        a_anchor, logp, _ = self.actor.sample(obs_n)
        z_i = self.rep_trunk(obs_n, a_anchor)
        S = self._poincare_sim(z_i, z_k)

        tau_row = self._kernel_tau_instate(S, base_temp=softmax_temp)
        W = torch.softmax(S / tau_row, dim=1)
        if eps > 0.0:
            W = (1.0 - eps) * W + eps / K

        q_bar = (W.unsqueeze(-1) * qk).sum(dim=1, keepdim=True).detach()
        q_tilde = qk - q_bar
        Qhat = (W.unsqueeze(-1) * q_tilde).sum(dim=1)
        return Qhat, logp

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
            next_obs_n = self.obs_rms.normalize(next_obs)
        else:
            obs_n = obs
            next_obs_n = next_obs

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():
            a2, _, _ = self.actor.sample(next_obs_n)
            a2 = (a2 + torch.randn_like(a2) * self.noise_std).clamp(-1, 1)
            a2 = ((a2 + 1) / 2) * (self.high - self.low) + self.low
            z_next = self.rep_trunk_targ(next_obs_n, a2)

            q1t, q2t = self.q_targ(obs_n, act)
            q_targ = torch.min(q1t, q2t).squeeze(-1)

        S = self._pairwise_poincare(z)
        S_next = self._pairwise_poincare(z_next)

        u = q_targ.view(-1, 1)
        G = (u - u.T).abs()
        with torch.no_grad():
            beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
            beta = z.new_tensor(self.beta_ema.update(beta_batch) if self.beta_ema is not None else beta_batch)
        Delta = (G / beta).clamp(0.0, 1.0)
        T = 1.0 - 2.0 * (Delta ** float(self.rep_gamma_shape))

        alive = (1.0 - done.view(-1, 1)).to(S.dtype)
        Y = (1.0 - self.rep_lam) * T + self.rep_lam * alive * (self.gamma * S_next)

        mask = torch.ones_like(S, dtype=torch.bool)
        mask.fill_diagonal_(False)
        err = (S - Y)[mask]
        loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=self.rep_huber, reduction="mean")

        info = {
            "beta_batch": float(beta_batch),
            "beta_ema": float(beta),
            "mean_gap": float(G[mask].mean()),
            "mean_targets": float(Y[mask].mean()),
            "mean_sim": float(S[mask].mean()),
        }
        return loss, info


class DistAblationB7LaplacianKernel(DistAgent):
    """B7: Laplacian kernel exp(-||x-y||/sigma) for similarity."""

    def __init__(self, *args, **kwargs):
        self.laplace_sigma = kwargs.pop("laplace_sigma", 1.0)
        super().__init__(*args, **kwargs)

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                            softmax_temp: float = 1.0, eps: float = 0.05):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)

        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low
            z_k = self.rep_trunk_targ(obs_rep, a_k).view(B, K, -1)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)

        a_anchor, logp, _ = self.actor.sample(obs_n)
        z_i = self.rep_trunk(obs_n, a_anchor)
        dist = ((z_i.unsqueeze(1) - z_k) ** 2).sum(dim=-1).sqrt()
        S = torch.exp(-dist / self.laplace_sigma)

        tau_row = self._kernel_tau_instate(S, base_temp=softmax_temp)
        W = torch.softmax(S / tau_row, dim=1)
        if eps > 0.0:
            W = (1.0 - eps) * W + eps / K

        q_bar = (W.unsqueeze(-1) * qk).sum(dim=1, keepdim=True).detach()
        q_tilde = qk - q_bar
        Qhat = (W.unsqueeze(-1) * q_tilde).sum(dim=1)
        return Qhat, logp

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
            next_obs_n = self.obs_rms.normalize(next_obs)
        else:
            obs_n = obs
            next_obs_n = next_obs

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():
            a2, _, _ = self.actor.sample(next_obs_n)
            a2 = (a2 + torch.randn_like(a2) * self.noise_std).clamp(-1, 1)
            a2 = ((a2 + 1) / 2) * (self.high - self.low) + self.low
            z_next = self.rep_trunk_targ(next_obs_n, a2)

            q1t, q2t = self.q_targ(obs_n, act)
            q_targ = torch.min(q1t, q2t).squeeze(-1)

        dist = torch.cdist(z, z, p=2)
        dist_next = torch.cdist(z_next, z_next, p=2)
        S = torch.exp(-dist / self.laplace_sigma)
        S_next = torch.exp(-dist_next / self.laplace_sigma)

        u = q_targ.view(-1, 1)
        G = (u - u.T).abs()
        with torch.no_grad():
            beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
            beta = z.new_tensor(self.beta_ema.update(beta_batch) if self.beta_ema is not None else beta_batch)
        Delta = (G / beta).clamp(0.0, 1.0)
        T = 1.0 - 2.0 * (Delta ** float(self.rep_gamma_shape))

        alive = (1.0 - done.view(-1, 1)).to(S.dtype)
        Y = (1.0 - self.rep_lam) * T + self.rep_lam * alive * (self.gamma * S_next)

        mask = torch.ones_like(S, dtype=torch.bool)
        mask.fill_diagonal_(False)
        err = (S - Y)[mask]
        loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=self.rep_huber, reduction="mean")

        info = {
            "beta_batch": float(beta_batch),
            "beta_ema": float(beta),
            "mean_gap": float(G[mask].mean()),
            "mean_targets": float(Y[mask].mean()),
            "mean_sim": float(S[mask].mean()),
        }
        return loss, info


class DistAblationB8BilinearSim(DistAgent):
    """B8: Learned bilinear similarity z^T M z'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DistanceTrunk wraps an MLP; grab final linear layer output dim.
        out_dim = self.rep_trunk.net.net[-1].out_features
        self.metric_matrix = torch.nn.Parameter(torch.eye(out_dim, device=self.device))
        # Extend rep optimizer to include metric_matrix
        self.optim_rep = torch.optim.Adam(
            list(self.rep_trunk.parameters()) + [self.metric_matrix], lr=self.lr
        )

    def _metric(self):
        return 0.5 * (self.metric_matrix + self.metric_matrix.T)

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                            softmax_temp: float = 1.0, eps: float = 0.05):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)

        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low
            z_k = self.rep_trunk_targ(obs_rep, a_k).view(B, K, -1)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)

        a_anchor, logp, _ = self.actor.sample(obs_n)
        z_i = self.rep_trunk(obs_n, a_anchor)
        M = self._metric()
        # z_i (B,D), z_k (B,K,D) -> (B,K)
        S = torch.einsum('bd,dc,bkc->bk', z_i, M, z_k)

        tau_row = self._kernel_tau_instate(S, base_temp=softmax_temp)
        W = torch.softmax(S / tau_row, dim=1)
        if eps > 0.0:
            W = (1.0 - eps) * W + eps / K

        q_bar = (W.unsqueeze(-1) * qk).sum(dim=1, keepdim=True).detach()
        q_tilde = qk - q_bar
        Qhat = (W.unsqueeze(-1) * q_tilde).sum(dim=1)
        return Qhat, logp

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
            next_obs_n = self.obs_rms.normalize(next_obs)
        else:
            obs_n = obs
            next_obs_n = next_obs

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():
            a2, _, _ = self.actor.sample(next_obs_n)
            a2 = (a2 + torch.randn_like(a2) * self.noise_std).clamp(-1, 1)
            a2 = ((a2 + 1) / 2) * (self.high - self.low) + self.low
            z_next = self.rep_trunk_targ(next_obs_n, a2)

            q1t, q2t = self.q_targ(obs_n, act)
            q_targ = torch.min(q1t, q2t).squeeze(-1)

        M = self._metric()
        S = z @ M @ z.T
        S_next = z_next @ M @ z_next.T

        u = q_targ.view(-1, 1)
        G = (u - u.T).abs()
        with torch.no_grad():
            beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
            beta = z.new_tensor(self.beta_ema.update(beta_batch) if self.beta_ema is not None else beta_batch)
        Delta = (G / beta).clamp(0.0, 1.0)
        T = 1.0 - 2.0 * (Delta ** float(self.rep_gamma_shape))

        alive = (1.0 - done.view(-1, 1)).to(S.dtype)
        Y = (1.0 - self.rep_lam) * T + self.rep_lam * alive * (self.gamma * S_next)

        mask = torch.ones_like(S, dtype=torch.bool)
        mask.fill_diagonal_(False)
        err = (S - Y)[mask]
        loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=self.rep_huber, reduction="mean")

        info = {
            "beta_batch": float(beta_batch),
            "beta_ema": float(beta),
            "mean_gap": float(G[mask].mean()),
            "mean_targets": float(Y[mask].mean()),
            "mean_sim": float(S[mask].mean()),
        }
        return loss, info


class DistAblationB3NoCentering(DistAgent):
    """B3: Remove centering baseline q_bar."""

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                            softmax_temp: float = 1.0, eps: float = 0.05):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)

        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low
            z_k = F.normalize(self.rep_trunk_targ(obs_rep, a_k), p=2, dim=1)
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)

        a_anchor, logp, _ = self.actor.sample(obs_n)
        z_i = F.normalize(self.rep_trunk(obs_n, a_anchor), p=2, dim=1)
        z_k_view = z_k.view(B, K, -1)
        S = torch.einsum('bd,bkd->bk', z_i, z_k_view).clamp(-1.0, 1.0)
        tau_row = self._kernel_tau_instate(S, base_temp=softmax_temp)
        W = torch.softmax(S / tau_row, dim=1)
        if eps > 0.0:
            W = (1.0 - eps) * W + eps / K

        Qhat = (W.unsqueeze(-1) * qk).sum(dim=1)
        return Qhat, logp


class DistAblationB4CriticArgmax(DistAgent):
    """B4: Proposal set reduced to critic argmax approximation (pick best q_k)."""

    def _qhat_in_state_norm(self, obs, K: int = 64, noise_std: float = 0.1,
                            softmax_temp: float = 1.0, eps: float = 0.05):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs
        B = obs_n.shape[0]
        obs_rep = obs_n.repeat_interleave(K, dim=0)

        with torch.no_grad():
            a_k, _, _ = self.actor.sample(obs_rep)
            if noise_std > 0:
                a_k = (a_k + noise_std * torch.randn_like(a_k)).clamp(-1, 1)
                a_k = ((a_k + 1) / 2) * (self.high - self.low) + self.low
            q1k, q2k = self.q_targ(obs_rep, a_k)
            qk = torch.min(q1k, q2k).view(B, K, 1)
            q_max, _ = qk.max(dim=1)

        a_anchor, logp, _ = self.actor.sample(obs_n)
        Qhat = q_max  # (B,1)
        return Qhat, logp


class DistAblationB9NoAdaptiveTau(DistAgent):
    """B9: Fix kernel tau (no adapt, no anneal)."""

    def __init__(self, *args, **kwargs):
        kwargs["kernel_adaptive_tau"] = 0
        self.kernel_tau_fixed = kwargs.pop("kernel_tau_fixed", None)
        super().__init__(*args, **kwargs)

    def _kernel_tau_instate(self, S_rowwise: torch.Tensor, base_temp: float) -> torch.Tensor:
        B = S_rowwise.size(0)
        device = S_rowwise.device
        if self.kernel_tau_fixed is not None:
            tau = max(float(self.kernel_tau_fixed), 1e-6)
        else:
            tau = max(0.75, float(base_temp))
        return torch.full((B, 1), tau, device=device)


class DistAblationRewardOnly(DistAgent):
    """Reward-only ablation: drop critics; train rep/actor from rewards."""

    def __init__(self, *args, **kwargs):
        # True n-step reward return (discounted); computed from buffer trajectories.
        self.reward_nstep = max(1, int(kwargs.pop("reward_nstep", 1)))
        self.reward_scale = float(kwargs.pop("reward_scale", 1.0))
        self.normalize_reward = bool(kwargs.pop("normalize_reward", False))

        super().__init__(*args, **kwargs)

        # Freeze critics; they remain unused in this ablation.
        for p in self.qnet.parameters():
            p.requires_grad = False
        for p in self.q_targ.parameters():
            p.requires_grad = False

        self.reward_rms = None
        if self.normalize_reward:
            from dist_rl.utils import RunningMeanStd
            self.reward_rms = RunningMeanStd(1, device=self.device)

    def _reward_target(self, rew: torch.Tensor) -> torch.Tensor:
        return rew * self.reward_scale

    def _rep_loss(self, obs, act_env, next_obs, done, rew):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
            next_obs_n = self.obs_rms.normalize(next_obs)
        else:
            obs_n = obs
            next_obs_n = next_obs

        act = self._env_to_action(act_env)
        z = self.rep_trunk(obs_n, act)

        with torch.no_grad():
            a2, _, _ = self.actor.sample(next_obs_n)
            a2 = (a2 + torch.randn_like(a2) * self.noise_std).clamp(-1, 1)
            a2 = ((a2 + 1) / 2) * (self.high - self.low) + self.low
            z_next = self.rep_trunk_targ(next_obs_n, a2)

        reward_target = self._reward_target(rew)
        if self.reward_rms is not None:
            # Update running stats on detached rewards to avoid leaking gradients.
            self.reward_rms.update(reward_target.detach().unsqueeze(-1))
            reward_target = self.reward_rms.normalize(reward_target.unsqueeze(-1)).squeeze(-1)

        z_norm = F.normalize(z, p=2, dim=1)
        z_next_norm = F.normalize(z_next, p=2, dim=1)

        S = z_norm @ z_norm.T
        S_next = z_next_norm @ z_next_norm.T

        u = reward_target.view(-1, 1)
        G = (u - u.T).abs()
        with torch.no_grad():
            beta_batch = torch.quantile(G.reshape(-1), 0.95) + 1e-6
            beta = z.new_tensor(self.beta_ema.update(beta_batch) if self.beta_ema is not None else beta_batch)
        Delta = (G / beta).clamp(0.0, 1.0)
        T = 1.0 - 2.0 * (Delta ** float(self.rep_gamma_shape))

        alive = (1.0 - done.view(-1, 1)).to(S.dtype)
        gamma_n = self.gamma ** float(self.reward_nstep)
        Y = (1.0 - self.rep_lam) * T + self.rep_lam * alive * (gamma_n * S_next)

        mask = torch.ones_like(S, dtype=torch.bool)
        mask.fill_diagonal_(False)
        err = (S - Y)[mask]
        loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=self.rep_huber, reduction="mean")

        info = {
            "beta_batch": float(beta_batch),
            "beta_ema": float(beta),
            "mean_gap": float(G[mask].mean()),
            "mean_targets": float(Y[mask].mean()),
            "mean_cos": float(S[mask].mean()),
            "reward_nstep": int(self.reward_nstep),
            "reward_scale": float(self.reward_scale),
        }
        return loss, info

    def _actor_alpha_loss(self, obs, act_env, rew):
        if self.normalize_obs:
            obs_n = self.obs_rms.normalize(obs)
        else:
            obs_n = obs

        a_taken = self._env_to_action(act_env).clamp(-1, 1)
        logp = self.actor.log_prob(obs_n, a_taken, eps=1e-6).unsqueeze(-1)

        reward_target = self._reward_target(rew)
        if self.reward_rms is not None:
            reward_target = self.reward_rms.normalize(reward_target.unsqueeze(-1)).squeeze(-1)

        advantage = reward_target - reward_target.mean()
        actor_loss = -(advantage.detach().unsqueeze(-1) * logp).mean() + (self.log_alpha.exp() * logp).mean()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

        if wandb.run is not None:
            wandb.log({
                "train/actor_reward_mean": float(reward_target.mean().item()),
                "train/actor_adv_mean": float(advantage.mean().item()),
            }, step=self.steps)

        return actor_loss, alpha_loss

    def train(self):
        print(
            f"[Train-RewardOnly] Starting training for {self.total_steps} steps (eval every {self.eval_freq})")
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
                    f"[Train-RewardOnly] Warmup complete at step {self.steps}, switching to policy actions")
                self._printed_warmup_notice = True

            if done or trunc:
                print(
                    f"[Train-RewardOnly] Episode: return={ep_r:.2f}, length={ep_len}, total_steps={self.steps}")
                if wandb.run is not None:
                    wandb.log({"rollout/ep_reward": ep_r, "rollout/ep_len": ep_len,
                              "step": self.steps}, step=self.steps)
                ep_r = 0.0
                ep_len = 0
                o, _ = self.env.reset()

            if self.steps < max(self.warmup_steps, self.batch_size):
                continue

            for _ in range(self.updates_per_step):
                if self.reward_nstep > 1:
                    obs, next_obs, act_env, rew, done_b = self.replay.get_nstep_batch(
                        self.batch_size, self.reward_nstep, self.gamma)
                else:
                    obs, next_obs, act_env, rew, done_b = self.replay.get_batch(
                        self.batch_size)

                # Rep update (reward-based)
                rep_loss, rep_info = self._rep_loss(
                    obs, act_env, next_obs, done_b, rew)
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

                # Actor + alpha update (REINFORCE-style with reward target)
                actor_loss, alpha_loss = self._actor_alpha_loss(obs, act_env, rew)

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
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [self.log_alpha], self.max_grad_norm)
                    self.alpha_opt.step()
                    logs["train/alpha"] = float(self.log_alpha.exp().item())
                    logs["train/alpha_loss"] = float(alpha_loss.item())
                    logs["train/alpha_grad_norm"] = float(grad_norm)
                else:
                    logs["train/alpha"] = float(self.alpha)
                if wandb.run is not None:
                    wandb.log(logs, step=self.steps)

                # Only update rep trunk target; critics remain frozen.
                polyak_update(self.rep_trunk_targ, self.rep_trunk, self.tau)

            if (self.steps % self.eval_freq) == 0:
                print(f"[Train-RewardOnly] Evaluation at step {self.steps}")
                self.evaluate()

        self.env.close()
        self.eval_env.close()
        if wandb.run is not None:
            wandb.run.finish()

        print("[Train-RewardOnly] Training complete!")
        return


class DistAblationRewardOnlyN64(DistAblationRewardOnly):
    """Reward-only ablation with reward_nstep fixed to 64."""

    def __init__(self, *args, **kwargs):
        kwargs["reward_nstep"] = 64
        super().__init__(*args, **kwargs)
