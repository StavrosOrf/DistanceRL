# dist_rl_fix/algos/kernel_policy.py  (updated)
import torch, wandb
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class KernelCfg:
    temp: float = 0.5
    cand: int = 2048
    state_k: int = 64
    use_adv: bool = True
    adaptive_tau: bool = True
    use_q_targets: bool = True   # << use critics' Q(s_c,a_c), not returns

class KernelPolicyMixin:
    @staticmethod
    def attach(agent, temp=0.5, cand=2048, state_k=64, use_adv=True, adaptive_tau=True, use_q_targets=True):
        from types import MethodType
        agent.kernel_cfg = KernelCfg(temp=temp, cand=cand, state_k=state_k, use_adv=use_adv,
                                     adaptive_tau=adaptive_tau, use_q_targets=use_q_targets)
        agent._kernel_q_hat = MethodType(KernelPolicyMixin._kernel_q_hat, agent)
        agent._train_kernel_loop = MethodType(KernelPolicyMixin._train_kernel_loop, agent)
        agent.train_kernel = agent._train_kernel_loop

    def _kernel_q_hat(self, obs_i, act_i):
        cfg = self.kernel_cfg

        obs_c, act_env_c, rtg_c, nret_c, _ = self.replay.sample_candidates(cfg.cand)
        obs_c_n = self.obs_rms.normalize(obs_c)
        act_c = self._env_to_action(act_env_c)

        obs_i_n = self.obs_rms.normalize(obs_i)
        s_i = F.normalize(obs_i_n, p=2, dim=1)
        s_c = F.normalize(obs_c_n, p=2, dim=1)
        S_state = s_i @ s_c.T

        k = min(cfg.state_k, S_state.size(1))
        top_vals, top_idx = torch.topk(S_state, k=k, dim=1, largest=True)

        B, K = obs_i.size(0), k
        obs_c_k = torch.gather(obs_c_n.unsqueeze(0).expand(B, -1, -1), 1,
                               top_idx.unsqueeze(-1).expand(-1, -1, obs_c_n.size(1)))
        act_c_k = torch.gather(act_c.unsqueeze(0).expand(B, -1, -1), 1,
                               top_idx.unsqueeze(-1).expand(-1, -1, act_c.size(1)))

        # rep features for weights
        z_i = F.normalize(self.rep_trunk(obs_i_n, act_i), p=2, dim=1)             # (B,H)
        z_c = self.rep_trunk(obs_c_k.reshape(B*K, -1), act_c_k.reshape(B*K, -1))
        z_c = F.normalize(z_c, p=2, dim=1).reshape(B, K, -1)                      # (B,K,H)
        S = (z_i.unsqueeze(1) * z_c).sum(-1)                                      # (B,K)

        if cfg.adaptive_tau:
            sstd = S.std(dim=1, keepdim=True) + 1e-6
            W = torch.softmax(S / (cfg.temp * sstd), dim=1)
        else:
            W = torch.softmax(S / max(1e-6, cfg.temp), dim=1)

        # targets: critics' Q rather than returns (much better bias)
        if cfg.use_q_targets:
            q1c, q2c = self.qnet(obs_c_k.reshape(B*K, -1), act_c_k.reshape(B*K, -1))
            tgt = torch.min(q1c, q2c).reshape(B, K)
        else:
            tgt = rtg_c.gather(0, top_idx.reshape(-1)).reshape(B, K)  # fallback

        Q_hat = (W * tgt).sum(dim=1, keepdim=True)                    # (B,1)
        b_i = Q_hat.detach()
        if cfg.use_adv:
            A = tgt - b_i                                             # broadcast (B,K)
            Q_hat = (W * (A + b_i)).sum(dim=1, keepdim=True)

        return Q_hat, {"kernel/avg_top_state_sim": float(top_vals.mean().item())}

    def _train_kernel_loop(self):
        env = self.env
        o, _ = env.reset()
        ep_r = 0.0; ep_len = 0
        while self.steps < self.total_steps:
            ot = torch.as_tensor(o, device=self.device, dtype=torch.float32).unsqueeze(0)
            self.obs_rms.update(ot)
            otn = self.obs_rms.normalize(ot)

            with torch.no_grad():
                if self.steps < self.warmup_steps:
                    a_env = env.action_space.sample()
                else:
                    a, _, _ = self.actor.sample(otn)
                    a_env = ((a.clamp(-1, 1) + 1) / 2) * (self.high - self.low) + self.low
                    a_env = a_env.squeeze(0).cpu().numpy()

            o2, r, done, trunc, _ = env.step(a_env)
            self.replay.add(o, o2, a_env, r, done or trunc)
            self.steps += 1
            ep_r += r; ep_len += 1
            o = o2

            if done or trunc:
                if wandb.run is not None:
                    wandb.log({"rollout/ep_reward": ep_r, "rollout/ep_len": ep_len, "step": self.steps}, step=self.steps)
                ep_r = 0.0; ep_len = 0
                o, _ = env.reset()

            if self.steps < max(self.warmup_steps, self.batch_size):
                continue

            obs, act_env, rew, next_obs, done_b, rtg, nret, _ = self.replay.sample(self.batch_size)
            act = self._env_to_action(act_env)

            # optional rep loss
            if self.rep_loss_weight > 0.0:
                rep_loss, rep_info = self._rep_loss(obs, act_env, next_obs, done_b, nret)
                self.optim_rep.zero_grad(); (self.rep_loss_weight * rep_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.rep_trunk.parameters(), 10.0)
                self.optim_rep.step()
                rep_logs = {f"rep/{k}": v for k, v in rep_info.items()}
                rep_logs.update({"rep/loss": float(rep_loss.item()), "step": self.steps})
                if wandb.run is not None:
                    wandb.log(rep_logs, step=self.steps)

            # actor + alpha with kernel Q target
            obs_n = self.obs_rms.normalize(obs)
            a, logp, _ = self.actor.sample(obs_n)
            Q_hat, info = self._kernel_q_hat(obs, a)
            if self._alpha_fixed:
                alpha = self.alpha
                actor_loss = (alpha * logp - Q_hat).mean()
                alpha_loss = None
            else:
                alpha = self.log_alpha.exp()
                actor_loss = (alpha * logp - Q_hat).mean()
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

            self.optim_actor.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.optim_actor.step()

            if wandb.run is not None:
                wandb.log({"train/actor_loss": float(actor_loss.item()), "train/alpha": float(alpha), **info, "step": self.steps}, step=self.steps)

            if alpha_loss is not None:
                self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
                if wandb.run is not None:
                    wandb.log({"train/alpha": float(self.log_alpha.exp().item()), "step": self.steps}, step=self.steps)

            if (self.steps % self.eval_freq) == 0:
                self.evaluate()
                self._save("ckpt")
