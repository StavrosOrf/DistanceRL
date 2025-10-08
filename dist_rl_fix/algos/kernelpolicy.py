import torch, wandb
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class KernelCfg:
    temp: float = 0.5
    cand: int = 2048
    state_k: int = 64
    use_adv: bool = True

class KernelPolicyMixin:
    @staticmethod
    def attach(agent, temp=0.5, cand=2048, state_k=64, use_adv=True):
        from types import MethodType

        agent.kernel_cfg = KernelCfg(temp=temp, cand=cand, state_k=state_k, use_adv=use_adv)
        # Bind mixin methods to the agent instance
        agent._kernel_q_hat = MethodType(KernelPolicyMixin._kernel_q_hat, agent)
        agent._train_kernel_loop = MethodType(KernelPolicyMixin._train_kernel_loop, agent)
        agent.train_kernel = agent._train_kernel_loop  # exposed entrypoint

    def _kernel_q_hat(self, obs_i, act_i):
        cfg = self.kernel_cfg

        obs_c, act_env_c, rtg_c, nret_c, _ = self.replay.sample_candidates(cfg.cand)
        obs_c_n = self.obs_rms.normalize(obs_c)
        act_c = self._env_to_action(act_env_c)  # [-1,1]

        obs_i_n = self.obs_rms.normalize(obs_i)
        s_i = F.normalize(obs_i_n, p=2, dim=1)
        s_c = F.normalize(obs_c_n, p=2, dim=1)
        S_state = s_i @ s_c.T  # (B, M)

        k = min(cfg.state_k, S_state.size(1))
        top_vals, top_idx = torch.topk(S_state, k=k, dim=1, largest=True)

        obs_c_k = torch.gather(obs_c_n.unsqueeze(0).expand(obs_i.size(0), -1, -1), 1, top_idx.unsqueeze(-1).expand(-1, -1, obs_c_n.size(1)))
        act_c_k = torch.gather(act_c.unsqueeze(0).expand(obs_i.size(0), -1, -1), 1, top_idx.unsqueeze(-1).expand(-1, -1, act_c.size(1)))
        rtg_c_k = torch.gather(rtg_c.unsqueeze(0).expand(obs_i.size(0), -1), 1, top_idx)

        B, K = obs_i.size(0), k
        z_i = self.trunk(obs_i_n, act_i)
        z_i = F.normalize(z_i, p=2, dim=1)
        z_c = self.trunk(obs_c_k.reshape(B*K, -1), act_c_k.reshape(B*K, -1))
        z_c = F.normalize(z_c, p=2, dim=1).reshape(B, K, -1)

        S = (z_i.unsqueeze(1) * z_c).sum(-1)
        S = S / max(1e-6, cfg.temp)
        W = torch.softmax(S, dim=1)

        Q_hat = (W * rtg_c_k).sum(dim=1, keepdim=True)
        b_i = Q_hat.detach()
        if cfg.use_adv:
            A = rtg_c_k - b_i
            Q_hat = (W * (A + b_i)).sum(dim=1, keepdim=True)

        return Q_hat, {"kernel/avg_top_state_sim": float(top_vals.mean().item())}

    def _train_kernel_loop(self):
        agent = self
        cfg = agent.kernel_cfg
        env = agent.env

        o, _ = env.reset()
        ep_r = 0.0; ep_len = 0
        while agent.steps < agent.total_steps:
            ot = torch.as_tensor(o, device=agent.device, dtype=torch.float32).unsqueeze(0)
            agent.obs_rms.update(ot)
            otn = agent.obs_rms.normalize(ot)

            with torch.no_grad():
                if agent.steps < agent.warmup_steps:
                    a_env = env.action_space.sample()
                else:
                    a, _, _ = agent.actor.sample(otn)
                    a_env = ((a.clamp(-1,1)+1)/2)*(agent.high-agent.low)+agent.low
                    a_env = a_env.squeeze(0).cpu().numpy()

            o2, r, done, trunc, _ = env.step(a_env)
            agent.replay.add(o, o2, a_env, r, done or trunc)
            agent.steps += 1
            ep_r += r; ep_len += 1
            o = o2

            if done or trunc:
                if wandb.run is not None:
                    wandb.log({"rollout/ep_reward": ep_r, "rollout/ep_len": ep_len, "step": agent.steps}, step=agent.steps)
                ep_r = 0.0; ep_len = 0
                o, _ = env.reset()

            if agent.steps < max(agent.warmup_steps, agent.batch_size): 
                continue

            obs, act_env, rew, next_obs, done_b, rtg, nret, _ = agent.replay.sample(agent.batch_size)
            act = agent._env_to_action(act_env)

            if agent.rep_loss_weight > 0.0:
                rep_loss, rep_info = agent._rep_loss(obs, act, next_obs, done_b, nret)
                agent.optim_q.zero_grad(); (agent.rep_loss_weight * rep_loss).backward()
                torch.nn.utils.clip_grad_norm_(list(agent.trunk.parameters()), 10.0)
                agent.optim_q.step()
                rep_logs = {f"rep/{k}": v for k, v in rep_info.items()}
                rep_logs.update({"rep/loss": float(rep_loss.item()), "step": agent.steps})
                if wandb.run is not None:
                    wandb.log(rep_logs, step=agent.steps)

            obs_n = agent.obs_rms.normalize(obs)
            a, logp, _ = agent.actor.sample(obs_n)
            Q_hat, info = self._kernel_q_hat(obs, a)
            if agent._alpha_fixed:
                alpha = agent.alpha
                actor_loss = (alpha * logp - Q_hat).mean()
                alpha_loss = None
            else:
                alpha = agent.log_alpha.exp()
                actor_loss = (alpha * logp - Q_hat).mean()
                alpha_loss = -(agent.log_alpha * (logp + agent.target_entropy).detach()).mean()

            agent.optim_actor.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
            agent.optim_actor.step()
            logs = {"train/actor_loss": float(actor_loss.item()), "train/alpha": float(alpha), "step": agent.steps}
            if wandb.run is not None:
                wandb.log({**logs, **info}, step=agent.steps)
            if alpha_loss is not None:
                agent.alpha_opt.zero_grad(); alpha_loss.backward(); agent.alpha_opt.step()
                
                
                if wandb.run is not None:
                    wandb.log({"train/alpha": float(agent.log_alpha.exp().item()), "step": agent.steps}, step=agent.steps)

            if (agent.steps % agent.eval_freq) == 0:
                agent.evaluate()
                agent._save("ckpt")
