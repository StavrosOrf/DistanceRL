import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_init(layer, std=1.0, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# -----------------------------
# Actor-Critic model
# -----------------------------
class ActorCritic(nn.Module):
    """
    Actor: Gaussian policy with Tanh squashing (and change-of-variable corrected log_prob).
    Critic: State-value network.
    Designed for 1D continuous action (Pendulum: [-2, 2]).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, action_low=-2.0, action_high=2.0):
        super().__init__()
        assert act_dim == 1, "This quick test targets 1D action space (Pendulum)."

        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # Policy network outputs mean and log_std (state-independent log_std for simplicity)
        self.pi_body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden), std=math.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden), std=math.sqrt(2)),
            nn.Tanh(),
        )
        self.mu = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # trainable log std

        # Value function
        self.v = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden), std=math.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden), std=math.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def _tanh_squash(self, u: torch.Tensor) -> torch.Tensor:
        # Squash to (-1, 1), then scale/shift to action bounds
        return torch.tanh(u) * self.action_scale + self.action_bias

    def _log_prob_tanh_normal(self, u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        # log prob with tanh correction: log_prob(u) - sum log(1 - tanh(u)^2)
        std = torch.exp(log_std)
        base_log_prob = -0.5 * (((u - mean) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))
        base_log_prob = base_log_prob.sum(-1)  # sum over action dims
        # Change of variables for tanh: |det(J)| = prod(1 - tanh(u)^2)
        # Add small epsilon for numerical stability
        epsilon = 1e-6
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + epsilon).sum(-1)
        return base_log_prob - correction

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: action, log_prob, value, mean_action (deterministic)
        """
        h = self.pi_body(obs)
        mean = self.mu(h)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        # Reparameterization: u = mean + std * eps, then a = tanh(u) scaled
        eps = torch.randn_like(mean)
        u = mean + std * eps
        action = self._tanh_squash(u)
        log_prob = self._log_prob_tanh_normal(u, mean, log_std)

        value = self.v(obs).squeeze(-1)
        with torch.no_grad():
            mean_action = self._tanh_squash(mean)
        return action, log_prob, value, mean_action

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For PPO updates: compute log_probs of given actions and entropy from the current policy.
        Convert actions back to pre-squash space via atanh; then compute corrected log prob.
        """
        # Map actions a in [low, high] to pre-tanh u via atanh
        a = (actions - self.action_bias) / (self.action_scale + 1e-8)
        a = torch.clamp(a, -0.999999, 0.999999)  # stay inside domain
        u = 0.5 * torch.log((1 + a) / (1 - a))  # atanh

        h = self.pi_body(obs)
        mean = self.mu(h)
        log_std = self.log_std.expand_as(mean)

        log_prob = self._log_prob_tanh_normal(u, mean, log_std)
        # Entropy of tanh-normal is not closed-form; use base normal entropy as a proxy (common practice)
        std = torch.exp(log_std)
        base_entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)
        value = self.v(obs).squeeze(-1)
        return log_prob, base_entropy, value


# -----------------------------
# Rollout Buffer with GAE(λ)
# -----------------------------
@dataclass
class TrajectoryBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device):
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, action, logprob, reward, done, value):
        assert self.ptr < self.max_size, "Buffer overflow"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> TrajectoryBatch:
        advantages = torch.zeros_like(self.rewards, device=self.device)
        last_gae = 0.0
        for t in reversed(range(self.max_size)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.max_size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch = TrajectoryBatch(
            obs=self.obs.clone(),
            actions=self.actions.clone(),
            logprobs=self.logprobs.clone(),
            returns=returns.clone(),
            advantages=advantages.clone(),
            values=self.values.clone(),
        )
        # Reset pointer for next rollout
        self.ptr = 0
        return batch


# -----------------------------
# PPO Agent
# -----------------------------
class PPOAgent:
    def __init__(
        self,
        env_id="Pendulum-v1",
        seed=0,
        total_steps=200_000,
        rollout_steps=2048,
        update_epochs=10,
        minibatch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.00,
        max_grad_norm=0.5,
        lr=3e-4,
        hidden=64,
        device="cpu",
        eval_episodes=5,
    ):
        set_seed(seed)
        self.device = torch.device(device)

        # Env
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        assert self.env.action_space.shape == (1,)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))

        # Model
        self.ac = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            action_low=float(self.env.action_space.low[0]),
            action_high=float(self.env.action_space.high[0]),
        ).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        # Buffer + hyperparams
        self.buffer = RolloutBuffer(rollout_steps, obs_dim, act_dim, self.device)
        self.total_steps = total_steps
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.eval_episodes = eval_episodes

        # Sanity checks
        assert rollout_steps % minibatch_size == 0, "rollout_steps should be divisible by minibatch_size"

    def collect_rollout(self):
        obs, info = self.env.reset(seed=None)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        ep_returns = []
        ep_return = 0.0

        for _ in range(self.rollout_steps):
            obs_in = obs.unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value, _ = self.ac.act(obs_in)
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

            np_action = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(np_action)
            done = terminated or truncated

            self.buffer.add(obs, action, log_prob, float(reward), float(done), value)

            ep_return += float(reward)
            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                next_obs, _ = self.env.reset()

            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        # Bootstrap value for last state
        with torch.no_grad():
            last_value = self.ac.v(obs.unsqueeze(0)).squeeze(0).item()

        batch = self.buffer.compute_returns_advantages(last_value, self.gamma, self.gae_lambda)
        return batch, (np.mean(ep_returns) if ep_returns else None)

    def ppo_update(self, batch: TrajectoryBatch):
        b_inds = np.arange(self.rollout_steps)
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.rollout_steps, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = batch.obs[mb_inds]
                mb_actions = batch.actions[mb_inds]
                mb_old_logprobs = batch.logprobs[mb_inds]
                mb_advantages = batch.advantages[mb_inds]
                mb_returns = batch.returns[mb_inds]

                new_logprobs, entropy, values = self.ac.evaluate_actions(mb_obs, mb_actions)

                # PPO objective
                logratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(logratio)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                with torch.no_grad():
                    v_clipped = batch.values[mb_inds] + torch.clamp(values - batch.values[mb_inds],
                                                                   -self.clip_coef, self.clip_coef)
                v_loss_unclipped = (values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * (-entropy_loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def evaluate(self, episodes: int = 5) -> float:
        returns = []
        for _ in range(episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    _, _, _, mean_action = self.ac.act(obs_t)
                action = mean_action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                ep_ret += float(reward)
            returns.append(ep_ret)
        return float(np.mean(returns))

    def train(self):
        steps_collected = 0
        log_every = max(1, self.total_steps // 50)

        while steps_collected < self.total_steps:
            batch, avg_tr_ret = self.collect_rollout()
            print(f"Collected {self.rollout_steps} steps, avg_train_ep_ret={avg_tr_ret if avg_tr_ret is not None else float('nan'):.1f}")
            self.ppo_update(batch)
            steps_collected += self.rollout_steps

            if steps_collected % log_every == 0 or steps_collected >= self.total_steps:
                eval_ret = self.evaluate(self.eval_episodes)
                print(f"[{steps_collected:>7}/{self.total_steps}] "
                      f"avg_train_ep_ret={avg_tr_ret if avg_tr_ret is not None else float('nan'):.1f}  "
                      f"eval_ret={eval_ret:.1f}  "
                      f"mean_adv={batch.advantages.mean().item():+.3f}")

        # Final evaluation
        final_eval = self.evaluate(self.eval_episodes)
        print(f"Training complete. Final eval return over {self.eval_episodes} episodes: {final_eval:.1f}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Quick custom PPO for Gymnasium Pendulum-v1")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    agent = PPOAgent(
        env_id=args.env_id,
        seed=args.seed,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        eval_episodes=args.eval_episodes,
    )
    agent.train()


if __name__ == "__main__":
    main()
