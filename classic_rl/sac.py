import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from .td3 import ReplayBuffer, set_seed


LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.base = mlp(obs_dim, hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return (self.action_high + self.action_low) / 2.0 + raw_action * (self.action_high - self.action_low) / 2.0

    def forward(self, obs: torch.Tensor):
        h = self.base(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = self._squash(y_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        mean_action = self._squash(torch.tanh(mean))
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.q_net = mlp(obs_dim + act_dim, hidden_size, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q_net(torch.cat([obs, actions], dim=-1)).squeeze(-1)


class SACAgent:
    def __init__(
        self,
        env_id: str,
        seed: int,
        total_steps: int,
        buffer_size: int,
        batch_size: int,
        start_steps: int,
        gamma: float,
        tau: float,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        hidden_size: int,
        eval_interval: int,
        eval_episodes: int,
        device: str = "cpu",
        wandb_run=None,
    ):
        set_seed(seed)
        self.device = torch.device(device)
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)

        self.total_steps = total_steps
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma = gamma
        self.tau = tau
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.wandb_run = wandb_run

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_size, self.action_low, self.action_high).to(self.device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_size).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_size).to(self.device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_size).to(self.device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_size).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(act_dim)

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, self.device)
        self.min_action = torch.as_tensor(self.action_low, dtype=torch.float32, device=self.device)
        self.max_action = torch.as_tensor(self.action_high, dtype=torch.float32, device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _evaluate(self, step: int):
        returns = []
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    _, _, mean_action = self.policy(obs_tensor)
                action = mean_action.cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            returns.append(total_reward)
        if self.wandb_run is not None:
            self.wandb_run.log({"eval/episode_return": np.mean(returns)}, step=step)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def _update(self, step: int):
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action, log_prob, _ = self.policy(next_obs)
            next_q1 = self.q1_target(next_obs, next_action)
            next_q2 = self.q2_target(next_obs, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * log_prob
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        q_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_action, log_prob, _ = self.policy(obs)
        q1_pi = self.q1(obs, new_action)
        q2_pi = self.q2(obs, new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "train/critic_loss": q_loss.item(),
                    "train/actor_loss": policy_loss.item(),
                    "train/alpha": self.alpha.item(),
                },
                step=step,
            )

    def train(self):
        obs, _ = self.env.reset()
        global_step = 0
        episode_return = 0.0
        episode_length = 0

        while global_step < self.total_steps:
            if global_step < self.start_steps:
                action = self.env.action_space.sample()
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = self.policy(obs_tensor)
                action = action.cpu().numpy()[0]
            action = np.clip(action, self.action_low, self.action_high)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            episode_length += 1
            global_step += 1

            if done:
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "rollout/episode_return": episode_return,
                            "rollout/episode_length": episode_length,
                        },
                        step=global_step,
                    )
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0

            if global_step >= self.start_steps:
                self._update(global_step)

            if global_step % self.eval_interval == 0:
                self._evaluate(global_step)

        if self.wandb_run is not None:
            self._evaluate(global_step)

