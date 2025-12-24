import gymnasium as gym
import numpy as np
import torch
import time
import sys
import wandb
from baselines.redq.algos.redq_sac import REDQSACAgent
from baselines.redq.algos.core import mbpo_epoches, test_agent
from baselines.redq.utils.run_utils import setup_logger_kwargs
from baselines.redq.utils.bias_utils import log_bias_evaluation
from baselines.redq.utils.logx import EpochLogger


class _NoOpLogger:
    def store(self, **kwargs):
        return None

def _unwrap_reset(env):
    res = env.reset()
    return res[0] if isinstance(res, tuple) else res


def _step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        o2, r, terminated, truncated, _ = step_out
        done = terminated or truncated
    else:
        o2, r, done, _ = step_out
        truncated = False
    return o2, r, done, truncated


class REDQMainAgent:
    """
    Lightweight runner to plug REDQ into main.py with REDQ paper defaults.
    Exposes only the shared knobs from main (env_id, seed, total_steps, eval_freq, eval_episodes, wandb setup).
    """

    _DEFAULTS = dict(
        hidden_sizes=(256, 256),
        replay_size=int(1e6),
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        auto_alpha=True,
        target_entropy='mbpo',
        start_steps=5000,
        delay_update_steps='auto',
        utd_ratio=20,
        num_Q=10,
        num_min=2,
        q_target_mode='min',
        policy_update_delay=20,
    )

    def __init__(self,
                 env_id: str,
                 seed: int,
                 device: str,
                 total_steps: int,
                 eval_episodes: int,
                 eval_freq: int,
                 exp_prefix: str,
                 lightweight_wandb: bool = False,
                 **_):

        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        self.env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 1)
        self.env.action_space.seed(seed)
        self.eval_env.action_space.seed(seed + 1)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device(device)
        self.total_steps = total_steps
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.exp_prefix = exp_prefix
        self.steps = 0
        self.max_ep_len = self.env._max_episode_steps
        self.best_eval = -float('inf')
        self.lightweight_wandb = bool(lightweight_wandb)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        act_limit = self.env.action_space.high[0].item()

        defaults = self._DEFAULTS
        self.agent = REDQSACAgent(env_name=env_id,
                                  obs_dim=obs_dim,
                                  act_dim=act_dim,
                                  act_limit=act_limit,
                                  device=self.device,
                                  **defaults)

        self._logger = _NoOpLogger()

    def _log_rollout(self, ep_ret: float, ep_len: int):
        if wandb.run is not None:
            wandb.log({"rollout/ep_reward": ep_ret,
                       "rollout/ep_len": ep_len,
                       "step": self.steps}, step=self.steps)

    def _log_eval(self, avg_reward: float, avg_len: float):
        if wandb.run is not None:
            wandb.log({"eval/avg_reward": avg_reward,
                       "eval/avg_len": avg_len,
                       "step": self.steps}, step=self.steps)

    def evaluate(self):
        total_r = 0.0
        lens = []
        for _ in range(self.eval_episodes):
            o = _unwrap_reset(self.eval_env)
            ep_len = 0
            done = False
            while not done:
                a = self.agent.get_test_action(o)
                o, r, done, trunc = _step_env(self.eval_env, a)
                done = done or trunc
                total_r += r
                ep_len += 1
            lens.append(ep_len)

        avg_r = total_r / max(self.eval_episodes, 1)
        avg_len = sum(lens) / max(len(lens), 1)
        if avg_r > self.best_eval:
            self.best_eval = avg_r
            print(f"[REDQ] New best eval avg_return={avg_r:.2f}, avg_len={avg_len:.1f}")
        self._log_eval(avg_r, avg_len)
        return avg_r

    def train(self):
        print(f"[REDQ] Training for {self.total_steps} steps (eval every {self.eval_freq})")
        o = _unwrap_reset(self.env)
        ep_ret = 0.0
        ep_len = 0

        for _ in range(self.total_steps):
            a = self.agent.get_exploration_action(o, self.env)
            o2, r, done, trunc = _step_env(self.env, a)
            ep_len += 1
            time_limit = (self.max_ep_len is not None) and (ep_len >= self.max_ep_len)
            d_flag = False if time_limit else done

            self.agent.store_data(o, a, r, o2, d_flag)
            self.agent.train(self._logger)

            o = o2
            ep_ret += r
            self.steps += 1

            if done or trunc or time_limit:
                print(f"[REDQ] Rollout finished: ep_ret={ep_ret:.2f}, ep_len={ep_len}")
                self._log_rollout(ep_ret, ep_len)
                o = _unwrap_reset(self.env)
                ep_ret = 0.0
                ep_len = 0

            if (self.steps % self.eval_freq) == 0:
                print(f"[REDQ] Eval at step {self.steps}")
                self.evaluate()

        # Cleanup at end of training
        self.env.close()
        self.eval_env.close()
        if wandb.run is not None:
            wandb.finish()

