"""Command line entry point for training the discrete Distance RL agent on Atari."""
from __future__ import annotations

import argparse

import torch
import wandb

from discreteDistRL.dist_agent import AgentConfig, DiscreteDistAgent
from discreteDistRL.utils import set_seed
from discreteDistRL.wrappers import make_atari_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the discrete Distance RL agent on Atari.")
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5", help="Gymnasium Atari environment ID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda", help="Device identifier (cuda or cpu).")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Number of environment steps.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes for evaluation rollouts.")
    parser.add_argument("--eval-freq", type=int, default=100_000, help="Evaluation frequency (steps).")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Target smoothing factor.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for all optimizers.")
    parser.add_argument("--updates-per-step", type=int, default=1, help="Gradient updates per environment step.")
    parser.add_argument("--warmup-steps", type=int, default=50_000, help="Number of random-policy steps before updates.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for epsilon-greedy exploration.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon value.")
    parser.add_argument("--epsilon-decay", type=int, default=250_000, help="Steps to anneal epsilon.")
    parser.add_argument(
        "--policy-smoothing-eps",
        type=float,
        default=0.2,
        help="Amount of uniform mixing applied to policy logits for noisy action proposals.",
    )
    parser.add_argument(
        "--proposal-samples",
        type=int,
        default=32,
        help="Number of action proposals drawn per state for kernel Q aggregation.",
    )
    parser.add_argument(
        "--kernel-softmax-temp",
        type=float,
        default=1.0,
        help="Base temperature for the proposal similarity softmax kernel.",
    )
    parser.add_argument(
        "--kernel-eps",
        type=float,
        default=0.05,
        help="Epsilon smoothing term applied to kernel weights for stability.",
    )
    parser.add_argument(
        "--no-kernel-adaptive-tau",
        action="store_true",
        help="Disable adaptive adjustment of kernel temperature using similarity variance.",
    )
    parser.add_argument("--frames", type=int, default=4, help="Number of stacked frames.")
    parser.add_argument("--no-sticky", action="store_true", help="Disable sticky actions (repeat probability 0.25).")
    parser.add_argument("--no-clip", action="store_true", help="Disable reward clipping to [-1, 1].")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--project", type=str, default="distance-rl", help="Weights & Biases project name.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    set_seed(args.seed)

    env = make_atari_env(
        args.env_id,
        seed=args.seed,
        frames=args.frames,
        sticky=not args.no_sticky,
        clip_rewards=not args.no_clip,
    )
    eval_env = make_atari_env(
        args.env_id,
        seed=args.seed + 1,
        frames=args.frames,
        sticky=not args.no_sticky,
        clip_rewards=not args.no_clip,
    )

    cfg = AgentConfig(
        env_id=args.env_id,
        seed=args.seed,
        device=device,
        total_steps=args.total_steps,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        updates_per_step=args.updates_per_step,
        warmup_steps=args.warmup_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        policy_smoothing_eps=args.policy_smoothing_eps,
        proposal_samples=args.proposal_samples,
        kernel_softmax_temp=args.kernel_softmax_temp,
        kernel_eps=args.kernel_eps,
        kernel_adaptive_tau=not args.no_kernel_adaptive_tau,
        save_dir=args.save_dir,
    )

    if not args.no_wandb:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    agent = DiscreteDistAgent(env, eval_env, cfg)
    agent.train()

    env.close()
    eval_env.close()
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
