import argparse
from datetime import datetime
from pathlib import Path

import wandb
import yaml
import torch

from dist_rl.dist_agent import DistAgent
from dist_rl.ablations.agents import (
    DistAblationA1RandomEncoder,
    DistAblationA2ActorOnlyEncoder,
    DistAblationA3NoTemporalMix,
    DistAblationA4NoBetaScaling,
    DistAblationA5GammaFixed,
    DistAblationB1UniformKernel,
    DistAblationB2EuclideanSim,
    DistAblationB3NoCentering,
    DistAblationB4CriticArgmax,
    DistAblationB5FixedK,
    DistAblationB6PoincareSim,
    DistAblationB7LaplacianKernel,
    DistAblationB8BilinearSim,
)
from classic_rl.sb3_train import train_sb3_agent
from dist_rl.utils import set_seed

SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]
MUJOCO_ENVS = ['HalfCheetah-v5', 'Ant-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
               'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5']  # number of envs: 9
BOX2D_ENVS = ['LunarLanderContinuous-v3',
              'MountainCarContinuous-v0', 'Pendulum-v1']  # number of envs: 3
CLASSIC_ENVS = ['CartPole-v1', 'Acrobot-v1']
continuous_envs = MUJOCO_ENVS + BOX2D_ENVS

ABLATION_AGENTS = {
    "DistAblationA1": DistAblationA1RandomEncoder,
    "DistAblationA2": DistAblationA2ActorOnlyEncoder,
    "DistAblationA3": DistAblationA3NoTemporalMix,
    "DistAblationA4": DistAblationA4NoBetaScaling,
    "DistAblationA5": DistAblationA5GammaFixed,
    "DistAblationB1": DistAblationB1UniformKernel,
    "DistAblationB2": DistAblationB2EuclideanSim,
    "DistAblationB3": DistAblationB3NoCentering,
    "DistAblationB4": DistAblationB4CriticArgmax,
    "DistAblationB5": DistAblationB5FixedK,
    "DistAblationB6": DistAblationB6PoincareSim,
    "DistAblationB7": DistAblationB7LaplacianKernel,
    "DistAblationB8": DistAblationB8BilinearSim,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str,
                        # default="ALE/Breakout-v5", help="Gym environment ID")
                        default="HalfCheetah-v5", help="Gym environment ID")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # wandb args
    parser.add_argument("--lightweight_wandb", action="store_false", default=True,
                        help="If true, wandb will not save the code.")
    parser.add_argument("--exp-prefix", type=str, default="test")
    parser.add_argument("--group-name", type=str, default="")
    parser.add_argument("--project_name", type=str, default="DistRL ")
    parser.add_argument("--log_to_wandb", action="store_true", default=False,
                        help="If true, logs will be sent to wandb.")

    # algorithm args
    parser.add_argument("--algo", type=str, default="DistAgent") # DistAgent
    parser.add_argument("--save-dir", type=str,
                        default="./saved_models/")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument('--warmup-steps', type=int, default=5000,
                        help='Number of warmup steps for learning rate scheduling.')
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--expl-sigma", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument('--alpha', type=float, default=None,
                        help='If None, autotune alpha.')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)

    # Representation Loss args
    parser.add_argument('--rep-gamma-shape', type=float, default=0.5)
    parser.add_argument('--rep-lam', type=float, default=0.5)
    parser.add_argument('--rep-huber', type=float, default=0.2)
    parser.add_argument('--rep-fixed-scale', type=float, default=1.0,
                        help='Fixed beta scale for A4 ablation (ignored otherwise).')

    parser.add_argument('--normalize-obs', type=int, default=1,
                        help='Whether to normalize observations (1=True, 0=False).')

    # Actor Training args
    parser.add_argument('--updates-per-step', type=int, default=1,
                        help='Number of optimization rounds per environment step.')
    parser.add_argument('--target-entropy-scale', type=float, default=0.9,
                        help='Multiplier applied to -action_dim when computing the entropy target.')
    parser.add_argument('--kernel-adaptive-tau', type=int, default=1,
                        help='Whether to adapt kernel temperature per batch (1=True, 0=False).')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--fixed-K', dest='fixed_K', type=int, default=64,
                        help='Fixed K for B5 ablation (ignored otherwise).')

    args = parser.parse_args()

    # check if cuda is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        args.device = "cpu"

    set_seed(args.seed)

    if args.algo in SB3_ALGOS:
        group_name = args.group_name + args.env_id
        args.log_to_wandb = True  # always log sb3 runs to wandb
        exp_prefix = args.algo + "_SB3_seed=" + \
            str(args.seed) + "_" + datetime.now().strftime("%m%d_%H%M%S")
    else:
        exp_prefix = args.exp_prefix + "_" + datetime.now().strftime("%m%d_%H%M%S")
        group_name = args.group_name + args.env_id

    args.exp_prefix = exp_prefix
    
    model_save_path = Path(args.save_dir) / exp_prefix
    model_save_path.mkdir(parents=True, exist_ok=True)
    args.model_save_path = str(model_save_path)

    if args.log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            sync_tensorboard=True if args.algo in SB3_ALGOS else False,
            id=exp_prefix,
            project=args.project_name,
            entity='stavrosorf',
            save_code=(not args.lightweight_wandb),
            config=args,
        )

        if not args.lightweight_wandb:
            wandb.run.log_code(".")

    print("="*65)
    print(
        f"Training with {args.algo} on {args.env_id} with seed {args.seed} on {args.device}")

    if args.algo == "DistAgent" and args.env_id in continuous_envs:
        agent = DistAgent(**args.__dict__)
        print(f'Running {args.algo} with kernel policy updates.')
        agent.train()
    elif args.algo in ABLATION_AGENTS and args.env_id in continuous_envs:
        agent_cls = ABLATION_AGENTS[args.algo]
        agent = agent_cls(**args.__dict__)
        print(f'Running {args.algo} ablation agent on {args.env_id}.')
        agent.train()
    else:
        agent = train_sb3_agent(**args.__dict__)


if __name__ == "__main__":
    main()
