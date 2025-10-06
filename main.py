import argparse
from datetime import datetime
from pathlib import Path

import wandb
import yaml
import torch

from dist_rl.twin_distRL import DistanceAgent
# from dist_rl.distRL import DistanceAgent
from dist_rl.stoch_distRL import StochasticDistanceAgent
from classic_rl.sb3_train import train_sb3_agent
from dist_rl.utils import load_hyperparameters

SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str,
                        default="Hopper-v5")
    parser.add_argument("--optimal-run", action="store_true", default=False,
                        help="If true, uses optimal hyperparameters for the environment.")
    # parser.add_argument("--env-id", type=str, default="Pendulum-v1")
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
    parser.add_argument("--algo", type=str, default="DistRL")
    # parser.add_argument("--algo", type=str, default="DistRL")
    parser.add_argument("--model-save-path", type=str, default="./saved_models/")
    parser.add_argument("--noise-type", type=str, default="Scheduled")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--comp-samples", type=int, default=4096)
    parser.add_argument("--update-epochs-policy", type=int, default=1)
    parser.add_argument("--update-epochs-val", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--rtg-enabled", action="store_true", default=False,
                        help="If true, use RTG as returns, else use n-step returns.")
    parser.add_argument("--expl-sigma", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--policy-training-start", type=int, default=1000)
    parser.add_argument("--val-training-start", type=int, default=1000)
    parser.add_argument("--q-percentile", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--v_gamma", type=float, default=1)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--value-model-type", type=str, default="LSTM",
                        help='"LSTM" or "Transformer"')
    parser.add_argument("--dynamic-beta", action="store_true", default=False,
                        help="If true, beta will be set dynamically based on the max reward gap in the batch.")
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--policy-noise-clip", type=float, default=0.5)
    parser.add_argument("--actor-update-frequency", type=int, default=1)

    args = parser.parse_args()

    # check if cuda is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        args.device = "cpu"

    if args.algo in SB3_ALGOS:
        group_name = args.group_name + args.env_id + "_SB3"
        # args.device = "cpu"  # SB3 algorithms run on CPU by default
        args.log_to_wandb = True  # always log sb3 runs to wandb
        exp_prefix = args.algo + "_SB3_seed=" + \
            str(args.seed) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        exp_prefix = args.exp_prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        group_name = args.group_name + args.env_id + "_testv1"
            
    model_save_path = Path(args.model_save_path) / exp_prefix
    model_save_path.mkdir(parents=True, exist_ok=True)
    args.model_save_path = str(model_save_path)    
    

    if args.log_to_wandb:
        wandb_run = wandb.init(
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

        args.wandb_run = wandb_run
    else:
        args.wandb_run = None
        
    print("="*65)
    print(f"Training with {args.algo} on {args.env_id} with seed {args.seed} on {args.device}")
    
    #load optimal hyperparameters if specified
    if args.optimal_run and args.algo in ["DistRL", "StochDistRL"]:
        algo_name = args.algo.lower().replace("stoch", "stoch_").replace("dist", "dist_")
        params = load_hyperparameters(args.env_id, algo_name)
        print("Loaded optimal hyperparameters:")
        print(params)
        
        args.__dict__.update(params)
        print("Updated args:")
        print(args.__dict__)

    if args.algo == "DistRL":        
        agent = DistanceAgent(**args.__dict__)
    elif args.algo == "StochDistRL":        
        agent = StochasticDistanceAgent(**args.__dict__)
    else:
        agent = train_sb3_agent(**args.__dict__)

    if args.algo not in SB3_ALGOS:
        agent.train()


if __name__ == "__main__":
    main()
