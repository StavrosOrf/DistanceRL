import argparse
from datetime import datetime
from pathlib import Path

import wandb
import yaml
import torch

# from dist_rl.twin_distRL import DistanceAgent
from dist_rl.distRL import DistanceAgent
from dist_rl.stoch_distRL import StochasticDistanceAgent
from classic_rl.sb3_train import train_sb3_agent
from dist_rl.utils import load_hyperparameters
# from dist_rl.offlineRL_utils import load_minari_dataset_into_buffer

from dist_rl_fix.algos.sac_distance import SACDistanceAgent
from dist_rl_fix.algos.kernelpolicy import KernelPolicyMixin
from dist_rl_fix.algos.sac_distance_new import SACDistanceAgentNew
from dist_rl_fix.algos.sac_wasserstein import SACWassersteinAgent
from dist_rl_fix.utils import set_seed

SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str,
                        default="HalfCheetah-v5", help="Gym environment ID")
    parser.add_argument("--dataset", type=str, default="mujoco/halfcheetah/expert-v0",
                        help="Minari dataset name (e.g., mujoco/halfcheetah/expert-v0)")
    parser.add_argument("--max-dataset-episodes", type=int, default=None,
                        help="Maximum number of episodes to load from dataset")
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
    parser.add_argument("--algo", type=str, default="SACDistanceAgentNew")
    # parser.add_argument("--algo", type=str, default="DistRL")
    parser.add_argument("--model-save-path", type=str,
                        default="./saved_models/")
    parser.add_argument("--noise-type", type=str, default="Scheduled")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--comp-samples", type=int, default=10)
    parser.add_argument("--update-epochs-policy", type=int, default=1)
    parser.add_argument("--update-epochs-val", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--rtg-enabled", action="store_true", default=False,
                        help="If true, use RTG as returns, else use n-step returns.")
    parser.add_argument("--expl-sigma", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--policy-training-start", type=int, default=1000)
    parser.add_argument("--val-training-start", type=int, default=1000)
    parser.add_argument("--q-percentile", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dynamic-topk", action="store_true", default=True,
                        help="If true, top-k increases linearly from 5 to top_k over half training.")
    parser.add_argument("--v_gamma", type=float, default=1)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--policy-noise-clip", type=float, default=0.5)
    parser.add_argument("--actor-update-frequency", type=int, default=1)

    parser.add_argument('--alpha', type=float, default=None,
                        help='If None, autotune alpha.')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    
    parser.add_argument('--rep-loss-weight', type=float, default=0.1)
    parser.add_argument('--rep-gamma-shape', type=float, default=0.5)
    parser.add_argument('--rep-lam', type=float, default=0.5)
    parser.add_argument('--rep-huber', type=float, default=0.2)
    parser.add_argument('--rep-margin-scale', type=float, default=0.5)
    parser.add_argument('--rep-temp', type=float, default=1.0)

    parser.add_argument('--kernel-temp', type=float, default=0.5)
    parser.add_argument('--kernel-cand', type=int, default=10)
    parser.add_argument('--kernel-state-k', type=int, default=64)
    parser.add_argument('--kernel-adv', action='store_true',
                        help='Use advantage (recommended).')
    
    parser.add_argument('--updates-per-step', type=int, default=1,
                        help='Number of optimization rounds per environment step.')
    parser.add_argument('--target-entropy-scale', type=float, default=0.9,
                        help='Multiplier applied to -action_dim when computing the entropy target.')
    parser.add_argument('--alpha-cql', type=float, default=0.0,
                        help='Weight for the Conservative Q-Learning regularizer (0 disables).')
    parser.add_argument('--kernel-aux-weight', type=float, default=0.1,
                        help='Weight for the optional kernel auxiliary loss added to the actor.')
    parser.add_argument('--kernel-adaptive-tau', type=int, default=1,
                        help='Whether to adapt kernel temperature per batch (1=True, 0=False).')
    parser.add_argument('--logdir', type=str, default='./logs')
    
    # add the following parser args
    parser.add_argument('--ot-eta', type=float, default=0.1)
    parser.add_argument('--ot-eps', type=float, default=0.05)
    parser.add_argument('--ot-iters', type=int, default=10)
    parser.add_argument('--ot-K', type=int, default=16)
    parser.add_argument('--ot-Kt', type=int, default=32)
    parser.add_argument('--ot-std-scale', type=float, default=1.5)
    parser.add_argument('--ot-topk-target', type=bool, default=True)

    args = parser.parse_args()

    # check if cuda is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        args.device = "cpu"

    if args.max_dataset_episodes is not None:
        print(
            f'Offline training using up to {args.max_dataset_episodes} episodes from dataset {args.dataset}')
        # offline training only
        group_name = args.group_name + args.env_id + "_offline"
        exp_prefix = args.exp_prefix + "_" + args.env_id + \
            "_offline_" + datetime.now().strftime("%m%d_%H%M%S")
    else:
        if args.algo in SB3_ALGOS:
            group_name = args.group_name + args.env_id#+ "_SB3"
            # args.device = "cpu"  # SB3 algorithms run on CPU by default
            args.log_to_wandb = True  # always log sb3 runs to wandb
            exp_prefix = args.algo + "_SB3_seed=" + \
                str(args.seed) + "_" + datetime.now().strftime("%m%d_%H%M%S")
        else:
            exp_prefix = args.exp_prefix + "_" + datetime.now().strftime("%m%d_%H%M%S")
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
    print(
        f"Training with {args.algo} on {args.env_id} with seed {args.seed} on {args.device}")

    # load optimal hyperparameters if specified
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

        if args.max_dataset_episodes is not None:
            # Load Minari dataset into buffer
            # dataset_stats = load_minari_dataset_into_buffer(
            #     dataset_name=args.dataset,
            #     buffer=agent.buffer,
            #     device=args.device,
            #     max_episodes=args.max_dataset_episodes,
            # )
            raise NotImplementedError("Minari dataset loading not implemented.")

    elif args.algo == "StochDistRL":
        agent = StochasticDistanceAgent(**args.__dict__)

    elif "sacDistRL" in args.algo:
        set_seed(args.seed)

        setattr(args, 'n_step', args.K)  # for representation loss
        setattr(args, 'hidden', args.hidden_size)  # for representation loss
        setattr(args, 'alpha', args.alpha)  # for representation loss        
        setattr(args, 'save_dir', args.model_save_path)

        agent = SACDistanceAgent(**args.__dict__)
        if args.algo == 'sacDistRL':
            agent.train_sac()
        else:
            print(f'Running {args.algo} with kernel policy updates.')
            # Kernel policy update loop
            KernelPolicyMixin.attach(agent, temp=args.kernel_temp, cand=args.kernel_cand,
                                     state_k=args.kernel_state_k, use_adv=args.kernel_adv)
            agent.train_kernel()

    elif args.algo == "SACDistanceAgentNew":
        set_seed(args.seed)

        setattr(args, 'rep_gamma_shape', args.v_gamma)  # for representation loss        
        setattr(args, 'alpha', args.alpha)  # for representation loss        
        setattr(args, 'save_dir', args.model_save_path)
        
        agent = SACDistanceAgentNew(**args.__dict__)

        print(f'Running {args.algo} with kernel policy updates.')
    
    elif "sacWasserstein" in args.algo:
        set_seed(args.seed)

        setattr(args, 'n_step', args.K)  # for representation loss
        setattr(args, 'hidden', args.hidden_size)  # for representation loss
        setattr(args, 'save_dir', args.model_save_path)
        
        agent = SACWassersteinAgent(**args.__dict__)

        print(f'Running {args.algo} with kernel policy updates.')
    
    else:
        agent = train_sb3_agent(**args.__dict__)

    if args.algo not in SB3_ALGOS and args.max_dataset_episodes is None:
        agent.train()
    elif args.algo in SB3_ALGOS:
        pass  # training is done in train_sb3_agent()
    else:
        agent.train_offline()


if __name__ == "__main__":
    main()
