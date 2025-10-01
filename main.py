import argparse
from datetime import datetime
from pathlib import Path

import wandb
import yaml

from distRL import DistanceAgent
from classic_rl import make_agent


def load_classic_hyperparams(algo: str, env_id: str) -> dict:
    hyperparam_path = Path(__file__).parent / "classic_rl" / "hyperparams" / f"{algo.lower()}.yaml"
    if not hyperparam_path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found for {algo}: {hyperparam_path}")

    with open(hyperparam_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if env_id not in data:
        raise KeyError(f"No hyperparameters specified for env {env_id} and algo {algo}")

    return data[env_id]


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--env-id", type=str, default="LunarLanderContinuous-v2")
    # parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # wandb args
    parser.add_argument("--lightweight_wandb", action="store_false", default=True,
                        help="If true, wandb will not save the code.")
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--project_name", type=str, default="DistRL")
    parser.add_argument("--log_to_wandb", action="store_true", default=False,
                        help="If true, logs will be sent to wandb.")

    # algorithm args
    parser.add_argument("--algo", type=str, default="DistRL")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--update-epochs-policy", type=int, default=1)
    parser.add_argument("--update-epochs-val", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    # parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--policy-training-start", type=int, default=2000)
    parser.add_argument("--val-training-start", type=int, default=2000)
    parser.add_argument("--v_gamma", type=float, default=1.2)    
    parser.add_argument("--value-model-type", type=str, default="LSTM",
                        help='"LSTM" or "Transformer"')
    parser.add_argument("--dynamic-beta", action="store_true", default=False,
                        help="If true, beta will be set dynamically based on the max reward gap in the batch.")
    
    args = parser.parse_args()

    classic_hparams = None
    if args.algo != "DistRL":
        classic_hparams = load_classic_hyperparams(args.algo, args.env_id)
        for key, value in classic_hparams.items():
            setattr(args, key, value)

    exp_prefix = args.exp_prefix + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    group_name = args.env_id + "_test"

    if args.log_to_wandb:
        wandb_run = wandb.init(
            name=exp_prefix,
            group=group_name,
            id=exp_prefix,
            project=args.project_name,
            entity='stavrosorf',
            save_code=(not args.lightweight_wandb),
            config=args,
        )

        if not args.lightweight_wandb:
            wandb.run.log_code(".")
            # wandb_run.log_code("distRL.py")

        args.wandb_run = wandb_run
    else:
        args.wandb_run = None

    if args.algo == "DistRL":
        agent = DistanceAgent(**args.__dict__)
    else:
        if classic_hparams is None:
            raise RuntimeError("Classic hyperparameters were not loaded.")
        agent_kwargs = {**classic_hparams,
                        "env_id": args.env_id,
                        "seed": args.seed,
                        "device": args.device,
                        "wandb_run": args.wandb_run}
        agent = make_agent(args.algo, **agent_kwargs)

    agent.train()


if __name__ == "__main__":
    main()
