import argparse
import wandb

from distRL import DistanceAgent


def main():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    
    # wandb args
    parser.add_argument("--lightweight_wandb", action="store_false", default=True,
                        help="If true, wandb will not save the code.")                        
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--project_name", type=str, default="DistRL")
    parser.add_argument("--log_to_wandb", action="store_true", default=False,
                        help="If true, logs will be sent to wandb.")
    
    #algorithm args
    parser.add_argument("--algo", type=str, default="DistRL")        
    parser.add_argument("--total-steps", type=int, default=200_000)    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-epochs-train", type=int, default=1)    
    parser.add_argument("--update-epochs-val", type=int, default=1)    
    parser.add_argument("--buffer-size", type=int, default=100_000)    
    # parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--policy-training-start", type=int, default=500)
    parser.add_argument("--val-training-start", type=int, default=500)
    args = parser.parse_args()    
    
    exp_prefix = args.exp_prefix + f"{args.algo}-seed{args.seed}"
    group_name = args.env_id
    
    if args.log_to_wandb:
        wandb_run = wandb.init(
            name=args.exp_prefix,
            group=group_name,
            id=exp_prefix,
            project=args.project_name,
            entity='stavrosorf',
            save_code= (not args.lightweight_wandb),
            config=args,                    
        )
        
        if not args.lightweight_wandb:
            # wandb.run.log_code(".")
            wandb_run.log_code("distRL.py")

    if args.algo == "DistRL":
                        
        agent = DistanceAgent(
            env_id=args.env_id,
            seed=args.seed,
            device=args.device,
            
            total_steps=args.total_steps,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            update_epochs_train=args.update_epochs_train,
            update_epochs_val=args.update_epochs_val,            
            lr=args.lr,
            hidden_size=args.hidden_size,
            eval_episodes=args.eval_episodes,
            wandb_run=wandb_run if args.log_to_wandb else None,
            policy_training_start=args.policy_training_start,
            val_training_start=args.val_training_start
            
        )
                
    agent.train()


if __name__ == "__main__":
    main()
