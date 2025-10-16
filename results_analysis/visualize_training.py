"""Plot training curves for all environments in a single figure of subplots."""

from pathlib import Path
import argparse
import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"env", "algo", "seed", "step", "eval_reward"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.dropna(subset=["env", "algo", "step", "eval_reward"])


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a summary of the dataset showing algorithms and seeds per environment.
    
    Args:
        df: DataFrame with columns [env, algo, seed, step, eval_reward]
    """
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    # Overall statistics
    total_rows = len(df)
    total_envs = df['env'].nunique()
    total_algos = df['algo'].nunique()
    total_seeds = df['seed'].nunique()
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total data points:  {total_rows:,}")
    print(f"   Environments:       {total_envs}")
    print(f"   Algorithms:         {total_algos}")
    print(f"   Unique seeds:       {total_seeds}")
    
    # Per-environment breakdown
    print(f"\n🔍 Per-Environment Breakdown:")
    print(f"{'Environment':<30s} {'Algorithms':<20s} {'Seeds':<10s} {'Runs':<10s}")
    print("-" * 70)
    
    for env in sorted(df['env'].unique()):
        env_df = df[df['env'] == env]
        algos = sorted(env_df['algo'].unique())
        seeds = sorted(env_df['seed'].unique())
        
        # Count runs (unique algo-seed combinations)
        runs = env_df.groupby(['algo', 'seed']).ngroups
        
        # Format algorithm names
        algo_str = ', '.join(algos) if len(algos) <= 3 else f"{algos[0]}, ... ({len(algos)} total)"
        
        print(f"{env:<30s} {algo_str:<20s} {len(seeds):<10d} {runs:<10d}")
    
    # Algorithm-wise summary
    print(f"\n🤖 Algorithm-wise Summary:")
    print(f"{'Algorithm':<20s} {'Environments':<15s} {'Seeds':<10s} {'Total Runs':<12s}")
    print("-" * 70)
    
    for algo in sorted(df['algo'].unique()):
        algo_df = df[df['algo'] == algo]
        envs = algo_df['env'].nunique()
        seeds = sorted(algo_df['seed'].unique())
        runs = algo_df.groupby(['env', 'seed']).ngroups
        
        print(f"{algo:<20s} {envs:<15d} {len(seeds):<10d} {runs:<12d}")
    
    # Detailed run matrix
    print(f"\n📋 Run Matrix (Algo × Env):")
    
    # Create pivot table showing number of seeds per algo-env combination
    run_matrix = df.groupby(['algo', 'env'])['seed'].nunique().unstack(fill_value=0)
    
    # Print with nice formatting
    print("\n" + run_matrix.to_string())
    
    print("\n" + "="*70 + "\n")


def print_performance_tables(df: pd.DataFrame) -> None:
    """Print performance tables showing max reward statistics per algorithm per environment.
    
    Args:
        df: DataFrame with columns [env, algo, seed, step, eval_reward]
    """
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY - Maximum Reward Statistics")
    print("="*70)
    
    # Get maximum reward for each (env, algo, seed) combination
    max_rewards = df.groupby(['env', 'algo', 'seed'])['eval_reward'].max().reset_index()
    max_rewards.columns = ['env', 'algo', 'seed', 'max_reward']
    
    # Calculate mean and std across seeds for each (env, algo) combination
    stats = max_rewards.groupby(['env', 'algo'])['max_reward'].agg(['mean', 'std', 'count']).reset_index()
    
    # Print table for each environment
    for env in sorted(df['env'].unique()):
        print(f"\n📊 {env}")
        print("-" * 70)
        
        env_stats = stats[stats['env'] == env].copy()
        
        if len(env_stats) == 0:
            print("   No data available")
            continue
        
        # Sort by mean reward (descending)
        env_stats = env_stats.sort_values('mean', ascending=False)
        
        # Print header
        print(f"{'Algorithm':<20s} {'Mean Max Reward':<20s} {'Std':<15s} {'Seeds':<10s}")
        print("-" * 70)
        
        # Print each algorithm's stats
        for _, row in env_stats.iterrows():
            algo = row['algo']
            mean_val = row['mean']
            std_val = row['std'] if not pd.isna(row['std']) else 0.0
            count = int(row['count'])
            
            mean_str = f"{mean_val:>10.2f}"
            std_str = f"± {std_val:>8.2f}" if count > 1 else "N/A"
            
            print(f"{algo:<20s} {mean_str:<20s} {std_str:<15s} {count:<10d}")
    
    # Print overall summary table (all environments combined)
    print(f"\n" + "="*70)
    print("OVERALL SUMMARY (All Environments)")
    print("="*70)
    print(f"{'Algorithm':<20s} {'Avg Max Reward':<20s} {'Overall Std':<15s} {'Total Runs':<12s}")
    print("-" * 70)
    
    overall_stats = max_rewards.groupby('algo')['max_reward'].agg(['mean', 'std', 'count']).reset_index()
    overall_stats = overall_stats.sort_values('mean', ascending=False)
    
    for _, row in overall_stats.iterrows():
        algo = row['algo']
        mean_val = row['mean']
        std_val = row['std'] if not pd.isna(row['std']) else 0.0
        count = int(row['count'])
        
        mean_str = f"{mean_val:>10.2f}"
        std_str = f"± {std_val:>8.2f}"
        
        print(f"{algo:<20s} {mean_str:<20s} {std_str:<15s} {count:<12d}")
    
    print("\n" + "="*70 + "\n")


def apply_ema_smoothing(df: pd.DataFrame, smoothing_weight: float) -> pd.DataFrame:
    """Apply exponential moving average smoothing using pandas ewm.
    
    Args:
        df: DataFrame with columns [env, algo, seed, step, eval_reward]
        smoothing_weight: Smoothing factor (0 < smoothing_weight < 1).
                         This is converted to pandas ewm 'alpha' parameter.
                         Higher smoothing_weight = more smoothing.
                         pandas alpha = 1 - smoothing_weight (inverted for intuitive behavior)
    """
    if not (0.0 < smoothing_weight < 1.0):
        return df
    
    # Convert our smoothing_weight to pandas ewm alpha parameter
    # In pandas ewm: alpha closer to 1 = less smoothing (more weight to recent values)
    # We invert it so higher smoothing_weight = more smoothing
    pandas_alpha = 1.0 - smoothing_weight
    
    # Sort by step and apply ewm smoothing to eval_reward column
    # Group by (env, algo, seed) to smooth each run independently
    df = df.sort_values(['env', 'algo', 'seed', 'step']).copy()
    df['eval_reward'] = df.groupby(['env', 'algo', 'seed'])['eval_reward'].transform(
        lambda x: x.ewm(alpha=pandas_alpha, adjust=False).mean()
    )
    
    return df
    
    return df


def visualize(csv_path: Path,
              output_dir: Path,
              show: bool=False,
              ema_alpha: float=0.0,
              max_step: int=1_000_000,
              print_summary: bool=True) -> None:
    
    df = load_results(csv_path)
    
    envs = sorted(df["env"].unique())
    if not envs:
        raise ValueError("No environments found in results file")
    
    #drop algorithms on list
    drop_algos = ['SACDistanceAgentNew']
    df = df[~df["algo"].isin(drop_algos)]
    
    #drop envs on list
    drop_envs = ['LunarLanderContinuous-v3', 'MountainCarContinuous-v0']
    envs = [env for env in envs if env not in drop_envs]
    
    
    # Print data summary
    if print_summary:
        print_data_summary(df)
        print_performance_tables(df)
    
    if max_step is not None:
        df = df[df["step"] <= max_step]
    if 0.0 < ema_alpha < 1.0:
        df = apply_ema_smoothing(df, smoothing_weight=ema_alpha)
        


    sns.set_theme(style="whitegrid", context="talk", palette="tab10")

    n_env = len(envs)
    ncols = min(3, n_env)
    nrows = math.ceil(n_env / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    legend_handles = legend_labels = None

    for idx, env in enumerate(envs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        env_df = df[df["env"] == env]

        sns.lineplot(
            data=env_df,
            x="step",
            y="eval_reward",
            hue="algo",
            estimator="mean",
            errorbar="sd",
            ax=ax,
        )

        ax.set_title(env)
        ax.set_xlabel("Step")
        ax.set_ylabel("Evaluation Reward")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Hide any unused axes
    total_axes = nrows * ncols
    for idx in range(len(envs), total_axes):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncols=min(len(legend_labels), 4),
            frameon=False,
            fontsize="medium",
        )

    title = "\nTraining Curves Across Environments"
    if 0.0 < ema_alpha < 1.0:
        title += f" (EMA α={ema_alpha:.2f})"
    fig.suptitle(title, fontsize="x-large", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_dir.mkdir(parents=True, exist_ok=True)
 
    output_file = output_dir / "training_overview.png"
    print(f"Saving plot to {output_file}")
    
    fig.savefig(output_file, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from CSV results")
    parser.add_argument("--csv", type=Path, default=Path("results_analysis/data/results_full.csv"),
                        help="Path to results_full.csv")
    parser.add_argument("--out", type=Path, default=Path("results_analysis/plots"),
                        help="Directory to save plot images")
    parser.add_argument("--show", default=False, action="store_true", help="Display plots interactively")
    parser.add_argument("--ema-alpha", type=float, default=0.0,
                        help="EMA smoothing factor (0-1). Higher=more smoothing. 0=disabled. Recommended: 0.1-0.5")
    parser.add_argument("--max-step", type=int, default=1_000_000,
                        help="Maximum training step to plot (set negative to include all)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Disable printing dataset summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_step = args.max_step if args.max_step >= 0 else None
    visualize(args.csv,
              args.out,
              show=args.show,
              ema_alpha=args.ema_alpha,
              max_step=max_step,
              print_summary=not args.no_summary)


if __name__ == "__main__":
    main()
