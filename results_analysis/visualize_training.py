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
    
    


    # Print max step per algo per env
    max_steps = df.groupby(['env', 'algo'])['step'].max().reset_index()
    max_steps.columns = ['env', 'algo', 'max_step']
    print(max_steps)

    if max_step is not None:
        df = df[df["step"] <= max_step]
        
        
        # Print data summary
    if print_summary:
        print_data_summary(df)
        print_performance_tables(df)
        
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


def generate_latex_table(csv_path: Path,
                        output_dir: Path,
                        target_envs: list = None,
                        drop_algos: list = None,
                        max_step: int = None) -> None:
    """Generate LaTeX table with max average results (mean ± std) for each algorithm-environment pair.
    
    Args:
        csv_path: Path to results CSV file
        output_dir: Directory to save the LaTeX table
        target_envs: List of environments to include (None = all)
        drop_algos: List of algorithms to exclude (None = none)
        max_step: Maximum training step to include (None = all)
    """
    df = load_results(csv_path)
    
    # Drop unwanted algorithms
    if drop_algos:
        df = df[~df["algo"].isin(drop_algos)]
    
    # Filter by max step
    if max_step is not None:
        df = df[df["step"] <= max_step]
    
    # Filter environments if specified (otherwise use all)
    if target_envs:
        df = df[df["env"].isin(target_envs)]
    
    if len(df) == 0:
        print("Warning: No data found for LaTeX table generation")
        return
    
    # Rename algorithms for better display
    df['algo'] = df['algo'].replace({
        'sac': 'SAC',
        'td3': 'TD3',
        'ppo': 'PPO',
        'tqc': 'TQC',
    })
    
    # Get maximum reward for each (env, algo, seed) combination
    max_rewards = df.groupby(['env', 'algo', 'seed'])['eval_reward'].max().reset_index()
    max_rewards.columns = ['env', 'algo', 'seed', 'max_reward']
    
    # Calculate mean and std across seeds for each (env, algo) combination
    stats = max_rewards.groupby(['env', 'algo'])['max_reward'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate total (average across all environments) for each algorithm
    total_stats = max_rewards.groupby('algo')['max_reward'].agg(['mean', 'std']).reset_index()
    
    # Pivot to create table: rows=environments, columns=algorithms (VERTICAL)
    pivot_mean = stats.pivot(index='env', columns='algo', values='mean')
    pivot_std = stats.pivot(index='env', columns='algo', values='std')
    
    # Sort environments
    envs = sorted(pivot_mean.index)
    
    # Sort algorithms by total performance (ascending), but put DistAgent last
    total_sorted = total_stats.sort_values('mean', ascending=True)  # Worst to best
    algos = total_sorted['algo'].tolist()
    
    # Move DistAgent to the end if it exists
    if 'DistAgent' in algos:
        algos.remove('DistAgent')
        algos.append('DistAgent')
    
    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    
    # Add max_step info to caption if applicable
    caption = "Maximum average evaluation returns across environments. Results are reported as mean $\\pm$ std over seeds"
    if max_step is not None:
        caption += f" (up to {max_step:,} steps)"
    caption += "."
    
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append("\\label{tab:results}")
    
    # Table format: l for environment name, r for each algorithm
    table_format = "l" + "r" * len(algos)
    latex_lines.append(f"\\begin{{tabular}}{{{table_format}}}")
    latex_lines.append("\\toprule")
    
    # Header row: Environment & Algo1 & Algo2 & ...
    header = "\\textbf{Environment} & " + " & ".join([f"\\textbf{{{algo}}}" for algo in algos]) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Environment display names (cleaner for paper)
    env_names = {
        'Ant-v5': 'Ant',
        'HalfCheetah-v5': 'HalfCheetah',
        'Walker2d-v5': 'Walker2d',
        'Humanoid-v5': 'Humanoid',
        'Ant-v4': 'Ant',
        'HalfCheetah-v4': 'HalfCheetah',
        'Walker2d-v4': 'Walker2d',
        'Humanoid-v4': 'Humanoid',
        'Hopper-v4': 'Hopper',
        'Hopper-v5': 'Hopper',
        'Swimmer-v4': 'Swimmer',
        'Swimmer-v5': 'Swimmer',
    }
    
    # Data rows: one row per environment
    for env in envs:
        env_display_name = env_names.get(env, env)
        row_values = []
        
        for algo in algos:
            mean_val = pivot_mean.loc[env, algo] if algo in pivot_mean.columns and env in pivot_mean.index else None
            std_val = pivot_std.loc[env, algo] if algo in pivot_std.columns and env in pivot_std.index else None
            
            if pd.notna(mean_val):
                if pd.notna(std_val):
                    # Format: mean ± std
                    cell = f"${mean_val:.0f} \\pm {std_val:.0f}$"
                else:
                    cell = f"${mean_val:.0f}$"
            else:
                cell = "---"
            
            row_values.append(cell)
        
        row = f"{env_display_name} & " + " & ".join(row_values) + " \\\\"
        latex_lines.append(row)
    
    # Add Total row (average performance of each algorithm across all environments)
    latex_lines.append("\\midrule")
    total_row_values = []
    for algo in algos:
        total_mean = total_stats[total_stats['algo'] == algo]['mean'].values[0]
        total_std = total_stats[total_stats['algo'] == algo]['std'].values[0]
        total_cell = f"${total_mean:.0f} \\pm {total_std:.0f}$"
        total_row_values.append(total_cell)
    
    total_row = "\\textbf{Total} & " + " & ".join(total_row_values) + " \\\\"
    latex_lines.append(total_row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Join all lines
    latex_table = "\n".join(latex_lines)
    
    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results_table.tex"
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\n{'='*80}")
    print(f"LaTeX table saved to: {output_file}")
    print(f"{'='*80}\n")
    print("Table preview:")
    print("-" * 80)
    print(latex_table)
    print("-" * 80)
    print("\nNote: This table uses booktabs package. Add to your LaTeX preamble:")
    print("  \\usepackage{booktabs}")
    print()
    
    # Also print a plain text version for quick viewing
    print("\nPlain text version:")
    print("=" * 100)
    
    # Create plain text table
    col_width = 18
    header_plain = f"{'Environment':<{col_width}}" + "".join([f"{algo:>{col_width}}" for algo in algos])
    print(header_plain)
    print("-" * len(header_plain))
    
    for env in envs:
        env_display_name = env_names.get(env, env)
        row_values_plain = []
        
        for algo in algos:
            mean_val = pivot_mean.loc[env, algo] if algo in pivot_mean.columns and env in pivot_mean.index else None
            std_val = pivot_std.loc[env, algo] if algo in pivot_std.columns and env in pivot_std.index else None
            
            if pd.notna(mean_val):
                if pd.notna(std_val):
                    cell = f"{mean_val:.0f} ± {std_val:.0f}"
                else:
                    cell = f"{mean_val:.0f}"
            else:
                cell = "---"
            
            row_values_plain.append(f"{cell:>{col_width}}")
        
        row_plain = f"{env_display_name:<{col_width}}" + "".join(row_values_plain)
        print(row_plain)
    
    # Add Total row
    print("-" * len(header_plain))
    total_row_values_plain = []
    for algo in algos:
        total_mean = total_stats[total_stats['algo'] == algo]['mean'].values[0]
        total_std = total_stats[total_stats['algo'] == algo]['std'].values[0]
        total_cell = f"{total_mean:.0f} ± {total_std:.0f}"
        total_row_values_plain.append(f"{total_cell:>{col_width}}")
    
    total_row_plain = f"{'Total':<{col_width}}" + "".join(total_row_values_plain)
    print(total_row_plain)
    
    print("=" * 100)


def visualize_paper_quality_mujoco(csv_path: Path,
                                    output_dir: Path,
                                    show: bool=False,
                                    ema_alpha: float=0.4,
                                    max_step: int=2_000_000) -> None:
    """Create publication-quality figure with 4 MuJoCo environments in a single row.
    
    Args:
        csv_path: Path to results CSV file
        output_dir: Directory to save the figure
        show: Whether to display the plot
        ema_alpha: EMA smoothing factor (0-1, higher = more smoothing)
        max_step: Maximum training step to include
    """
    df = load_results(csv_path)
    
    # Filter to only the 4 main MuJoCo environments
    target_envs = [ 'Walker2d-v5', 'Humanoid-v5','Ant-v5', 'HalfCheetah-v5',]
    df = df[df["env"].isin(target_envs)]
    
    if len(df) == 0:
        print("Warning: No data found for target environments")
        return
    
    # Drop unwanted algorithms if any
    drop_algos = ['SACDistanceAgentNew']
    df = df[~df["algo"].isin(drop_algos)]
    
    # Filter by max step
    if max_step is not None:
        df = df[df["step"] <= max_step]
    
    # Apply EMA smoothing
    if 0.0 < ema_alpha < 1.0:
        df = apply_ema_smoothing(df, smoothing_weight=ema_alpha)
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.3)
    sns.set_palette("colorblind")
    
    # Create figure with 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    # Environment display names (cleaner for paper)
    env_names = {
        'Ant-v5': 'Ant',
        'HalfCheetah-v5': 'HalfCheetah',
        'Walker2d-v5': 'Walker2d',
        'Humanoid-v5': 'Humanoid'
    }
    
    #rename algorithms for better display
    df['algo'] = df['algo'].replace({
        'sac': 'SAC',
        'td3': 'TD3',
        'ppo': 'PPO',
        'tqc': 'TQC',
    })

    # Color palette for algorithms (professional colors)
    algo_colors = {
        'DistAgent': '#1f77b4',  # Blue
        'TD3': '#ff7f0e',  # Orange
        'PPO': '#2ca02c',  # Green
        'SAC': '#d62728',  # Red
        'TQC': '#9467bd',  # Purple
    }
    
    legend_handles = None
    legend_labels = None
    
    for idx, env in enumerate(target_envs):
        ax = axes[idx]
        env_df = df[df["env"] == env]
        
        if len(env_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(env_names.get(env, env), fontsize=14, fontweight='bold')
            continue
        
        # Plot each algorithm
        for algo in sorted(env_df['algo'].unique()):
            algo_df = env_df[env_df['algo'] == algo]
            
            # Group by seed and compute mean and std
            grouped = algo_df.groupby('step')['eval_reward'].agg(['mean', 'std', 'count']).reset_index()
            
            steps = grouped['step']
            means = grouped['mean']
            stds = grouped['std']
            
            color = algo_colors.get(algo, None)
            
            # Plot mean line
            line = ax.plot(steps, means, label=algo, linewidth=2.5, alpha=0.9, color=color)
            
            # Plot shaded std region
            if len(grouped) > 0 and not stds.isna().all():
                ax.fill_between(steps, 
                               means - stds, 
                               means + stds, 
                               alpha=0.2, 
                               color=line[0].get_color())
        
        # Styling
        ax.set_title(env_names.get(env, env), fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel('Training Steps', fontsize=13)
        
        if idx == 0:
            ax.set_ylabel('Episode Return', fontsize=13)
        
        # Format x-axis to show steps in millions
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(6,6))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Capture legend
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    
    # Add shared legend at the top
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            ncol=len(legend_labels),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=13,
            bbox_to_anchor=(0.5, 1.02)
        )
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "training_mujoco_paper_quality.png"
    print(f"\n{'='*80}")
    print(f"Saving publication-quality figure to: {output_file}")
    print(f"{'='*80}\n")
    
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Also save as PDF for LaTeX papers
    output_pdf = output_dir / "training_mujoco_paper_quality.pdf"
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"Also saved PDF version to: {output_pdf}\n")
    
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
    parser.add_argument("--ema-alpha", type=float, default=0.7,
                        help="EMA smoothing factor (0-1). Higher=more smoothing. 0=disabled. Recommended: 0.1-0.5")
    parser.add_argument("--max-step", type=int, default=1_000_000,
                        help="Maximum training step to plot (set negative to include all)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Disable printing dataset summary")
    parser.add_argument("--paper-quality", action="store_true",
                        help="Generate publication-quality figure for 4 MuJoCo environments")
    parser.add_argument("--latex-table", action="store_true",
                        help="Generate LaTeX table with max average results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_step = args.max_step if args.max_step >= 0 else None
    
    if args.latex_table:
        # Generate LaTeX table with max average results for ALL environments
        drop_algos = ['SACDistanceAgentNew']
        generate_latex_table(
            args.csv,
            args.out,
            target_envs=None,  # None = all environments
            drop_algos=drop_algos,
            max_step=max_step
        )
    elif args.paper_quality:
        # Generate publication-quality figure for main MuJoCo environments
        visualize_paper_quality_mujoco(
            args.csv,
            args.out,
            show=args.show,
            ema_alpha=args.ema_alpha,
            max_step=max_step
        )
    else:
        # Generate standard multi-environment overview
        visualize(args.csv,
                  args.out,
                  show=args.show,
                  ema_alpha=args.ema_alpha,
                  max_step=max_step,
                  print_summary=not args.no_summary)


if __name__ == "__main__":
    main()
