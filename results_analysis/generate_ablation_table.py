"""Generate LaTeX table for ablation study results.

Reads ablation_results_full.csv and produces a formatted LaTeX table
with mean ± standard error for specified environments at 500k steps.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def load_ablation_results(csv_path: Path) -> pd.DataFrame:
    """Load and validate ablation results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"env", "algo", "seed", "step", "eval_reward"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.dropna(subset=["env", "algo", "step", "eval_reward"])


def get_results_at_step(df: pd.DataFrame, target_step: int, tolerance: int = 5000) -> pd.DataFrame:
    """Extract results at or near target step for each env/algo/seed combination."""
    results = []
    
    for (env, algo, seed), group in df.groupby(['env', 'algo', 'seed']):
        # Find closest step to target
        group = group.copy()
        group['step_diff'] = abs(group['step'] - target_step)
        closest_idx = group['step_diff'].idxmin()
        closest_row = group.loc[closest_idx]
        
        # Only include if within tolerance
        if closest_row['step_diff'] <= tolerance:
            results.append({
                'env': env,
                'algo': algo,
                'seed': seed,
                'step': closest_row['step'],
                'eval_reward': closest_row['eval_reward']
            })
    
    return pd.DataFrame(results)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard error for each env/algo combination."""
    stats = df.groupby(['env', 'algo'])['eval_reward'].agg(['mean', 'std', 'count']).reset_index()
    # Standard error = std / sqrt(n)
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    return stats


def format_result(mean: float, se: float) -> str:
    """Format result as mean ± se with appropriate precision."""
    # Determine appropriate decimal places based on magnitude
    if abs(mean) >= 1000:
        return f"${mean:.0f} \\pm {se:.0f}$"
    elif abs(mean) >= 100:
        return f"${mean:.1f} \\pm {se:.1f}$"
    else:
        return f"${mean:.2f} \\pm {se:.2f}$"


def generate_latex_table(
    csv_path: Path,
    target_step: int = 500_000,
    environments: list[str] = None,
    output_path: Path = None,
    repk_csv: Path = None,
) -> str:
    """Generate LaTeX table for ablation results."""
    
    # Load data
    df = load_ablation_results(csv_path)
    
    # Load repK baseline if provided
    if repk_csv and repk_csv.exists():
        repk_df = pd.read_csv(repk_csv)
        # Add best v2DistAgent from repK as "DistAgent"
        if 'v2DistAgent' in repk_df['algo'].values:
            repk_baseline = repk_df[repk_df['algo'] == 'v2DistAgent'].copy()
            repk_baseline['algo'] = 'DistAgent'
            df = pd.concat([df, repk_baseline], ignore_index=True)
    
    # Filter environments if specified
    if environments:
        df = df[df['env'].isin(environments)]
    else:
        environments = sorted(df['env'].unique())
    
    # Get results at target step
    results_df = get_results_at_step(df, target_step)
    
    if len(results_df) == 0:
        raise ValueError(f"No results found at step {target_step}")
    
    # Compute statistics
    stats = compute_statistics(results_df)
    
    # Define ablation mapping (algo name -> display name)
    ablation_mapping = {
        'DistAgent': 'DistRL (full)',
        'v2DistAgent': 'DistRL (full)',
        'DistAblationA1': 'w/o repr. loss (A1)',
        'DistAblationRewardOnlyN64': '64-step reward w/o Q',
        'DistAblationA3': 'w/o adaptive beta scaling (A3)',
        'DistAblationA4': 'w/o adaptive beta scaling (A4)',
        'DistAblationB1': 'uniform kernel weighting (B1)',
        'DistAblationB9NoAdaptiveTau': 'w/o adaptive $\\tau$',
        'DistAblationRewardOnly': 'reward-only baseline',
    }
    
    # Ordered list of ablations to include in table
    ablation_order = [
        'v2DistAgent',  # Use v2DistAgent (repK baseline) as the main baseline
        'DistAblationA1',
        'DistAblationRewardOnlyN64',
        'DistAblationA4',
        'DistAblationB1',
        'DistAblationB9NoAdaptiveTau',
    ]
    
    # Build LaTeX table
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Design choice ablation study on two MuJoCo environments.")
    lines.append("Each row removes or modifies one architectural component of the proposed method.")
    lines.append("Results are reported as mean evaluation return $\\pm$ standard error over multiple seeds")
    lines.append(f"after {target_step//1000}k environment steps.}}")
    lines.append("\\label{tab:ablation}")
    
    # Table header
    env_cols = ' & '.join([f"\\textbf{{{env}}}" for env in environments])
    lines.append(f"\\begin{{tabular}}{{l{'c' * len(environments)}}}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Variant}} & {env_cols} \\\\")
    lines.append("\\midrule")
    
    # Table rows
    for algo in ablation_order:
        if algo not in stats['algo'].values:
            continue
        
        variant_name = ablation_mapping.get(algo, algo)
        row_data = [variant_name]
        
        for env in environments:
            env_algo_stats = stats[(stats['env'] == env) & (stats['algo'] == algo)]
            if len(env_algo_stats) == 0:
                row_data.append("--")
            else:
                mean = env_algo_stats.iloc[0]['mean']
                se = env_algo_stats.iloc[0]['se']
                row_data.append(format_result(mean, se))
        
        lines.append(' & '.join(row_data) + " \\\\")
        lines.append("")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    latex_table = '\n'.join(lines)
    
    # Save to file if output path specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_table)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX ablation table")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results_analysis/data/ablation_results_full.csv"),
        help="Path to ablation_results_full.csv",
    )
    parser.add_argument(
        "--repk-csv",
        type=Path,
        default=Path("results_analysis/data/repK_results_full.csv"),
        help="Path to repK_results_full.csv for baseline",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=500_000,
        help="Target step for evaluation (default: 500000)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default="HalfCheetah-v5,Humanoid-v5",
        help="Comma-separated list of environments (default: HalfCheetah-v5,Humanoid-v5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_analysis/data/ablation_table.tex"),
        help="Output path for LaTeX table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    environments = [e.strip() for e in args.envs.split(',') if e.strip()]
    
    latex_table = generate_latex_table(
        csv_path=args.csv,
        target_step=args.step,
        environments=environments,
        output_path=args.output,
        repk_csv=args.repk_csv,
    )
    
    print("\n" + "=" * 80)
    print("GENERATED LATEX TABLE")
    print("=" * 80)
    print(latex_table)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
