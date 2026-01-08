"""Plot ablation training curves from ablation_results_full.csv.

Creates per-environment subplots with one curve per ablation variant (algo),
optionally applying EMA smoothing and limiting the max step. Designed to mirror
existing training visualizations while being self-contained.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
from typing import Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns


# ---------------------------- Data utilities -----------------------------

def load_ablation_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"env", "algo", "seed", "step", "eval_reward"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.dropna(subset=["env", "algo", "step", "eval_reward"])


def _load_repk_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"repK results file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"env", "algo", "seed", "step", "eval_reward", "K", "rep_gamma_shape"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in repK results: {sorted(missing)}")
    return df.dropna(subset=["env", "algo", "seed", "step", "eval_reward"])


def apply_ema_smoothing(df: pd.DataFrame, smoothing_weight: float) -> pd.DataFrame:
    """Apply exponential moving average smoothing using pandas ewm."""
    if not (0.0 < smoothing_weight < 1.0):
        return df
    pandas_alpha = 1.0 - smoothing_weight  # higher smoothing_weight = more smoothing
    df = df.sort_values(["env", "algo", "seed", "step"]).copy()
    df["eval_reward"] = df.groupby(["env", "algo", "seed"])["eval_reward"].transform(
        lambda x: x.ewm(alpha=pandas_alpha, adjust=False).mean()
    )
    return df


def _select_best_repk_variant(repk_df: pd.DataFrame, env: str, algo_name: str) -> Optional[pd.DataFrame]:
    env_df = repk_df[(repk_df["env"] == env) & (repk_df["algo"].str.lower() == algo_name.lower())]
    if len(env_df) == 0:
        print(f"No repK rows for env={env} and algo={algo_name}; skipping baseline for this env.")
        return None

    required_cols = {"K", "rep_gamma_shape"}
    if not required_cols.issubset(env_df.columns):
        print(f"repK rows for env={env} missing hyperparam columns; skipping baseline.")
        return None

    run_max = (
        env_df.groupby(["K", "rep_gamma_shape", "seed"])["eval_reward"]
        .max()
        .reset_index(name="max_reward")
    )

    config_stats = (
        run_max.groupby(["K", "rep_gamma_shape"])["max_reward"]
        .mean()
        .reset_index(name="mean_max_reward")
        .sort_values("mean_max_reward", ascending=False)
    )

    best = config_stats.iloc[0]
    mask = (env_df["K"] == best["K"]) & (env_df["rep_gamma_shape"] == best["rep_gamma_shape"])
    best_rows = env_df[mask].copy()
    return best_rows


def add_repk_baseline(
    base_df: pd.DataFrame,
    repk_csv: Optional[Path],
    target_envs: Sequence[str],
    repk_algo_name: str = "v2DistAgent",
    baseline_label: str = "DistAgent",
) -> pd.DataFrame:
    """Append best repK variant per env as a baseline curve."""

    if repk_csv is None:
        return base_df

    repk_df = _load_repk_results(Path(repk_csv))

    # Keep only the target algo
    repk_df = repk_df[repk_df["algo"].str.lower() == repk_algo_name.lower()]
    if len(repk_df) == 0:
        print(f"No rows with algo={repk_algo_name} in repK csv; skipping baseline.")
        return base_df

    collected: list[pd.DataFrame] = []
    for env in target_envs:
        best_rows = _select_best_repk_variant(repk_df, env, repk_algo_name)
        if best_rows is None or len(best_rows) == 0:
            continue
        best_rows = best_rows.copy()
        best_rows["algo"] = baseline_label
        collected.append(best_rows)

    if not collected:
        return base_df

    merged_baseline = pd.concat(collected, ignore_index=True)

    # Drop any existing rows with the same env and baseline label to avoid duplicates
    drop_mask = (base_df["env"].isin(target_envs)) & (base_df["algo"].str.lower() == baseline_label.lower())
    base_df = base_df[~drop_mask]

    return pd.concat([base_df, merged_baseline], ignore_index=True)


def print_ablation_summary(df: pd.DataFrame) -> None:
    """Print max average reward per ablation (algo) per environment."""

    max_rewards = df.groupby(["env", "algo", "seed"])["eval_reward"].max().reset_index(name="max_reward")
    stats = max_rewards.groupby(["env", "algo"])['max_reward'].agg(['mean', 'std', 'count']).reset_index()

    print("\n" + "=" * 70)
    print("ABLATION PERFORMANCE SUMMARY (max over training per seed)")
    print("=" * 70)

    for env in sorted(stats["env"].unique()):
        env_stats = stats[stats["env"] == env].sort_values("mean", ascending=False)
        print(f"\n{env}")
        print("-" * 70)
        print(f"{'Algo':<35s} {'Mean Max':<12s} {'Std':<12s} {'Seeds':<8s}")
        for _, row in env_stats.iterrows():
            mean_val = row['mean']
            std_val = 0.0 if pd.isna(row['std']) else row['std']
            count = int(row['count'])
            std_str = f"± {std_val:6.1f}" if count > 1 else "±   0.0"
            print(f"{row['algo']:<35s} {mean_val:10.1f} {std_str:<12s} {count:<8d}")
    print("\n" + "=" * 70 + "\n")


# ---------------------------- Plotting ----------------------------------

def visualize_ablation(
    csv_path: Path,
    output_dir: Path,
    show: bool = False,
    ema_alpha: float = 0.0,
    max_step: Optional[float] = 1_000_000,
    env_filter: Optional[Sequence[str]] = None,
    repk_csv: Optional[Path] = None,
    repk_algo_name: str = "v2DistAgent",
    repk_baseline_label: str = "DistAgent",
    print_summary: bool = True,
    dpi: int = 250,
) -> None:
    """Plot ablation training curves grouped by environment."""

    df = load_ablation_results(csv_path)

    # Drop unwanted ablation baselines
    drop_algos = {"DistAblationRewardOnly"}
    df = df[~df["algo"].isin(drop_algos)]

    if env_filter:
        df = df[df["env"].isin(env_filter)]

    if max_step is not None:
        df = df[df["step"] <= max_step]

    if repk_csv is not None:
        df = add_repk_baseline(
            base_df=df,
            repk_csv=repk_csv,
            target_envs=env_filter if env_filter else df["env"].unique(),
            repk_algo_name=repk_algo_name,
            baseline_label=repk_baseline_label,
        )

    if len(df) == 0:
        raise ValueError("No data after filtering for ablation visualization")

    if print_summary:
        print_ablation_summary(df)

    if 0.0 < ema_alpha < 1.0:
        df = apply_ema_smoothing(df, smoothing_weight=ema_alpha)

    envs = sorted(df["env"].unique())
    n_env = len(envs)
    ncols = min(3, n_env)
    nrows = math.ceil(n_env / ncols)

    sns.set_theme(style="whitegrid", context="talk", palette="tab10")

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

        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7, prune=None))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(6, 6))
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

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

    title = "\nAblation Training Curves"
    if 0.0 < ema_alpha < 1.0:
        title += f" (EMA α={ema_alpha:.2f})"
    fig.suptitle(title, fontsize="x-large", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "training_ablation.png"
    print(f"Saving ablation plot to {output_file}")
    fig.savefig(output_file, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------- CLI --------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation training curves from CSV results")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results_analysis/data/ablation_results_full.csv"),
        help="Path to ablation_results_full.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_analysis/plots"),
        help="Directory to save plot images",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="EMA smoothing factor (0-1). Higher=more smoothing. 0=disabled.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=1_000_000,
        help="Maximum training step to plot (set negative to include all)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=None,
        help="Comma-separated list of environments to include (default: all)",
    )
    parser.add_argument(
        "--repk-csv",
        type=Path,
        default=Path("results_analysis/data/repK_results_full.csv"),
        help="Path to repK_results_full.csv for baseline extraction",
    )
    parser.add_argument(
        "--no-repk-baseline",
        action="store_true",
        help="Disable adding the best repK baseline curve",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable printing per-environment max average rewards",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="DPI for saved figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_step = args.max_step if args.max_step is None or args.max_step >= 0 else None
    env_filter = None
    if args.envs:
        env_filter = [e.strip() for e in args.envs.split(',') if e.strip()]

    repk_csv = None if args.no_repk_baseline else args.repk_csv

    visualize_ablation(
        csv_path=args.csv,
        output_dir=args.out,
        show=args.show,
        ema_alpha=args.ema_alpha,
        max_step=max_step,
        env_filter=env_filter,
        repk_csv=repk_csv,
        print_summary=not args.no_summary,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
