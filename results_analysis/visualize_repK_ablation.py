"""Generate paper-quality box plots for the repK ablation study."""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Create box plots of the max evaluation reward for each env and repK ablation.",
        add_help=True,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results_analysis/data/repK_results_full.csv"),
        help="Path to the full repK results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_analysis/plots/repK_ablation"),
        help="Directory where the generated figures will be saved.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Subset of environments to plot (default: all).",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="Maximum training step to include when computing max rewards.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI used when saving figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after creating it.",
    )
    return parser


def load_repk_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_columns = {"env", "seed", "step", "eval_reward", "K", "rep_gamma_shape"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in repK results: {sorted(missing)}")
    return df


def compute_max_rewards(df: pd.DataFrame, max_step: Optional[float] = None) -> pd.DataFrame:
    if max_step is not None:
        df = df[df["step"] <= max_step]
    grouped = (
        df.groupby(["env", "seed", "K", "rep_gamma_shape"])["eval_reward"]
        .max()
        .reset_index()
        .rename(columns={"eval_reward": "max_reward"})
    )
    return grouped


def summarize_results(max_rewards: pd.DataFrame) -> None:
    summary = (
        max_rewards.groupby("env")
        .agg(
            seeds=("seed", "nunique"),
            runs=("max_reward", "count"),
            Ks=("K", "nunique"),
            rep_gammas=("rep_gamma_shape", "nunique"),
        )
        .sort_index()
    )
    print("\nENVIRONMENT SUMMARY")
    print(summary.to_string())


def _format_env_label(env: str) -> str:
    """Format environment name without version and in bold."""
    if "-v" in env:
        env = env.rsplit("-v", 1)[0]
    return env


def _apply_box_colors(ax: plt.Axes, colors: List) -> None:
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.9)


def filter_for_k_ablation(max_rewards: pd.DataFrame, target_gamma: float = 1.5) -> pd.DataFrame:
    return max_rewards[max_rewards["rep_gamma_shape"] == target_gamma]


def filter_for_gamma_ablation(max_rewards: pd.DataFrame, target_k: int = 256) -> pd.DataFrame:
    return max_rewards[max_rewards["K"] == target_k]


def _plot_single_ablation(
    data: pd.DataFrame,
    envs: List[str],
    x_col: str,
    order: List,
    colors: List,
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    dpi: int,
    show: bool,
) -> None:
    if not envs:
        raise ValueError("No environments selected for plotting.")

    output_dir.mkdir(parents=True, exist_ok=True)

    width_per_panel = 3.2
    fig_width = max(3.8, width_per_panel * len(envs))
    fig_height = 2.8

    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        rc={
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": dpi,
        },
    )
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(envs),
        figsize=(fig_width, fig_height),
        sharey=False,
        squeeze=False,
    )

    for col_idx, env in enumerate(envs):
        ax = axes[0, col_idx]
        env_df = data[data["env"] == env]
        if env_df.empty:
            ax.set_visible(False)
            continue

        env_df = env_df.copy()
        order_labels = [str(o) for o in order]
        env_df["_x_cat"] = env_df[x_col].astype(str)

        palette_map = dict(zip(order_labels, colors))
        sns.boxplot(
            data=env_df,
            x="_x_cat",
            y="max_reward",
            order=order_labels,
            hue="_x_cat",
            hue_order=order_labels,
            dodge=False,
            ax=ax,
            palette=palette_map,
            width=0.65,
            fliersize=2.5,
            flierprops={"marker": "x", "markersize": 4, "alpha": 0.6},
            boxprops={"edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(_format_env_label(env), fontweight="bold")
        ax.set_xlabel(xlabel)
        if col_idx == 0:
            ax.set_ylabel(ylabel, labelpad=6)
        else:
            ax.set_ylabel("")
            
        # Rotate tick labels to save horizontal space
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Scientific notation if rewards are large
        ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))

        # Custom y ticks: majors every 1000, minors every 500 (aligned to 500)
        ymin, ymax = env_df["max_reward"].min(), env_df["max_reward"].max()
        if np.isfinite(ymin) and np.isfinite(ymax):
            minor_step = 500.0
            major_step = 1000.0
            start = np.floor(ymin / minor_step) * minor_step
            end = np.ceil(ymax / minor_step) * minor_step

            major_ticks = np.arange(
                np.floor(start / major_step) * major_step,
                np.ceil(end / major_step) * major_step + major_step * 0.1,
                major_step,
            )
            minor_ticks = np.arange(start, end + minor_step * 0.1, minor_step)

            if len(major_ticks) >= 2:
                ax.set_yticks(major_ticks)
            if len(minor_ticks) >= 2:
                ax.set_yticks(minor_ticks, minor=True)

        # Grids for both major and minor ticks
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

        # Lighten spines for a clean paper look
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figures to {png_path} and {pdf_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_ablation_boxplots(
    max_rewards: pd.DataFrame,
    envs: List[str],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    # K ablation: fix rep_gamma_shape to the design value (observed 1.5 in the dataset)
    k_data = filter_for_k_ablation(max_rewards)
    k_order = sorted(k_data["K"].dropna().unique())
    k_colors = sns.color_palette("tab10", len(k_order))

    _plot_single_ablation(
        data=k_data,
        envs=envs,
        x_col="K",
        order=k_order,
        colors=k_colors,
        xlabel="K",
        ylabel="Max Reward",
        output_dir=output_dir,
        stem="repK_ablation_K_only",
        dpi=dpi,
        show=show,
    )

    # rep_gamma ablation: fix K to the design value (observed 256 in the dataset)
    gamma_data = filter_for_gamma_ablation(max_rewards)
    rep_order = sorted(gamma_data["rep_gamma_shape"].dropna().unique())
    rep_labels = [f"{v:.2f}" for v in rep_order]
    label_map = dict(zip(rep_order, rep_labels))
    gamma_data = gamma_data.copy()
    gamma_data["rep_gamma_label"] = gamma_data["rep_gamma_shape"].map(label_map)
    rep_colors = sns.color_palette("tab10", len(rep_order))

    _plot_single_ablation(
        data=gamma_data,
        envs=envs,
        x_col="rep_gamma_label",
        order=rep_labels,
        colors=rep_colors,
        xlabel="Gamma",
        ylabel="Max Reward",
        output_dir=output_dir,
        stem="repK_ablation_gamma_only",
        dpi=dpi,
        show=show,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = load_repk_results(args.csv)
    max_rewards = compute_max_rewards(df, max_step=args.max_step)

    if max_rewards.empty:
        raise ValueError("No max reward data available after filtering.")

    env_candidates = sorted(max_rewards["env"].unique())
    envs = args.envs or env_candidates
    missing = set(envs) - set(env_candidates)
    if missing:
        raise ValueError(f"Requested environments not found: {sorted(missing)}")

    summarize_results(max_rewards)
    plot_ablation_boxplots(
        max_rewards=max_rewards,
        envs=envs,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()