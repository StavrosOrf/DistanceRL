"""Generate line+errorbar plots for the repK ablation study.

This variant keeps a numeric x-axis (no categorical binning), draws a
continuous line with standard-deviation error bars, and overlays the
individual run outcomes ("outliers") so distribution shape is visible.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Create line plots of max evaluation reward for each env and repK ablation. "
            "Lines are drawn over numeric x-axes with error bars and individual runs shown."
        ),
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
        "--target-gamma",
        type=float,
        default=1.5,
        help="rep_gamma_shape value to hold fixed when plotting K sweep.",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=256,
        help="K value to hold fixed when plotting gamma sweep.",
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


def filter_for_k_ablation(max_rewards: pd.DataFrame, target_gamma: float) -> pd.DataFrame:
    return max_rewards[max_rewards["rep_gamma_shape"] == target_gamma]


def filter_for_gamma_ablation(max_rewards: pd.DataFrame, target_k: int) -> pd.DataFrame:
    return max_rewards[max_rewards["K"] == target_k]


def _format_env_label(env: str) -> str:
    # Drop version suffix (e.g., "-v5") for cleaner titles.
    if "-v" in env:
        env = env.rsplit("-v", 1)[0]
    return env


def _line_plot_with_errorbars(
    data: pd.DataFrame,
    envs: List[str],
    x_col: str,
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    dpi: int,
    show: bool,
    jitter: float = 0.00001,
) -> None:
    def _format_tick(value: float, _: int) -> str:
        # Remove trailing zeros and drop leading zero for |x| < 1 (e.g., .25, .5).
        if np.isclose(value, 0.0):
            return "0"
        text = f"{value:g}"
        if text.startswith("-0."):
            text = "-" + text[2:]
        elif text.startswith("0."):
            text = text[1:]
        return text

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
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 13,
            "figure.dpi": dpi,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        },
    )


    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(envs),
        figsize=(fig_width, fig_height),
        sharey=False,
        squeeze=False,
    )

    rng = np.random.default_rng(0)
    # Professional, high-contrast, colorblind-safe palette (seaborn "deep")
    palette = sns.color_palette("deep", len(envs))

    for col_idx, env in enumerate(envs):
        ax = axes[0, col_idx]
        env_df = data[data["env"] == env]
        if env_df.empty:
            ax.set_visible(False)
            continue

        # Aggregate by x value
        agg = (
            env_df.groupby(x_col)["max_reward"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
            .sort_values(x_col)
        )

        x_vals = agg[x_col].to_numpy(dtype=float)
        means = agg["mean"].to_numpy()
        stds = agg["std"].fillna(0.0).to_numpy()

        color = palette[col_idx % len(palette)]
        ax.errorbar(
            x_vals,
            means,
            yerr=stds,
            fmt="-o",
            color=color,
            ecolor=color,
            elinewidth=1,
            capsize=3,
            markersize=4.5,
            linewidth=1.4,
            alpha=0.95,
        )

        # Scatter individual runs to visualize dispersion/outliers.
        scatter_x = env_df[x_col].to_numpy(dtype=float)
        scatter_y = env_df["max_reward"].to_numpy()
        if jitter > 0:
            scatter_x = scatter_x + rng.normal(loc=0.0, scale=jitter, size=len(scatter_x))

        ax.scatter(
            scatter_x,
            scatter_y,
            s=14,
            color="gray",
            alpha=0.45,
            edgecolors="none",
        )

        ax.set_title(_format_env_label(env), fontweight="bold")
        ax.set_xlabel(xlabel)
        if col_idx == 0:
            ax.set_ylabel(ylabel, labelpad=6)
        else:
            ax.set_ylabel("")

        # Keep numeric x-axis ticks at the actual tested values.
        ax.set_xticks(np.sort(env_df[x_col].unique()))

        # y-axis formatting similar to v1 for readability
        ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_tick))
        # ax.yaxis.set_major_formatter(FuncFormatter(_format_tick))

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

        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.9, alpha=0.4)
        ax.grid(axis="y", which="minor", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.3)

        ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=6,
            width=1,
            colors="black",
            labelcolor="black",
            left=True,
            right=False,
            zorder=3,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            direction="out",
            length=3,
            width=0.75,
            colors="black",
            labelcolor="black",
            left=True,
            right=False,
            zorder=3,
        )
        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            length=3,
            width=1,
            colors="black",
            labelcolor="black",
            top=False,
            bottom=True,
        )
        ax.tick_params(
            axis="x",
            which="minor",
            direction="out",
            length=4,
            width=1.0,
            colors="black",
            labelcolor="black",
            top=False,
            bottom=True,
        )
        ax.set_facecolor("white")

    fig.tight_layout()
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figures to {png_path} and {pdf_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_ablation_lines(
    max_rewards: pd.DataFrame,
    envs: List[str],
    output_dir: Path,
    dpi: int,
    show: bool,
    target_gamma: float,
    target_k: int,
) -> None:
    # K sweep with gamma fixed
    k_data = filter_for_k_ablation(max_rewards, target_gamma=target_gamma)
    _line_plot_with_errorbars(
        data=k_data,
        envs=envs,
        x_col="K",
        xlabel="K",
        ylabel="Max Episode Return",
        output_dir=output_dir,
        stem=f"repK_ablation_K_only_v2_gamma_{target_gamma}",
        dpi=dpi,
        show=show,
    )

    # gamma sweep with K fixed
    gamma_data = filter_for_gamma_ablation(max_rewards, target_k=target_k)
    _line_plot_with_errorbars(
        data=gamma_data,
        envs=envs,
        x_col="rep_gamma_shape",
        xlabel="$\lambda$",
        ylabel="Max Episode Return",
        output_dir=output_dir,
        stem=f"repK_ablation_gamma_only_v2_K_{target_k}",
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

    plot_ablation_lines(
        max_rewards=max_rewards,
        envs=envs,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=args.show,
        target_gamma=args.target_gamma,
        target_k=args.target_k,
    )


if __name__ == "__main__":
    main()
