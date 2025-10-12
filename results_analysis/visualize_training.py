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


def apply_ema_smoothing(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if not (0.0 < alpha < 1.0):
        return df

    def _smooth(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("step").copy()
        group["eval_reward"] = group["eval_reward"].ewm(alpha=alpha, adjust=False).mean()
        return group

    return df.groupby(["env", "algo", "seed"], group_keys=False).apply(_smooth)


def visualize(csv_path: Path,
              output_dir: Path,
              show: bool=False,
              ema_alpha: float=0.0,
              max_step: int=1_000_000) -> None:
    df = load_results(csv_path)
    if max_step is not None:
        df = df[df["step"] <= max_step]
    if 0.0 < ema_alpha < 1.0:
        df = apply_ema_smoothing(df, ema_alpha)
    envs = sorted(df["env"].unique())
    if not envs:
        raise ValueError("No environments found in results file")

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
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--ema-alpha", type=float, default=0.95,
                        help="Optional EMA smoothing parameter (0 disables)")
    parser.add_argument("--max-step", type=int, default=1_000_000,
                        help="Maximum training step to plot (set negative to include all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_step = args.max_step if args.max_step >= 0 else None
    visualize(args.csv,
              args.out,
              show=args.show,
              ema_alpha=args.ema_alpha,
              max_step=max_step)


if __name__ == "__main__":
    main()
