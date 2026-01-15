#!/usr/bin/env python3
"""Summarize hours needed to reach 1M steps per algorithm/environment."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize hours-to-1M-steps by algorithm and environment.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results_analysis/data/main_results_summary.csv"),
        help="Path to main_results_summary.csv",
    )
    return parser


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"algorithm", "env", "seed", "hours_to_1M"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df


def summarize_hours(df: pd.DataFrame) -> pd.DataFrame:
    # Only keep finite values for aggregation.
    valid = df[pd.notna(df["hours_to_1M"])]
    if valid.empty:
        return pd.DataFrame(columns=["algorithm", "env", "hours_to_1M_mean", "runs"])

    grouped = (
        valid.groupby(["algorithm", "env"])
        .agg(hours_to_1M_mean=("hours_to_1M", "mean"), runs=("hours_to_1M", "count"))
        .reset_index()
    )
    grouped["hours_to_1M_mean"] = grouped["hours_to_1M_mean"].round(2)
    return grouped


def main() -> None:
    args = build_parser().parse_args()
    df = load_data(args.csv)
    summary = summarize_hours(df)

    if summary.empty:
        print("No finite hours_to_1M values found in the summary CSV.")
        return

    pivot = (
        summary.pivot(index="env", columns="algorithm", values="hours_to_1M_mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # Round and keep two decimals for readability.
    pivot = pivot.round(2)

    print("Hours to reach 1M steps (mean over seeds):")
    print(pivot.to_markdown(tablefmt="github", floatfmt=".2f"))


if __name__ == "__main__":
    main()
