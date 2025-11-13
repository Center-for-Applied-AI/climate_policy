#!/usr/bin/env python3
"""Summarize role rankings from s31 similarity outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_INPUT_PATH = Path("outputs/s31_role_similarity_all_agencies_all_roles.xlsx")
DEFAULT_OUTPUT_CSV = Path("outputs/s32_role_rank_summary.csv")
DEFAULT_OUTPUT_PLOT = Path("plots/s32_role_rank_summary.png")
DEFAULT_OUTPUT_PLOT_SVG = Path("plots/s32_role_rank_summary.svg")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Summarize role rankings.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the s31 role similarity Excel file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination for the summary CSV.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_OUTPUT_PLOT,
        help="Destination for the summary plot (PNG).",
    )
    parser.add_argument(
        "--output-plot-svg",
        type=Path,
        default=DEFAULT_OUTPUT_PLOT_SVG,
        help="Destination for the summary plot (SVG).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Number of bootstrap iterations to estimate rank confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrap sampling.",
    )
    return parser.parse_args()

def load_rank_matrix(path: Path) -> pd.DataFrame:
    """Load the s31 similarity results and convert them into a rank matrix.

    Parameters
    ----------
    path
        Location of the Excel file produced by s31.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by seed with roles as columns and integer ranks as values.
    """

    df = pd.read_excel(path)
    if "seed" not in df.columns:
        raise ValueError("Input data must contain a 'seed' column identifying simulations.")

    ranked = (
        df[df["seed"].notna()]
        .copy()
        .assign(seed=lambda d: d["seed"].astype(int))
        .sort_values(["seed", "similarity_to_official_final"], ascending=[True, False])
    )

    ranked["rank"] = ranked.groupby("seed").cumcount() + 1

    pivot = ranked.pivot(index="seed", columns="role", values="rank")
    if pivot.isna().any().any():
        missing = pivot.columns[pivot.isna().any()].tolist()
        raise ValueError(f"Missing rank assignments detected for roles: {missing}")
    return pivot.sort_index()

def bootstrap_mean_ranks(rank_matrix: pd.DataFrame, iterations: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the mean ranks across seeds for each role."""

    values = rank_matrix.to_numpy()
    n_seeds, n_roles = values.shape
    if n_seeds < 2:
        raise ValueError("At least two seeds are required for bootstrap resampling.")

    means = np.empty((iterations, n_roles), dtype=float)
    for i in range(iterations):
        sample_idx = rng.integers(0, n_seeds, size=n_seeds)
        sample = values[sample_idx, :]
        means[i, :] = sample.mean(axis=0)
    return means

def summarise_ranks(rank_matrix: pd.DataFrame, bootstrap_means: np.ndarray, roles: np.ndarray) -> pd.DataFrame:
    """Build a summary DataFrame with average ranks and confidence intervals."""

    observed_mean = rank_matrix.mean(axis=0).to_numpy()
    observed_std = rank_matrix.std(axis=0, ddof=1).to_numpy()
    n = rank_matrix.shape[0]

    ci_lower = np.percentile(bootstrap_means, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_means, 97.5, axis=0)

    summary = pd.DataFrame(
        {
            "role": roles,
            "mean_rank": observed_mean,
            "std_rank": observed_std,
            "n": n,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )
    summary["se_rank"] = summary["std_rank"] / np.sqrt(summary["n"])
    return summary.sort_values("mean_rank").reset_index(drop=True)

def plot_rank_summary(summary: pd.DataFrame, output_png: Path, output_svg: Path) -> None:
    """Plot average ranks with 95% confidence intervals and save the figure."""

    roles = summary["role"].to_numpy()
    means = summary["mean_rank"].to_numpy()
    lower = summary["ci_lower"].to_numpy()
    upper = summary["ci_upper"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    y_positions = np.arange(len(roles))
    error_lower = means - lower
    error_upper = upper - means

    ax.errorbar(
        means,
        y_positions,
        xerr=[error_lower, error_upper],
        fmt="o",
        color="tab:blue",
        ecolor="tab:blue",
        capsize=4,
        linewidth=1.5,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(roles)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (1 = highest cosine similarity)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_png_path = output_png.resolve()
    output_svg_path = output_svg.resolve()

    fig.savefig(output_png_path, dpi=300)
    fig.savefig(output_svg_path)
    plt.close(fig)

def main() -> None:
    args = parse_args()

    rank_matrix = load_rank_matrix(args.input)

    rng = np.random.default_rng(args.seed)
    bootstrap_means = bootstrap_mean_ranks(rank_matrix, args.iterations, rng)
    summary = summarise_ranks(rank_matrix, bootstrap_means, rank_matrix.columns.to_numpy())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    plot_rank_summary(summary, args.output_plot, args.output_plot_svg)

if __name__ == "__main__":
    main()
