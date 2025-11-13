#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplot2tikz import save as tikz_save
import matplotlib as mpl


INPUT_FILE = Path("outputs/comment_climate_engagement_scores.xlsx")
PLOTS_DIR = Path("plots")
GREY_COLOR = "#7f7f7f"

OUTPUT_BASE_HIST_OCC = PLOTS_DIR / "score_histogram_OCC"
OUTPUT_BASE_HIST_FDIC = PLOTS_DIR / "score_histogram_FDIC"
OUTPUT_BASE_HIST_FED = PLOTS_DIR / "score_histogram_FED"
OUTPUT_BASE_HIST_ALL = PLOTS_DIR / "score_histogram_ALL"
OUTPUT_BASE_COUNTS = PLOTS_DIR / "score_counts_by_author"
OUTPUT_BASE_MEANS = PLOTS_DIR / "score_means_by_author"

mpl.rcParams["svg.fonttype"] = "none"


def save_all_formats(fig: plt.Figure, base_path: Path) -> None:
    """Save a figure to PNG, SVG, and TiKZ (.tex).
    
    Args:
        fig: The matplotlib figure to save.
        base_path: Base path without extension.
    """
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    tikz_save(str(base_path.with_suffix(".tex")), figure=fig)


def detect_source(path_value: object) -> str:
    """Detect source agency from comment path.
    
    Args:
        path_value: The comment path value.
        
    Returns:
        Agency name (OCC, FDIC, FED, or Unknown).
    """
    text = str(path_value).lower()
    if "comments/occ" in text:
        return "OCC"
    if "comments/fdic" in text:
        return "FDIC"
    if "comments/fed" in text:
        return "FED"
    return "Unknown"


def normalize_author_type(value: object) -> str:
    """Normalize author_type groupings.
    
    Args:
        value: The author type value.
        
    Returns:
        Normalized author type string.
    """
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if text.startswith("professional on behalf of"):
        return "organization"
    if text in {"academic", "private individual"}:
        return "private individuals"
    return text


def normalize_org_type(value: object) -> str:
    """Normalize organization type values.
    
    Args:
        value: The organization type value.
        
    Returns:
        Normalized organization type string.
    """
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if text in {"gov agency", "government agency"}:
        return "government"
    return text


def ci_95_margin(x: pd.Series) -> float:
    """Calculate the 95% confidence interval margin of error.
    
    Args:
        x: Series of values.
        
    Returns:
        Margin of error for 95% CI.
    """
    if len(x) < 2:
        return np.nan

    standard_error = st.sem(x)
    margin = standard_error * st.t.ppf((1 + 0.95) / 2.0, len(x) - 1)
    return margin


def aggregate_scores(dataframe: pd.DataFrame, group_by_col: str) -> pd.DataFrame:
    """Calculate count, mean, and 95% CI for score grouped by a column.
    
    Includes nulls (NaNs) in the grouping column by treating them as 'Unknown'.
    
    Args:
        dataframe: Input dataframe with score column.
        group_by_col: Column name to group by.
        
    Returns:
        Aggregated dataframe with count, mean, and ci_95_margin.
    """
    df_filled = dataframe.copy()
    df_filled[group_by_col] = df_filled[group_by_col].fillna("Unknown")

    agg_df = (
        df_filled.groupby(group_by_col)["score"]
        .agg(count="size", mean="mean", ci_95_margin=ci_95_margin)
        .reset_index()
    )

    return agg_df


def main() -> None:
    """Main function to plot climate engagement score distributions and aggregates."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found at '{INPUT_FILE}'")

    comment_score_df = pd.read_excel(INPUT_FILE)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    comment_score_df["source"] = comment_score_df["comment_path"].apply(detect_source)

    comment_score_valid_df = comment_score_df[
        comment_score_df["score"].apply(lambda x: isinstance(x, (int, float)))
    ]
    comment_score_valid_df["score"] = pd.to_numeric(comment_score_valid_df["score"])

    print("Generating individual histogram plots...")
    occ_scores = comment_score_valid_df.loc[comment_score_valid_df["source"] == "OCC", "score"]
    fdic_scores = comment_score_valid_df.loc[comment_score_valid_df["source"] == "FDIC", "score"]
    fed_scores = comment_score_valid_df.loc[comment_score_valid_df["source"] == "FED", "score"]
    all_scores = comment_score_valid_df["score"]

    bins = np.arange(1, 7) - 0.5

    fig_occ, ax_occ = plt.subplots(figsize=(6, 6))
    ax_occ.hist(occ_scores, bins=bins, color=GREY_COLOR, edgecolor="black")
    ax_occ.set_xlabel("Climate Engagement Score")
    ax_occ.set_ylabel("Count")
    ax_occ.set_xticks(range(1, 6))
    plt.tight_layout()
    save_all_formats(fig_occ, OUTPUT_BASE_HIST_OCC)
    print(
        f"Saved OCC histogram to {OUTPUT_BASE_HIST_OCC.with_suffix('.png')}, "
        f"{OUTPUT_BASE_HIST_OCC.with_suffix('.svg')}, and "
        f"{OUTPUT_BASE_HIST_OCC.with_suffix('.tex')}"
    )

    fig_fdic, ax_fdic = plt.subplots(figsize=(6, 6))
    ax_fdic.hist(fdic_scores, bins=bins, color=GREY_COLOR, edgecolor="black")
    ax_fdic.set_xlabel("Climate Engagement Score")
    ax_fdic.set_ylabel("Count")
    ax_fdic.set_xticks(range(1, 6))
    plt.tight_layout()
    save_all_formats(fig_fdic, OUTPUT_BASE_HIST_FDIC)
    print(
        f"Saved FDIC histogram to {OUTPUT_BASE_HIST_FDIC.with_suffix('.png')}, "
        f"{OUTPUT_BASE_HIST_FDIC.with_suffix('.svg')}, and "
        f"{OUTPUT_BASE_HIST_FDIC.with_suffix('.tex')}"
    )

    if not fed_scores.empty:
        fig_fed, ax_fed = plt.subplots(figsize=(6, 6))
        ax_fed.hist(fed_scores, bins=bins, color=GREY_COLOR, edgecolor="black")
        ax_fed.set_xlabel("Climate Engagement Score")
        ax_fed.set_ylabel("Count")
        ax_fed.set_xticks(range(1, 6))
        plt.tight_layout()
        save_all_formats(fig_fed, OUTPUT_BASE_HIST_FED)
        print(
            f"Saved FED histogram to {OUTPUT_BASE_HIST_FED.with_suffix('.png')}, "
            f"{OUTPUT_BASE_HIST_FED.with_suffix('.svg')}, and "
            f"{OUTPUT_BASE_HIST_FED.with_suffix('.tex')}"
        )

    fig_all, ax_all = plt.subplots(figsize=(6, 6))
    ax_all.hist(all_scores, bins=bins, color=GREY_COLOR, edgecolor="black")
    ax_all.set_xlabel("Climate Engagement Score")
    ax_all.set_ylabel("Count")
    ax_all.set_xticks(range(1, 6))
    plt.tight_layout()
    save_all_formats(fig_all, OUTPUT_BASE_HIST_ALL)
    print(
        f"Saved ALL histogram to {OUTPUT_BASE_HIST_ALL.with_suffix('.png')}, "
        f"{OUTPUT_BASE_HIST_ALL.with_suffix('.svg')}, and "
        f"{OUTPUT_BASE_HIST_ALL.with_suffix('.tex')}"
    )

    print("\nCalculating aggregates (count, average, 95% CI)...")

    comment_score_valid_df["author_type_grouped"] = comment_score_valid_df["author_type"].apply(
        normalize_author_type
    )
    comment_score_valid_df["author_organization_type_grouped"] = comment_score_valid_df[
        "author_organization_type"
    ].apply(normalize_org_type)

    grouping_columns = [
        "author_type_grouped",
        "author_organization_type_grouped",
        "author_state",
    ]
    results = {}

    for col in grouping_columns:
        df_for_aggregation = comment_score_valid_df

        if col == "author_organization_type_grouped":
            exclude_types = ["private individuals"]
            print(f"\nApplying filter for '{col}' analysis: Excluding author types {exclude_types}")

            is_excluded = df_for_aggregation["author_type_grouped"].fillna("").str.lower().isin(exclude_types)
            df_for_aggregation = df_for_aggregation[~is_excluded]

            print(
                f"Original row count: {len(comment_score_valid_df)}. "
                f"Rows for org analysis: {len(df_for_aggregation)}."
            )

        results[col] = aggregate_scores(df_for_aggregation, col)
        print(f"\n--- Aggregates by {col} ---")
        print(results[col])

    print("\nGenerating count plots (separate files)...")
    for col in grouping_columns:
        data = results[col].sort_values("count", ascending=False)
        fig_c, ax_c = plt.subplots(figsize=(12, 5))
        ax_c.bar(data[col], data["count"], color=GREY_COLOR, edgecolor="black")
        ax_c.set_ylabel("Number of Comments")
        plt.setp(ax_c.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax_c.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        safe_name = col.replace("_grouped", "").replace("_", "-")
        base = Path(f"{OUTPUT_BASE_COUNTS}__{safe_name}")
        save_all_formats(fig_c, base)
        print(
            f"Saved counts plot to {base.with_suffix('.png')}, "
            f"{base.with_suffix('.svg')}, and {base.with_suffix('.tex')}"
        )

    print("\nGenerating average score plots (separate files)...")
    for col in grouping_columns:
        data = results[col].sort_values("mean", ascending=False)
        y_err = data["ci_95_margin"].fillna(0)
        fig_m, ax_m = plt.subplots(figsize=(12, 5))
        ax_m.bar(
            data[col],
            data["mean"],
            yerr=y_err,
            capsize=5,
            color=GREY_COLOR,
            edgecolor="black",
            alpha=0.9,
        )
        ax_m.set_ylabel("Average Score")
        if not data.empty:
            max_y = (data["mean"] + y_err).max()
            ax_m.set_ylim(bottom=0, top=max_y * 1.1)
        plt.setp(ax_m.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax_m.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        safe_name = col.replace("_grouped", "").replace("_", "-")
        base = Path(f"{OUTPUT_BASE_MEANS}__{safe_name}")
        save_all_formats(fig_m, base)
        print(
            f"Saved average scores plot to {base.with_suffix('.png')}, "
            f"{base.with_suffix('.svg')}, and {base.with_suffix('.tex')}"
        )


if __name__ == "__main__":
    main()
