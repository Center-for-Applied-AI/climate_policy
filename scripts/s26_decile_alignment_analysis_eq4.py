#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplot2tikz import save as tikz_save
import matplotlib as mpl
import statsmodels.formula.api as smf


DATA_FILE = Path("outputs/before_after_comment_similarity_embeddings.xlsx")
SCORES_FILE = Path("outputs/comment_climate_engagement_scores.xlsx")
PLOTS_DIR = Path("plots")
OUTPUT_BASE = PLOTS_DIR / "eq4_deciles_avg_score"
OUTPUT_BASE_CLUSTERED = PLOTS_DIR / "eq4_deciles_avg_score_clustered"
OUTPUT_BASE_BY_SCORE = PLOTS_DIR / "eq4_mean_delta_vs_score"

GREY_COLOR = "#7f7f7f"
SEED = 42


def save_all_formats(fig: plt.Figure, base_path: Path) -> None:
    """Save a figure to PNG, SVG, and TiKZ (.tex).
    
    Args:
        fig: The matplotlib figure to save.
        base_path: Base path without extension.
    """
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    tikz_save(str(base_path.with_suffix(".tex")), figure=fig)


def infer_agency(path_value: object) -> str:
    """Infer agency from path string.
    
    Args:
        path_value: Path string or object.
        
    Returns:
        Agency name (occ, fed, or fdic).
    """
    text = str(path_value).lower()
    if "comments/occ" in text:
        return "occ"
    if "comments/fed" in text:
        return "fed"
    return "fdic"


def load_similarity_and_scores() -> pd.DataFrame:
    """Load similarity table, merge scores by comment_path with fallbacks.
    
    Returns:
        A DataFrame with columns: sim_after, sim_before, delta, score, comment_idx, paragraph_num.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {DATA_FILE}")
    if not SCORES_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {SCORES_FILE}")

    paragraph_comment_similarity_df = pd.read_excel(DATA_FILE)
    comment_score_df = pd.read_excel(SCORES_FILE)

    paragraph_comment_similarity_df["comment_file"] = paragraph_comment_similarity_df["comment_path"].astype(
        str
    ).str.extract(r"([^/]+\.txt)$", expand=False)
    paragraph_comment_similarity_df["comment_agency"] = paragraph_comment_similarity_df["comment_path"].map(
        infer_agency
    )

    if "comment_file" not in comment_score_df.columns:
        comment_score_df["comment_file"] = comment_score_df["comment_path"].astype(str).str.extract(
            r"([^/]+\.txt)$", expand=False
        )
    if "comment_agency" not in comment_score_df.columns:
        comment_score_df["comment_agency"] = comment_score_df["comment_path"].map(infer_agency)

    merged = paragraph_comment_similarity_df.merge(
        comment_score_df[["comment_path", "score"]], on="comment_path", how="left"
    )

    miss = merged["score"].isna()
    if miss.any():
        left_fb = paragraph_comment_similarity_df.loc[miss, ["comment_file", "comment_agency"]].copy()
        left_fb["__idx__"] = left_fb.index
        right_fb = comment_score_df[["comment_file", "comment_agency", "score"]].drop_duplicates()
        fb = left_fb.merge(right_fb, on=["comment_file", "comment_agency"], how="left")
        merged.loc[fb["__idx__"], "score"] = fb["score"].values

    miss2 = merged["score"].isna()
    if miss2.any():
        left_fb2 = paragraph_comment_similarity_df.loc[miss2, ["comment_file"]].copy()
        left_fb2["__idx__"] = left_fb2.index
        right_fb2 = comment_score_df[["comment_file", "score"]].drop_duplicates()
        fb2 = left_fb2.merge(right_fb2, on=["comment_file"], how="left")
        merged.loc[fb2["__idx__"], "score"] = fb2["score"].values

    merged = merged.dropna(subset=["sim_after", "sim_before", "score"]).copy()
    merged["delta"] = merged["sim_after"] - merged["sim_before"]
    return merged


def compute_deciles_and_means(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute deciles of delta and average score per decile with 95% CI.
    
    Args:
        df: Input dataframe with delta and score columns.
        
    Returns:
        A tuple of (binned_df, summary_df).
    """
    non_na = df.dropna(subset=["delta", "score"]).copy()
    if non_na.empty:
        return non_na, non_na

    bins = pd.qcut(
        non_na["delta"], q=min(10, max(1, non_na["delta"].nunique())), labels=False, duplicates="drop"
    )

    non_na["decile"] = bins.astype(int) + 1
    order = sorted(non_na["decile"].unique())
    grouped = non_na.groupby("decile", sort=True)["score"].agg(["mean", "std", "count"]).reindex(order)
    se = grouped["std"] / np.sqrt(grouped["count"].replace(0, np.nan))
    margin = 1.96 * se
    summary = grouped.reset_index().rename(columns={"mean": "mean_score", "std": "std_score", "count": "n"})
    summary["ci_95_margin"] = margin.values
    return non_na, summary


def plot_deciles(summary: pd.DataFrame) -> plt.Figure:
    """Create the decile bar plot with 95% CI error bars and return the figure.
    
    Args:
        summary: Summary dataframe with decile statistics.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary["decile"].astype(int).tolist()
    y = summary["mean_score"].astype(float).tolist()
    y_err = summary["ci_95_margin"].fillna(0).astype(float).tolist()
    ax.bar(x, y, yerr=y_err, capsize=5, color=GREY_COLOR, edgecolor="black", alpha=0.9)
    ax.set_xlabel(r"Deciles of $\Delta_{alignment}$")
    ax.set_ylabel("Average climate engagement score")
    ax.set_xticks(x)
    if len(y) > 0:
        upper = np.array(y) + np.array(y_err)
        ymax = float(np.nanmax(upper)) if np.isfinite(upper).any() else max(y)
        ax.set_ylim(bottom=0, top=ymax * 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def compute_two_way_cluster_ci(binned: pd.DataFrame) -> pd.DataFrame:
    """Compute two-way cluster-robust 95% CI for the mean score per decile.
    
    Args:
        binned: Binned dataframe with decile, score, comment_idx, and paragraph_num columns.
        
    Returns:
        DataFrame with decile and ci_95_margin_cluster columns.
    """
    records = []
    deciles = sorted(binned["decile"].unique())
    for d in deciles:
        sub = binned.loc[binned["decile"] == d]
        if sub.empty:
            records.append({"decile": int(d), "ci_95_margin_cluster": np.nan})
            continue
        
        fit = smf.ols("score ~ 1", data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": [sub["comment_idx"], sub["paragraph_num"]]},
        )
        se_val = float(fit.bse.get("Intercept", np.nan))
        margin = 1.96 * se_val if np.isfinite(se_val) else np.nan
        
        records.append({"decile": int(d), "ci_95_margin_cluster": margin})
    return pd.DataFrame.from_records(records)


def plot_deciles_with_err(summary: pd.DataFrame, yerr_col: str) -> plt.Figure:
    """Create decile bar plot with provided error column as yerr.
    
    Args:
        summary: Summary dataframe.
        yerr_col: Column name for error bars.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary["decile"].astype(int).tolist()
    y = summary["mean_score"].astype(float).tolist()
    y_err = summary[yerr_col].fillna(0).astype(float).tolist()
    ax.bar(x, y, yerr=y_err, capsize=5, color=GREY_COLOR, edgecolor="black", alpha=0.9)
    ax.set_xlabel(r"Deciles of $\Delta_{alignment}$")
    ax.set_ylabel("Average climate engagement score")
    ax.set_xticks(x)
    if len(y) > 0:
        upper = np.array(y) + np.array(y_err)
        ymax = float(np.nanmax(upper)) if np.isfinite(upper).any() else max(y)
        ax.set_ylim(bottom=0, top=ymax * 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def compute_score_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean Δ_alignment by raw climate engagement score categories (1..5).
    
    Args:
        df: Input dataframe with score and delta columns.
        
    Returns:
        DataFrame with score, mean_delta, std_delta, n, ci_95_margin columns.
    """
    non_na = df.dropna(subset=["score", "delta"]).copy()
    if non_na.empty:
        return non_na

    non_na["score"] = non_na["score"].astype(float).round().astype(int)
    non_na = non_na.loc[non_na["score"].between(1, 5)].copy()

    grouped = non_na.groupby("score", sort=True)["delta"].agg(["mean", "std", "count"])
    order = [1, 2, 3, 4, 5]
    grouped = grouped.reindex(order)
    se = grouped["std"] / np.sqrt(grouped["count"].replace(0, np.nan))
    margin = 1.96 * se
    summary = grouped.reset_index().rename(columns={"mean": "mean_delta", "std": "std_delta", "count": "n"})
    summary["ci_95_margin"] = margin.values
    return summary


def plot_bar_score_vs_delta(summary: pd.DataFrame) -> plt.Figure:
    """Bar plot: X — climate engagement score (1..5); Y — mean Δ_alignment with 95% CI.
    
    Args:
        summary: Summary dataframe.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = summary.dropna(subset=["n"]).copy()
    valid = valid.loc[valid["n"].fillna(0) > 0]
    x = valid["score"].astype(int).tolist()
    y = valid["mean_delta"].astype(float).tolist()
    y_err = valid["ci_95_margin"].fillna(0).astype(float).tolist()
    ax.bar(x, y, yerr=y_err, capsize=5, color=GREY_COLOR, edgecolor="black", alpha=0.9)
    ax.set_xlabel("Climate Engagement Score")
    ax.set_ylabel(r"Mean $\Delta_{alignment}$")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.5, 5.5)
    if len(y) > 0:
        upper = np.array(y) + np.array(y_err)
        y_min = float(np.nanmin(np.array(y) - np.array(y_err)))
        y_max = float(np.nanmax(upper)) if np.isfinite(upper).any() else max(y)
        ax.set_ylim(bottom=y_min * 1.1 if y_min < 0 else 0, top=y_max * 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def main() -> None:
    """Plot EQ4 decile relationships between similarity deltas and scores."""
    np.random.seed(SEED)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["svg.fonttype"] = "none"

    paragraph_comment_delta_df = load_similarity_and_scores()
    binned, summary = compute_deciles_and_means(paragraph_comment_delta_df)
    if summary.empty:
        print("No data available to plot after filtering for valid deltas and scores.")
        return

    fig = plot_deciles(summary)
    save_all_formats(fig, OUTPUT_BASE)
    print(
        f"Saved decile plot to {OUTPUT_BASE.with_suffix('.png')}, "
        f"{OUTPUT_BASE.with_suffix('.svg')}, and {OUTPUT_BASE.with_suffix('.tex')}"
    )

    clustered = compute_two_way_cluster_ci(binned)
    if not clustered.empty:
        merged = summary.merge(clustered, on=["decile"], how="left")
        fig_c = plot_deciles_with_err(merged, "ci_95_margin_cluster")
        save_all_formats(fig_c, OUTPUT_BASE_CLUSTERED)
        print(
            f"Saved clustered-CI plot to {OUTPUT_BASE_CLUSTERED.with_suffix('.png')}, "
            f"{OUTPUT_BASE_CLUSTERED.with_suffix('.svg')}, and {OUTPUT_BASE_CLUSTERED.with_suffix('.tex')}"
        )

    first_dec = int(min(binned["decile"]))
    last_dec = int(max(binned["decile"]))
    sub = binned.loc[binned["decile"].isin([first_dec, last_dec])].copy()
    sub["last"] = (sub["decile"] == last_dec).astype(int)

    fit_naive = smf.ols("score ~ last", data=sub).fit()
    diff = float(fit_naive.params.get("last", np.nan))
    se = float(fit_naive.bse.get("last", np.nan))
    tstat = float(fit_naive.tvalues.get("last", np.nan))
    pval = float(fit_naive.pvalues.get("last", np.nan))
    ci_low = diff - 1.96 * se if np.isfinite(se) else np.nan
    ci_high = diff + 1.96 * se if np.isfinite(se) else np.nan
    n1 = int((sub["decile"] == first_dec).sum())
    n10 = int((sub["decile"] == last_dec).sum())

    print(
        "\nNaive t-test (i.i.d.) — difference in means (D10 − D1):\n"
        f"  diff={diff:.4f}, se={se:.4f}, t={tstat:.2f}, p={pval:.4g}, CI95=[{ci_low:.4f}, {ci_high:.4f}]\n"
        f"  n_D1={n1}, n_D10={n10}"
    )

    fit_cl = smf.ols("score ~ last", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": [sub["comment_idx"], sub["paragraph_num"]]}
    )
    diff_c = float(fit_cl.params.get("last", np.nan))
    se_c = float(fit_cl.bse.get("last", np.nan))
    tstat_c = float(fit_cl.tvalues.get("last", np.nan))
    pval_c = float(fit_cl.pvalues.get("last", np.nan))
    ci_low_c = diff_c - 1.96 * se_c if np.isfinite(se_c) else np.nan
    ci_high_c = diff_c + 1.96 * se_c if np.isfinite(se_c) else np.nan

    print(
        "Two-way clustered t-test — difference in means (D10 − D1):\n"
        f"  diff={diff_c:.4f}, se={se_c:.4f}, t={tstat_c:.2f}, p={pval_c:.4g}, CI95=[{ci_low_c:.4f}, {ci_high_c:.4f}]\n"
        f"  clusters: comments={sub['comment_idx'].nunique()}, paragraphs={sub['paragraph_num'].nunique()}"
    )

    score_summary = compute_score_category_summary(paragraph_comment_delta_df)
    if not score_summary.empty:
        fig_b = plot_bar_score_vs_delta(score_summary)
        save_all_formats(fig_b, OUTPUT_BASE_BY_SCORE)
        print(
            f"Saved bar CI plot to {OUTPUT_BASE_BY_SCORE.with_suffix('.png')}, "
            f"{OUTPUT_BASE_BY_SCORE.with_suffix('.svg')}, and {OUTPUT_BASE_BY_SCORE.with_suffix('.tex')}"
        )
    else:
        print("No data available to summarize by climate engagement score.")


if __name__ == "__main__":
    main()
