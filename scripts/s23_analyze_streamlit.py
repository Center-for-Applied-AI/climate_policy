#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplot2tikz import save as tikz_save


RANDOM_SEED = 42

INPUT_SURVEY_PATH = Path("data/streamlit_climate_responses.xlsx")
INPUT_MODEL_SCORES_PATH = Path("outputs/comment_climate_engagement_scores.xlsx")
PLOTS_DIR = Path("plots")

GREY_COLOR = "#7f7f7f"
HEATMAP_CMAP = "Blues"

OUTPUT_BASE_MEAN_MODEL_BY_HUMAN = PLOTS_DIR / "prolific_validation_survey_mean_model_by_human"
OUTPUT_BASE_MEAN_HUMAN_BY_MODEL = PLOTS_DIR / "prolific_validation_survey_mean_human_by_model"
OUTPUT_BASE_JOINT_DISTRIBUTION = PLOTS_DIR / "prolific_validation_survey_joint_distribution"
OUTPUT_BASE_TIME_BY_QUESTION = PLOTS_DIR / "prolific_validation_survey_time_by_question"
OUTPUT_BASE_CUM_TIME_BY_QUESTION = PLOTS_DIR / "prolific_validation_survey_cumulative_time_by_question"


def save_all_formats(fig: plt.Figure, base_path: Path) -> None:
    """Save a figure to PNG, SVG, and TiKZ (.tex).
    
    Args:
        fig: The matplotlib figure to save.
        base_path: Base path without extension.
    """
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    tikz_save(str(base_path.with_suffix(".tex")), figure=fig)


def build_attention_pass_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with attention-check pass flags merged by prolific_id.
    
    The logic: count the number of attention checks passed (score equals 2 on the attention-check rows).
    A participant is considered to have passed attention checks if they passed at least one of the two.
    
    Args:
        df: Input survey dataframe.
        
    Returns:
        DataFrame with attention check flags added.
    """
    df_ac = df[df["comment"].astype(str).str.contains("attention_check", case=False, na=False)].copy()
    df_ac["attention_check_pass_single"] = df_ac["human_score"].astype(str).astype(float) == 2
    passed_counts = df_ac.groupby("prolific_id")["attention_check_pass_single"].sum().reset_index()
    passed_counts["attention_check_passed"] = passed_counts["attention_check_pass_single"].apply(
        lambda x: True if x >= 1 else False
    )
    passed_counts.rename(columns={"attention_check_pass_single": "attention_checks_passed"}, inplace=True)
    out = df.merge(
        passed_counts[["prolific_id", "attention_check_passed", "attention_checks_passed"]],
        on="prolific_id",
        how="left",
    )
    return out


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load survey and model-score dataframes and return filtered frames.
    
    Returns:
        Tuple of (row_level_survey_df, comment_aggregate_df).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    _ = rng

    if not INPUT_SURVEY_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_SURVEY_PATH}")
    if not INPUT_MODEL_SCORES_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_MODEL_SCORES_PATH}")

    response_df = pd.read_excel(INPUT_SURVEY_PATH)
    response_df.columns = ["prolific_id", "comment", "human_score", "timestamp"]
    response_df = response_df.drop_duplicates()

    response_df = build_attention_pass_flags(response_df)

    response_df = response_df[
        ~response_df["comment"].astype(str).str.contains("attention_check", case=False, na=False)
    ]
    response_df = response_df[~response_df["comment"].astype(str).str.contains("LLM_USAGE", case=False, na=False)]

    response_df = response_df[response_df["attention_check_passed"] == True]

    comment_score_df = pd.read_excel(INPUT_MODEL_SCORES_PATH)
    comment_score_df["comment_path"] = comment_score_df["comment_path"].astype(str).apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )
    comment_score_df = comment_score_df[["comment_path", "score"]]

    response_df = pd.merge(
        response_df,
        comment_score_df,
        left_on="comment",
        right_on="comment_path",
        how="left",
    )

    response_df = response_df[response_df["score"].astype(str) != "na"]
    response_df["human_score"] = response_df["human_score"].astype(int)
    response_df["score"] = pd.to_numeric(response_df["score"], errors="coerce")
    response_df = response_df.dropna(subset=["score"])

    comment_aggregate_df = (
        response_df.groupby("comment")
        .agg(
            score=("score", "mean"),
            human_score=(
                "human_score",
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            ),
        )
        .reset_index()
        .dropna(subset=["human_score"])
    )

    return response_df, comment_aggregate_df


def plot_mean_model_by_human(comment_aggregate_df: pd.DataFrame) -> None:
    """Bar plot: mean model score by majority-vote human score with 95% CI.
    
    Args:
        comment_aggregate_df: DataFrame with comment-level aggregates.
    """
    grouped = comment_aggregate_df.groupby("human_score")["score"]
    means = grouped.mean()
    counts = grouped.count()
    stds = grouped.std(ddof=1)
    cis = 1.96 * (stds / np.sqrt(counts))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(means.index, means.values, yerr=cis.values, capsize=5, color=GREY_COLOR, edgecolor="black", alpha=0.9)
    ax.set_xlabel("Human Score")
    ax.set_ylabel("Mean Model Score")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_all_formats(fig, OUTPUT_BASE_MEAN_MODEL_BY_HUMAN)


def plot_mean_human_by_model(comment_aggregate_df: pd.DataFrame) -> None:
    """Bar plot: mean human score by model score with 95% CI.
    
    Args:
        comment_aggregate_df: DataFrame with comment-level aggregates.
    """
    grouped = comment_aggregate_df.groupby("score")["human_score"]
    means = grouped.mean()
    counts = grouped.count()
    stds = grouped.std(ddof=1)
    cis = 1.96 * (stds / np.sqrt(counts))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(means.index, means.values, yerr=cis.values, capsize=5, color=GREY_COLOR, edgecolor="black", alpha=0.9)
    ax.set_xlabel("Model Score")
    ax.set_ylabel("Mean Human Score")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_all_formats(fig, OUTPUT_BASE_MEAN_HUMAN_BY_MODEL)


def plot_joint_distribution(response_df: pd.DataFrame) -> None:
    """Heatmap of Model vs. Human score distribution (percent of total).
    
    Args:
        response_df: Row-level survey dataframe.
    """
    score_range = [1, 2, 3, 4, 5]
    counts = pd.crosstab(response_df["score"], response_df["human_score"]).reindex(
        index=score_range, columns=score_range, fill_value=0
    )
    percentages = counts / counts.values.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        percentages,
        annot=True,
        fmt=".1f",
        cmap=HEATMAP_CMAP,
        cbar_kws={"label": "% of Total Responses"},
        ax=ax,
    )
    ax.set_xlabel("Human Score")
    ax.set_ylabel("Model Score")
    ax.invert_yaxis()
    plt.tight_layout()
    save_all_formats(fig, OUTPUT_BASE_JOINT_DISTRIBUTION)


def compute_time_deltas(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute per-question time deltas in seconds for attention-check passers.
    
    Args:
        df_raw: Raw survey dataframe.
        
    Returns:
        DataFrame with columns: prolific_id, question_number, delta_sec.
    """
    d = df_raw.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values(["prolific_id", "timestamp"]).reset_index(drop=True)
    d["question_number"] = d.groupby("prolific_id").cumcount() + 1
    d["delta_sec"] = d.groupby("prolific_id")["timestamp"].diff().dt.total_seconds()
    d = d.dropna(subset=["delta_sec"]).reset_index(drop=True)
    return d[["prolific_id", "question_number", "delta_sec"]]


def plot_time_by_question(df_raw: pd.DataFrame) -> None:
    """Plot mean, median, and empirical 95% bands of completion time by question.
    
    Args:
        df_raw: Raw survey dataframe.
    """
    time_df = compute_time_deltas(df_raw)
    agg = (
        time_df.groupby("question_number")["delta_sec"]
        .agg(
            mean="mean",
            median="median",
            q025=lambda x: np.quantile(x, 0.025),
            q975=lambda x: np.quantile(x, 0.975),
        )
        .reset_index()
        .sort_values("question_number")
    )

    if (agg["question_number"] == 16).any():
        agg = agg[agg["question_number"] != 16]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(agg["question_number"], agg["q025"], agg["q975"], color="#d9d9d9", alpha=0.7, label="95% band")
    ax.plot(agg["question_number"], agg["mean"], color=GREY_COLOR, linewidth=2, label="Mean")
    ax.plot(agg["question_number"], agg["median"], color="#4d4d4d", linewidth=2, linestyle="--", label="Median")

    ax.set_xlabel("Question number")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    save_all_formats(fig, OUTPUT_BASE_TIME_BY_QUESTION)


def plot_cumulative_time_by_question(df_raw: pd.DataFrame) -> None:
    """Plot cumulative time (sum of deltas) by question with mean/median/95% band.
    
    Args:
        df_raw: Raw survey dataframe.
    """
    time_df = compute_time_deltas(df_raw)

    time_df = time_df.sort_values(["prolific_id", "question_number"]).reset_index(drop=True)

    def _cum(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        first_idx = g.index.min()
        g.loc[first_idx, "delta_sec"] = (
            g.loc[first_idx, "delta_sec"] if not np.isnan(g.loc[first_idx, "delta_sec"]) else 0.0
        )
        g["cum_sec"] = g["delta_sec"].fillna(0).cumsum()
        return g

    time_df = time_df.groupby("prolific_id", group_keys=False).apply(_cum)

    agg = (
        time_df.groupby("question_number")["cum_sec"]
        .agg(
            mean="mean",
            median="median",
            q025=lambda x: np.quantile(x, 0.025),
            q975=lambda x: np.quantile(x, 0.975),
        )
        .reset_index()
        .sort_values("question_number")
    )
    if (agg["question_number"] == 16).any():
        agg = agg[agg["question_number"] != 16]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(agg["question_number"], agg["q025"], agg["q975"], color="#d9d9d9", alpha=0.7, label="95% band")
    ax.plot(agg["question_number"], agg["mean"], color=GREY_COLOR, linewidth=2, label="Mean")
    ax.plot(agg["question_number"], agg["median"], color="#4d4d4d", linewidth=2, linestyle="--", label="Median")

    ax.set_xlabel("Question number")
    ax.set_ylabel("Cumulative seconds")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    save_all_formats(fig, OUTPUT_BASE_CUM_TIME_BY_QUESTION)


def main() -> None:
    """Run analysis and generate all plots."""
    mpl.rcParams["svg.fonttype"] = "none"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    response_df, comment_aggregate_df = load_and_prepare()

    plot_mean_model_by_human(comment_aggregate_df)
    plot_mean_human_by_model(comment_aggregate_df)
    plot_joint_distribution(response_df)

    df_time_source = pd.read_excel(INPUT_SURVEY_PATH)
    df_time_source.columns = ["prolific_id", "comment", "human_score", "timestamp"]
    df_time_source = df_time_source.drop_duplicates()
    df_time_source = build_attention_pass_flags(df_time_source)
    df_time_source = df_time_source[df_time_source["attention_check_passed"] == True]
    plot_time_by_question(df_time_source)
    plot_cumulative_time_by_question(df_time_source)


if __name__ == "__main__":
    main()
