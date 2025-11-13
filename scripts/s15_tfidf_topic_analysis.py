#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm


DATA_DIRS = [
    Path("data/comments/occ/txt"),
    Path("data/comments/fdic/txt"),
]
SCORES_FILE = Path("outputs/comment_climate_engagement_scores.xlsx")
OUTPUT_XLSX = Path("outputs/topic_analysis_results.xlsx")
PLOTS_DIR = Path("plots")
TOP_K = 25
SEED = 42
N_JOBS = 1

URL_PATTERN = re.compile(r"\b\w*(www|http|https)\w*\b", flags=re.IGNORECASE)
DIGIT_ONLY_PATTERN = re.compile(r"\b\d+\b")
ALPHANUMERIC_PATTERN = re.compile(r"\b\w*\d+\w*\b")
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b"


def custom_preprocessor(text: str) -> str:
    """Remove URLs and numbers from text.
    
    Args:
        text: Input text.
        
    Returns:
        Cleaned text.
    """
    text = URL_PATTERN.sub(" ", text)
    text = DIGIT_ONLY_PATTERN.sub(" ", text)
    text = ALPHANUMERIC_PATTERN.sub(" ", text)
    return text


def load_comment_metadata_df(scores_file: Path) -> pd.DataFrame:
    """Load the engagement scores workbook.
    
    Args:
        scores_file: Path to the Excel workbook produced by s9.
        
    Returns:
        DataFrame with at least comment_path, score and source columns.
    """
    comment_score_df = pd.read_excel(scores_file)
    comment_score_df = comment_score_df[
        comment_score_df["score"].apply(lambda x: isinstance(x, (int, float)))
    ]
    comment_score_df["comment_path"] = comment_score_df["comment_path"].astype(str)
    return comment_score_df


def read_comment(path_str: str) -> str:
    """Read a comment file robustly.
    
    Args:
        path_str: String path to a text file.
        
    Returns:
        File contents as UTF-8 text.
    """
    return Path(path_str).read_text(encoding="utf-8", errors="ignore")


def build_comment_text_df(comment_metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Attach raw text to the scores DataFrame.
    
    Args:
        comment_metadata_df: DataFrame containing comment paths and metadata.
        
    Returns:
        DataFrame with an added text column.
    """
    texts: list[str] = []
    for path in tqdm(comment_metadata_df["comment_path"], desc="Reading comments"):
        texts.append(read_comment(path))
    
    comment_text_df = comment_metadata_df.copy()
    comment_text_df["text"] = texts
    comment_text_df["agency"] = np.where(
        comment_text_df["comment_path"].str.contains(str(DATA_DIRS[0])), "OCC", "FDIC"
    )
    return comment_text_df


def tfidf_matrix(corpus: list[str], ngram_range: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Compute a TF-IDF matrix for a corpus, excluding stopwords, numbers, and URLs.
    
    Args:
        corpus: List of documents.
        ngram_range: Inclusive n-gram length range for the vectorizer.
        
    Returns:
        Tuple of (dense matrix, feature names).
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        preprocessor=custom_preprocessor,
        token_pattern=TOKEN_PATTERN,
    )
    matrix = vectorizer.fit_transform(corpus)
    return matrix.toarray(), vectorizer.get_feature_names_out().tolist()


def top_tokens_for_groups(
    matrix: np.ndarray,
    feature_names: list[str],
    groups: list[str],
    group_labels: np.ndarray,
    top_k: int = TOP_K,
) -> dict[str, pd.DataFrame]:
    """Compute the top tokens for each group based on mean TF-IDF.
    
    Args:
        matrix: Document × token TF-IDF array.
        feature_names: Token names corresponding to columns of matrix.
        groups: Unique group labels.
        group_labels: Array mapping each document to its group.
        top_k: Number of top tokens to return.
        
    Returns:
        Mapping from group label to sorted DataFrame of tokens and scores.
    """
    results: dict[str, pd.DataFrame] = {}
    feature_arr = np.array(feature_names)
    for grp in groups:
        grp_mask = group_labels == grp
        if not grp_mask.any():
            continue
        grp_mean = matrix[grp_mask].mean(axis=0)
        order = np.argsort(-grp_mean)
        tokens = feature_arr[order][: top_k * 10]
        scores = grp_mean[order][: top_k * 10]
        token_tfidf_df = pd.DataFrame({"token": tokens, "mean_tfidf": scores})
        results[str(grp)] = token_tfidf_df
    return results


def overall_top_tokens(matrix: np.ndarray, feature_names: list[str], top_k: int = 100) -> pd.DataFrame:
    """Compute the overall most popular tokens by mean TF-IDF.
    
    Args:
        matrix: Document × token TF-IDF array.
        feature_names: Token names.
        top_k: Number of top tokens to return.
        
    Returns:
        DataFrame with tokens and mean TF-IDF scores.
    """
    mean_scores = matrix.mean(axis=0)
    order = np.argsort(-mean_scores)
    tokens = np.array(feature_names)[order][: top_k * 10]
    scores = mean_scores[order][: top_k * 10]
    return pd.DataFrame({"token": tokens, "mean_tfidf": scores})


def write_to_excel(writer: pd.ExcelWriter, data: dict[str, pd.DataFrame], sheet_prefix: str) -> None:
    """Write grouped DataFrames to an Excel writer.
    
    Args:
        writer: Open pd.ExcelWriter instance.
        data: Mapping from group label to DataFrame.
        sheet_prefix: Prefix for sheet names.
    """
    for label, df in data.items():
        name = f"{sheet_prefix}_{label}"[:31]
        df.to_excel(writer, index=False, sheet_name=name)


def bar_plot(grouped_data: dict[str, pd.DataFrame], top_k: int, title: str, file_path: Path) -> None:
    """Create a bar plot for the top tokens per group.
    
    Args:
        grouped_data: Mapping from group label to DataFrame of tokens.
        top_k: Number of tokens to plot.
        title: Plot title.
        file_path: Output PNG path.
    """
    nonempty_data = {k: v for k, v in grouped_data.items() if not v.empty}
    if not nonempty_data:
        return
    n_groups = len(nonempty_data)
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, 4 * n_groups), constrained_layout=True)
    if n_groups == 1:
        axes = [axes]
    for ax, (label, df) in zip(axes, nonempty_data.items()):
        display_df = df.head(top_k)
        ax.barh(display_df["token"], display_df["mean_tfidf"], color="steelblue")
        ax.set_title(str(label))
        ax.invert_yaxis()
    fig.suptitle(title, fontsize=14)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the full topic analysis pipeline."""
    comment_metadata_df = load_comment_metadata_df(SCORES_FILE)
    comment_text_df = build_comment_text_df(comment_metadata_df)

    unigrams_mat, unigrams_features = tfidf_matrix(comment_text_df["text"].tolist(), (1, 1))
    bigrams_mat, bigrams_features = tfidf_matrix(comment_text_df["text"].tolist(), (2, 2))

    score_labels = comment_text_df["score"].astype(str).to_numpy()
    scores = sorted(comment_text_df["score"].astype(str).unique())

    agency_labels = comment_text_df["agency"].to_numpy()
    agencies = sorted(comment_text_df["agency"].unique())

    unigram_by_score = top_tokens_for_groups(unigrams_mat, unigrams_features, scores, score_labels)
    bigram_by_score = top_tokens_for_groups(bigrams_mat, bigrams_features, scores, score_labels)

    unigram_by_agency = top_tokens_for_groups(unigrams_mat, unigrams_features, agencies, agency_labels)
    bigram_by_agency = top_tokens_for_groups(bigrams_mat, bigrams_features, agencies, agency_labels)

    overall_unigrams = overall_top_tokens(unigrams_mat, unigrams_features)
    overall_bigrams = overall_top_tokens(bigrams_mat, bigrams_features)

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        write_to_excel(writer, unigram_by_score, "score_unigrams")
        write_to_excel(writer, bigram_by_score, "score_bigrams")
        write_to_excel(writer, unigram_by_agency, "agency_unigrams")
        write_to_excel(writer, bigram_by_agency, "agency_bigrams")
        overall_unigrams.to_excel(writer, sheet_name="overall_unigrams", index=False)
        overall_bigrams.to_excel(writer, sheet_name="overall_bigrams", index=False)

    bar_plot(unigram_by_score, TOP_K, "Top words by engagement score", PLOTS_DIR / "top_words_by_score.png")
    bar_plot(bigram_by_score, TOP_K, "Top bigrams by engagement score", PLOTS_DIR / "top_bigrams_by_score.png")
    bar_plot(unigram_by_agency, TOP_K, "Top words by agency", PLOTS_DIR / "top_words_by_agency.png")
    bar_plot(bigram_by_agency, TOP_K, "Top bigrams by agency", PLOTS_DIR / "top_bigrams_by_agency.png")

    plot_configs = [
        ("overall_unigrams.png", overall_unigrams, "Overall top words"),
        ("overall_bigrams.png", overall_bigrams, "Overall top bigrams"),
    ]
    
    for name, df, title in plot_configs:
        fig, ax = plt.subplots(figsize=(8, 6))
        display_df = df.head(TOP_K)
        ax.barh(display_df["token"], display_df["mean_tfidf"], color="steelblue")
        ax.set_title(title)
        ax.invert_yaxis()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / name, dpi=300)
        plt.close(fig)

    print(f"Saved topic analysis results to {OUTPUT_XLSX}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
