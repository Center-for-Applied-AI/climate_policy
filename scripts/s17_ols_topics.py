#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import openai
import numpy as np


INPUT_FILE = Path("outputs/comment_topics.xlsx")
OUTPUT_FILE = Path("outputs/ols_topics_results.xlsx")
SHEET_NAME = "topics_dummies"

import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

EMBED_MODEL = "text-embedding-ada-002"
MIN_TOPIC_COUNT = 2
SIMILARITY_THRESHOLD = 0.9


def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for a text string.
    
    Args:
        text: Input text.
        
    Returns:
        Embedding vector as numpy array.
    """
    resp = openai.embeddings.create(input=[text], model=EMBED_MODEL)
    return np.array(resp.data[0].embedding)


def compute_cosine_similarity(emb_i: np.ndarray, emb_j: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.
    
    Args:
        emb_i: First embedding vector.
        emb_j: Second embedding vector.
        
    Returns:
        Cosine similarity score.
    """
    return float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))


def main() -> None:
    """Load topic dummies, run OLS, and save results."""
    row_topic_df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    y = row_topic_df["score"]
    X = row_topic_df.drop(columns=["comment_path", "score"])

    topic_counts = X.sum(axis=0)
    X = X.loc[:, topic_counts > MIN_TOPIC_COUNT]
    topic_cols = [col for col in X.columns if col != "const"]

    topic_embeddings = {t: get_embedding(t) for t in topic_cols}

    pairs = []
    for idx_i, i in enumerate(topic_cols):
        emb_i = topic_embeddings[i]
        for idx_j, j in enumerate(topic_cols):
            if idx_j <= idx_i:
                continue
            emb_j = topic_embeddings[j]
            sim = compute_cosine_similarity(emb_i, emb_j)
            pairs.append({"topic_1": i, "topic_2": j, "cosine": sim})
    
    topic_similarity_df = pd.DataFrame(pairs)
    topic_similarity_df = topic_similarity_df.sort_values("cosine", ascending=False)

    rename_map = {t: t for t in topic_cols}
    used = set()
    for _, row in topic_similarity_df.iterrows():
        t1, t2, sim = row["topic_1"], row["topic_2"], row["cosine"]
        if sim <= SIMILARITY_THRESHOLD:
            break

        if t2 not in used and t1 != t2:
            for k, v in rename_map.items():
                if v == t2:
                    rename_map[k] = t1
            used.add(t2)

    X_renamed = X.rename(columns=rename_map)

    X_renamed = X_renamed.groupby(level=0, axis=1).sum().clip(upper=1)
    X_renamed = sm.add_constant(X_renamed)

    topic_renaming_df = pd.DataFrame(list(rename_map.items()), columns=["original_topic", "renamed_topic"])

    model = sm.OLS(y, X_renamed).fit()
    ols_summary_df = model.summary2().tables[1].reset_index()
    ols_summary_df.rename(columns={"index": "variable"}, inplace=True)
    means = X_renamed.mean(axis=0)
    ols_summary_df["mean_dummy"] = ols_summary_df["variable"].map(means)

    exog_matrix = X_renamed.values
    exog_columns = list(X_renamed.columns)
    vif_records = []
    for idx, col_name in enumerate(exog_columns):
        if col_name == "const":
            continue
        vif_value = float(variance_inflation_factor(exog_matrix, idx))
        vif_records.append({"variable": col_name, "vif": vif_value})
    
    vif_df = pd.DataFrame(vif_records).sort_values("vif", ascending=False)
    
    r2_df = pd.DataFrame({"r2": [model.rsquared]})
    
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        ols_summary_df.to_excel(writer, index=False, sheet_name="ols_summary")
        topic_similarity_df.to_excel(writer, index=False, sheet_name="cosine_pairs")
        topic_renaming_df.to_excel(writer, index=False, sheet_name="topic_renaming")
        r2_df.to_excel(writer, index=False, sheet_name="r2")
        vif_df.to_excel(writer, index=False, sheet_name="vif")
    
    print(f"Saved OLS results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
