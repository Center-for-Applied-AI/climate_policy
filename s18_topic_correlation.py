#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_FILE = Path("outputs/comment_topics.xlsx")
SHEET_NAME = "topics_dummies"
OUTPUT_IMG = Path("plots/topic_correlation_heatmap.png")
EMBED_MODEL = "text-embedding-ada-002"
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

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
    """Load topic dummies, merge similar topics, and plot correlation matrix."""
    row_topic_df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
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

    topic_corr_matrix = X_renamed.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        topic_corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Topic Correlation Matrix (after merging and filtering)")
    plt.tight_layout()
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Saved topic correlation heatmap to {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
