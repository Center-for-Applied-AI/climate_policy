#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import openai
import pandas as pd
from tqdm.auto import tqdm
import instructor
from pydantic import BaseModel, Field


import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

CLIENT = instructor.patch(openai.OpenAI(api_key=openai.api_key))

COMMENTS_FILE = Path("outputs/comment_climate_engagement_scores.xlsx")
OUTPUT_FILE = Path("outputs/comment_topics.xlsx")
MODEL_NAME = "gpt-4o-2024-08-06"
MAX_WORKERS = 4
SEED = 42
MIN_TOPIC_COUNT = 3


class TopicExtraction(BaseModel):
    """Data model for extracted topics."""

    topics: list[str] = Field(
        ..., description="List of up to three topics, each a phrase of at most three words."
    )


def extract_topics(comment: str, model: str = MODEL_NAME) -> dict:
    """Extract up to three topics from a comment using LLM.
    
    Each topic is a phrase of at most three words with no special characters.
    
    Args:
        comment: The comment text to analyze.
        model: The OpenAI model to use.
        
    Returns:
        Dictionary with topics list.
    """
    prompt = (
        "You will receive a comment on a policy proposal. Extract the top three topics discussed in the comment. "
        "Each topic should be a phrase of at most three words. Do not use hyphens or any special characters in topic names; "
        "use only letters and spaces. Return a JSON list of up to three topics.\n\n[COMMENT]\n" + comment
    )
    
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=TopicExtraction,
        seed=SEED,
    )
    return resp.model_dump()


def process_row(row: pd.Series) -> dict:
    """Extract topics for a single comment row.
    
    Args:
        row: DataFrame row with comment_path, score, and text.
        
    Returns:
        Dictionary with comment_path, score, and topics.
    """
    comment = row["text"]
    result = extract_topics(comment)
    return {"comment_path": row["comment_path"], "score": row["score"], "topics": result["topics"]}


def main() -> None:
    """Extract topics for each comment, then plot and print topic stats."""
    comment_score_df = pd.read_excel(COMMENTS_FILE)
    comment_score_df = comment_score_df[
        comment_score_df["score"].apply(lambda x: isinstance(x, (int, float)))
    ]
    comment_score_df["comment_path"] = comment_score_df["comment_path"].astype(str)
    comment_score_df["text"] = comment_score_df["comment_path"].apply(
        lambda p: Path(p).read_text(encoding="utf-8", errors="ignore")
    )

    records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(comment_score_df), desc="Extracting topics") as pbar:
            futures = [executor.submit(process_row, row) for _, row in comment_score_df.iterrows()]
            for future in as_completed(futures):
                records.append(future.result())
                pbar.update(1)

    comment_topic_df = pd.DataFrame(records)

    comment_topic_exploded_df = comment_topic_df.explode("topics")
    comment_topic_exploded_df = comment_topic_exploded_df[
        comment_topic_exploded_df["topics"].notnull() & (comment_topic_exploded_df["topics"] != "error")
    ]

    comment_topic_exploded_df["topics"] = comment_topic_exploded_df["topics"].str.strip().str.lower()

    topic_stats_df = comment_topic_exploded_df.groupby("topics")["score"].agg(["mean", "count", "std"])
    topic_stats_df = topic_stats_df[topic_stats_df["count"] >= MIN_TOPIC_COUNT]
    topic_stats_df = topic_stats_df.sort_values("mean", ascending=False)
    
    plt.figure(figsize=(10, max(4, len(topic_stats_df) // 2)))
    plt.barh(topic_stats_df.index, topic_stats_df["mean"], xerr=topic_stats_df["std"], color="steelblue")
    plt.xlabel("Average Score")
    plt.title("Average Climate Engagement Score by Topic")
    plt.tight_layout()
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/topic_avg_score.png")
    print("Saved plots/topic_avg_score.png")

    topic_count_series = comment_topic_exploded_df["topics"].value_counts()
    plt.figure(figsize=(10, max(4, len(topic_count_series) // 2)))
    topic_count_series.plot(kind="barh", color="steelblue")
    plt.xlabel("Count")
    plt.title("Topic Counts")
    plt.tight_layout()
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/topic_counts.png")
    print("Saved plots/topic_counts.png")

    print("\nTop topics (count):")
    print(topic_stats_df["count"].sort_values(ascending=False).head(30))

    print("\nTopic counts by score:")
    topic_score_counts_df = comment_topic_exploded_df.groupby(["topics", "score"]).size().unstack(fill_value=0)
    print(topic_score_counts_df.head(30))

    all_topics = sorted(
        set(t for topics in comment_topic_df["topics"] for t in topics if t and t != "error")
    )

    def make_dummies(row: pd.Series) -> pd.Series:
        present = set([t.strip().lower() for t in row["topics"] if t and t != "error"])
        return pd.Series([1 if t in present else 0 for t in all_topics], index=all_topics)

    topic_dummies_df = comment_topic_df.apply(make_dummies, axis=1)
    comment_topic_with_dummies_df = pd.concat(
        [comment_topic_df[["comment_path", "score"]], topic_dummies_df], axis=1
    )

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        comment_topic_df.to_excel(writer, index=False, sheet_name="topics")
        comment_topic_with_dummies_df.to_excel(writer, sheet_name="topics_dummies", index=False)
    print(f"Saved {OUTPUT_FILE} with topics and topics_dummies sheets")


if __name__ == "__main__":
    main()
