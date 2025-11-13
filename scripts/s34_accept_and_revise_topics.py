#!/usr/bin/env python3
"""Analyze role-specific topics from accept-and-revise simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openai
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from sklearn.feature_extraction.text import TfidfVectorizer

SEED: int = 42
TEMPERATURE: float = 1.0
MODEL_NAME: str = "gpt-5"
EMB_MODEL: str = "text-embedding-3-large"
TOPIC_MERGE_THRESHOLD: float = 0.9
MAX_WORKERS: int = 10

S33_TRACE_PATH: Path = Path("outputs/s33_role_trace_all_agencies_all_roles.xlsx")
S33_SIM_PATH: Path = Path("outputs/s33_role_similarity_all_agencies_all_roles.xlsx")

TOPIC_CACHE_PATH: Path = Path("outputs/s34_topic_cache.json")
MERGED_TOPICS_PATH: Path = Path("outputs/s34_merged_topics.json")
TFIDF_MATRIX_PATH: Path = Path("outputs/s34_tfidf_matrix.csv")

OUTPUT_TFIDF_SIM_SUMMARY_CSV: Path = Path("outputs/s34_tfidf_similarity_to_final_by_role_mc.csv")
OUTPUT_TFIDF_SIM_SUMMARY_PNG: Path = Path("plots/s34_tfidf_similarity_to_final_with_ci.png")
OUTPUT_TFIDF_SIM_SUMMARY_SVG: Path = Path("plots/s34_tfidf_similarity_to_final_with_ci.svg")
OUTPUT_TFIDF_STACKED_PNG: Path = Path("plots/s34_tfidf_similarity_stacked_official_final.png")
OUTPUT_TFIDF_STACKED_SVG: Path = Path("plots/s34_tfidf_similarity_stacked_official_final.svg")
OUTPUT_TFIDF_TTEST_MATRIX_CSV: Path = Path("outputs/s34_tfidf_role_ttest_matrix.csv")
OUTPUT_TFIDF_TTEST_HEATMAP_PNG: Path = Path("plots/s34_tfidf_role_ttest_heatmap.png")
OUTPUT_TFIDF_TTEST_HEATMAP_SVG: Path = Path("plots/s34_tfidf_role_ttest_heatmap.svg")
OUTPUT_PARA_SIM_CSV: Path = Path("outputs/s34_paragraph_similarity_to_final_by_role_mc.csv")
OUTPUT_TFIDF_SIM_CSV: Path = Path("outputs/s34_tfidf_similarity_to_final_by_role_mc.csv")
OUTPUT_TF_SIM_CSV: Path = Path("outputs/s34_tf_similarity_to_final_by_role_mc.csv")
OUTPUT_VOCAB_SIM_CSV: Path = Path("outputs/s34_vocab_similarity_to_final_by_role_mc.csv")
OUTPUT_JACCARD_SIM_CSV: Path = Path("outputs/s34_jaccard_similarity_to_final_by_role_mc.csv")
OUTPUT_JACCARD_PNG: Path = Path("plots/s34_jaccard_similarity_to_final_with_ci.png")
OUTPUT_JACCARD_SVG: Path = Path("plots/s34_jaccard_similarity_to_final_with_ci.svg")
OUTPUT_TF_PNG: Path = Path("plots/s34_tf_similarity_to_final_with_ci.png")
OUTPUT_TF_SVG: Path = Path("plots/s34_tf_similarity_to_final_with_ci.svg")
OUTPUT_TOPIC_FREQUENCY_CSV: Path = Path("outputs/s34_topic_frequency.csv")
OUTPUT_TOPIC_FREQUENCY_PNG: Path = Path("plots/s34_topic_frequency.png")
OUTPUT_TOPIC_FREQUENCY_SVG: Path = Path("plots/s34_topic_frequency.svg")
OUTPUT_ROLE_TOPICS_PNG: Path = Path("plots/s34_distinctive_topics_by_role.png")
OUTPUT_ROLE_TOPICS_SVG: Path = Path("plots/s34_distinctive_topics_by_role.svg")
OUTPUT_ROLE_TOPICS_CSV: Path = Path("outputs/s34_distinctive_topics_by_role.csv")

ORIGINAL_FINAL_PATH: Path = Path("data/policies/final.txt")
AGENCY_TO_DRAFT: Dict[str, Path] = {
    "fed": Path("data/policies/drafts/fed.txt"),
    "occ": Path("data/policies/drafts/occ.txt"),
    "fdic": Path("data/policies/drafts/fdic.txt"),
}

ROLES: List[str] = [
    "monetary",
    "banking",
    "bureaucrat",
    "nonpartisan",
    "american",
    "democratic",
    "republican",
    "wealth",
    "worldwide",
    "openai",
]

class TopicExtraction(BaseModel):
    """Structured output for topic extraction."""

    topics: List[str] = Field(
        ...,
        description="List of topics discussed in the text. Each topic should be a phrase of at most three words."
    )

def configure_openai_client() -> OpenAI:
    """Configure OpenAI client using environment variable."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    client = OpenAI(api_key=openai.api_key)
    return client

def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file and return its contents stripped."""

    return path.read_text(encoding="utf-8", errors="ignore").strip()

_WORD_RE = re.compile(r"\w+")

def word_count(text: str) -> int:
    """Compute the number of word tokens in a string."""

    return len(_WORD_RE.findall(text))

def extract_topics_from_text(client: OpenAI, text: str, doc_id: str) -> List[str]:
    """Extract topics from a text using GPT-5 with structured output."""

    prompt = (
        "You will receive a policy document. Extract the main topics discussed in the document. "
        "Each topic should be a phrase of at most three words. Do not use hyphens or any special "
        "characters in topic names; use only letters and spaces. Return a list of topics.\n\n"
        "[DOCUMENT]\n" + text
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        seed=SEED,
        response_format={"type": "json_schema", "json_schema": {
            "name": "topic_extraction",
            "schema": TopicExtraction.model_json_schema()
        }}
    )
    parsed = TopicExtraction.model_validate_json(response.choices[0].message.content)
    return parsed.topics

def extract_topics_parallel(doc_items: List[Tuple[str, str]], client: OpenAI) -> Dict[str, List[str]]:
    """Extract topics from multiple documents in parallel."""

    def process_doc(item: Tuple[str, str]) -> Tuple[str, List[str]]:
        doc_id, text = item
        topics = extract_topics_from_text(client, text, doc_id)
        return doc_id, topics

    doc_topics: Dict[str, List[str]] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(doc_items), desc="Extracting topics") as pbar:
            futures = [executor.submit(process_doc, item) for item in doc_items]
            for future in as_completed(futures):
                doc_id, topics = future.result()
                doc_topics[doc_id] = topics
                pbar.update(1)

    return doc_topics

def embed_text(text: str) -> Optional[List[float]]:
    """Compute embedding for a text using OpenAI API."""

    resp = openai.embeddings.create(model=EMB_MODEL, input=[text])
    return resp.data[0].embedding

def cosine_similarity_vectors(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> Optional[float]:
    """Compute cosine similarity between two vectors."""

    if vec_a is None or vec_b is None:
        return None
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float((a / na) @ (b / nb))

def merge_similar_topics(topics: List[str], threshold: float = TOPIC_MERGE_THRESHOLD) -> Dict[str, str]:
    """Merge topics with embedding similarity > threshold using multiple greedy passes.

    Performs greedy merging repeatedly until no more merges occur.

    Returns a mapping from original topic to canonical topic.
    """

    unique_topics = list(set(topics))
    if len(unique_topics) <= 1:
        return {t: t for t in unique_topics}

    print(f"Computing embeddings for {len(unique_topics)} unique topics in parallel...")
    topic_embeddings: Dict[str, Optional[List[float]]] = {}

    def embed_topic(topic: str) -> Tuple[str, Optional[List[float]]]:
        return topic, embed_text(topic)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(unique_topics), desc="Embedding topics") as pbar:
            futures = [executor.submit(embed_topic, topic) for topic in unique_topics]
            for future in as_completed(futures):
                topic, embedding = future.result()
                topic_embeddings[topic] = embedding
                pbar.update(1)

    topic_to_canonical: Dict[str, str] = {}
    canonical_topics: List[str] = []

    print(f"Merging topics with similarity > {threshold} (multiple greedy passes)...")

    pass_num = 0
    while True:
        pass_num += 1
        merges_this_pass = 0

        old_mapping = topic_to_canonical.copy()
        topic_to_canonical = {}
        canonical_topics = []

        for topic in unique_topics:

            vec_topic = topic_embeddings.get(topic)
            if vec_topic is None:
                topic_to_canonical[topic] = topic
                if topic not in canonical_topics:
                    canonical_topics.append(topic)
                continue

            merged = False
            for canonical in canonical_topics:
                vec_canonical = topic_embeddings.get(canonical)
                sim = cosine_similarity_vectors(vec_topic, vec_canonical)
                if sim is not None and sim > threshold:
                    topic_to_canonical[topic] = canonical
                    merged = True

                    if old_mapping.get(topic) != canonical:
                        merges_this_pass += 1
                    break

            if not merged:
                topic_to_canonical[topic] = topic
                canonical_topics.append(topic)

        print(f"  Pass {pass_num}: {len(canonical_topics)} canonical topics, {merges_this_pass} new merges")

        if merges_this_pass == 0:
            break

    print(f"Converged after {pass_num} passes: {len(unique_topics)} topics → {len(canonical_topics)} canonical topics")
    return topic_to_canonical

def compute_tfidf_topic_vectors(
    doc_topics: Dict[str, List[str]],
    topic_mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute TF-IDF vectors for documents based on topic occurrences.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Tuple of (tfidf_df with docs as rows and topics as columns, list of doc_ids)
    """

    doc_ids = list(doc_topics.keys())

    mapped_doc_topics: Dict[str, List[str]] = {}
    for doc_id, topics in doc_topics.items():
        mapped_doc_topics[doc_id] = [topic_mapping.get(t, t) for t in topics]

    all_canonical_topics = sorted(set(topic_mapping.values()))

    n_docs = len(doc_ids)
    n_topics = len(all_canonical_topics)

    tf_matrix = np.zeros((n_docs, n_topics))
    for i, doc_id in enumerate(doc_ids):
        topics = mapped_doc_topics[doc_id]
        for topic in topics:
            if topic in all_canonical_topics:
                j = all_canonical_topics.index(topic)
                tf_matrix[i, j] += 1

    df = np.sum(tf_matrix > 0, axis=0)
    idf = np.log(n_docs / (df + 1))

    tfidf_matrix = tf_matrix * idf

    tfidf_df = pd.DataFrame(
        tfidf_matrix,
        index=doc_ids,
        columns=all_canonical_topics
    )

    return tfidf_df, doc_ids

def cosine_similarity_tfidf(tfidf_df: pd.DataFrame, doc_a: str, doc_b: str) -> Optional[float]:
    """Compute cosine similarity between two documents using their TF-IDF topic vectors."""

    if doc_a not in tfidf_df.index or doc_b not in tfidf_df.index:
        return None

    vec_a = tfidf_df.loc[doc_a].values
    vec_b = tfidf_df.loc[doc_b].values

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return None

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def compute_tf_topic_vectors(
    doc_topics: Dict[str, List[str]],
    topic_mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute TF vectors for documents based on topic occurrences (without IDF).

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Tuple of (tf_df with docs as rows and topics as columns, list of doc_ids)
    """

    doc_ids = list(doc_topics.keys())

    mapped_doc_topics: Dict[str, List[str]] = {}
    for doc_id, topics in doc_topics.items():
        mapped_doc_topics[doc_id] = [topic_mapping.get(t, t) for t in topics]

    all_canonical_topics = sorted(set(topic_mapping.values()))

    n_docs = len(doc_ids)
    n_topics = len(all_canonical_topics)

    tf_matrix = np.zeros((n_docs, n_topics))
    for i, doc_id in enumerate(doc_ids):
        topics = mapped_doc_topics[doc_id]
        for topic in topics:
            if topic in all_canonical_topics:
                j = all_canonical_topics.index(topic)
                tf_matrix[i, j] += 1

    tf_df = pd.DataFrame(
        tf_matrix,
        index=doc_ids,
        columns=all_canonical_topics
    )

    return tf_df, doc_ids

def cosine_similarity_tf(tf_df: pd.DataFrame, doc_a: str, doc_b: str) -> Optional[float]:
    """Compute cosine similarity between two documents using their TF topic vectors."""

    if doc_a not in tf_df.index or doc_b not in tf_df.index:
        return None

    vec_a = tf_df.loc[doc_a].values
    vec_b = tf_df.loc[doc_b].values

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return None

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def jaccard_similarity_topics(
    doc_topics: Dict[str, List[str]],
    topic_mapping: Dict[str, str],
    doc_a: str,
    doc_b: str
) -> Optional[float]:
    """Compute Jaccard similarity between two documents based on topic overlap.

    Parameters
    ----------
    doc_topics
        Dictionary mapping document IDs to lists of topics.
    topic_mapping
        Dictionary mapping original topics to canonical topics.
    doc_a
        First document ID.
    doc_b
        Second document ID.

    Returns
    -------
    Optional[float]
        Jaccard similarity = |topics_A ∩ topics_B| / |topics_A ∪ topics_B|
    """

    if doc_a not in doc_topics or doc_b not in doc_topics:
        return None

    topics_a = set(topic_mapping.get(t, t) for t in doc_topics[doc_a])
    topics_b = set(topic_mapping.get(t, t) for t in doc_topics[doc_b])

    if not topics_a and not topics_b:
        return 1.0
    if not topics_a or not topics_b:
        return 0.0

    intersection = len(topics_a.intersection(topics_b))
    union = len(topics_a.union(topics_b))

    if union == 0:
        return None

    return float(intersection / union)

def compute_distinctive_topics_by_role(
    tfidf_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    top_n: int = 5
) -> pd.DataFrame:
    """Compute the most distinctive topics for each role based on TF-IDF scores.

    Parameters
    ----------
    tfidf_df
        TF-IDF matrix with documents as rows and topics as columns.
    trace_df
        Trace dataframe with role information.
    top_n
        Number of top topics to extract per role.

    Returns
    -------
    pd.DataFrame
        DataFrame with role, topic, and average TF-IDF score.
    """

    role_topics_data = []

    for role in ROLES:

        role_docs = []
        for _, row in trace_df.iterrows():
            if str(row["role"]) == role:
                seed = row.get("seed")
                if pd.notna(seed):
                    doc_id = f"final_{role}_seed_{int(seed)}"
                else:
                    doc_id = f"final_{role}"

                if doc_id in tfidf_df.index:
                    role_docs.append(doc_id)

        if not role_docs:
            continue

        role_tfidf = tfidf_df.loc[role_docs].mean(axis=0)

        top_topics = role_tfidf.nlargest(top_n)

        for topic, score in top_topics.items():
            role_topics_data.append({
                "role": role,
                "topic": topic,
                "avg_tfidf_score": score
            })

    return pd.DataFrame(role_topics_data)

def compute_topic_frequencies(doc_topics: Dict[str, List[str]], topic_mapping: Dict[str, str]) -> pd.DataFrame:
    """Compute topic frequencies across all documents.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: topic, frequency, document_count, avg_frequency_per_doc
    """

    all_topics = []
    topic_doc_counts = {}

    for doc_id, topics in doc_topics.items():
        mapped_topics = [topic_mapping.get(t, t) for t in topics]
        all_topics.extend(mapped_topics)

        unique_topics = set(mapped_topics)
        for topic in unique_topics:
            topic_doc_counts[topic] = topic_doc_counts.get(topic, 0) + 1

    topic_counts = Counter(all_topics)
    total_docs = len(doc_topics)

    frequency_data = []
    for topic, count in topic_counts.most_common():
        frequency_data.append({
            "topic": topic,
            "frequency": count,
            "document_count": topic_doc_counts[topic],
            "avg_frequency_per_doc": count / total_docs,
            "document_percentage": (topic_doc_counts[topic] / total_docs) * 100
        })

    return pd.DataFrame(frequency_data)

def main() -> None:
    """Entry point for topic-based similarity analysis using s33 cached data."""

    np.random.seed(SEED)

    client = configure_openai_client()

    print("Loading s33 trace data...")
    if not S33_TRACE_PATH.exists():
        raise FileNotFoundError(f"s33 trace file not found: {S33_TRACE_PATH}")

    trace_df = pd.read_excel(S33_TRACE_PATH)

    original_final = read_text_file(ORIGINAL_FINAL_PATH)
    fed_draft = read_text_file(AGENCY_TO_DRAFT["fed"])
    occ_draft = read_text_file(AGENCY_TO_DRAFT["occ"])
    fdic_draft = read_text_file(AGENCY_TO_DRAFT["fdic"])

    doc_texts: Dict[str, str] = {
        "official_final": original_final,
        "draft_fed": fed_draft,
        "draft_occ": occ_draft,
        "draft_fdic": fdic_draft,
    }

    for _, row in trace_df.iterrows():
        role = str(row["role"])
        seed = row.get("seed")
        revised_policy = str(row.get("revised_policy", ""))
        if revised_policy and revised_policy.strip():
            if pd.notna(seed):
                doc_id = f"final_{role}_seed_{int(seed)}"
            else:
                doc_id = f"final_{role}"
            doc_texts[doc_id] = revised_policy

    print(f"Total documents to process: {len(doc_texts)}")

    if TOPIC_CACHE_PATH.exists():
        print("Loading cached topics...")
        doc_topics: Dict[str, List[str]] = json.loads(TOPIC_CACHE_PATH.read_text(encoding="utf-8"))
        missing_docs = [doc_id for doc_id in doc_texts.keys() if doc_id not in doc_topics]
        if missing_docs:
            print(f"Extracting topics for {len(missing_docs)} new documents in parallel...")
            missing_items = [(doc_id, doc_texts[doc_id]) for doc_id in missing_docs]
            missing_topics = extract_topics_parallel(missing_items, client)
            doc_topics.update(missing_topics)
            TOPIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            TOPIC_CACHE_PATH.write_text(json.dumps(doc_topics, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print("Extracting topics from all documents in parallel...")
        doc_items = list(doc_texts.items())
        doc_topics = extract_topics_parallel(doc_items, client)

        TOPIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOPIC_CACHE_PATH.write_text(json.dumps(doc_topics, ensure_ascii=False, indent=2), encoding="utf-8")

    all_topics = [topic for topics in doc_topics.values() for topic in topics]
    print(f"Total topics extracted: {len(all_topics)}")
    print(f"Unique topics: {len(set(all_topics))}")

    if MERGED_TOPICS_PATH.exists():
        print("Loading cached topic mapping...")
        topic_mapping: Dict[str, str] = json.loads(MERGED_TOPICS_PATH.read_text(encoding="utf-8"))
    else:
        print("Merging similar topics...")
        topic_mapping = merge_similar_topics(all_topics, threshold=TOPIC_MERGE_THRESHOLD)
        MERGED_TOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        MERGED_TOPICS_PATH.write_text(json.dumps(topic_mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Computing TF-IDF topic vectors...")
    tfidf_df, doc_ids = compute_tfidf_topic_vectors(doc_topics, topic_mapping)

    TFIDF_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tfidf_df.to_csv(TFIDF_MATRIX_PATH)
    print(f"Saved TF-IDF matrix: {TFIDF_MATRIX_PATH}")

    print("Computing TF topic vectors...")
    tf_df, _ = compute_tf_topic_vectors(doc_topics, topic_mapping)

    TF_MATRIX_PATH = Path("outputs/s34_tf_matrix.csv")
    TF_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tf_df.to_csv(TF_MATRIX_PATH)
    print(f"Saved TF matrix: {TF_MATRIX_PATH}")

    print("Computing topic frequencies...")
    topic_frequency_df = compute_topic_frequencies(doc_topics, topic_mapping)

    OUTPUT_TOPIC_FREQUENCY_CSV.parent.mkdir(parents=True, exist_ok=True)
    topic_frequency_df.to_csv(OUTPUT_TOPIC_FREQUENCY_CSV, index=False)
    print(f"Saved topic frequency data: {OUTPUT_TOPIC_FREQUENCY_CSV}")

    print("Computing topic-based similarities...")

    role_seed_similarities: List[Dict[str, object]] = []

    for _, row in tqdm(trace_df.iterrows(), total=len(trace_df), desc="Computing similarities"):
        role = str(row["role"])
        seed = row.get("seed")

        if pd.notna(seed):
            doc_id = f"final_{role}_seed_{int(seed)}"
        else:
            doc_id = f"final_{role}"

        if doc_id not in tfidf_df.index:
            continue

        tfidf_sim_to_final = cosine_similarity_tfidf(tfidf_df, doc_id, "official_final")
        tfidf_sim_to_draft = cosine_similarity_tfidf(tfidf_df, doc_id, "draft_fed")

        tf_sim_to_final = cosine_similarity_tf(tf_df, doc_id, "official_final")
        tf_sim_to_draft = cosine_similarity_tf(tf_df, doc_id, "draft_fed")

        jaccard_sim_to_final = jaccard_similarity_topics(doc_topics, topic_mapping, doc_id, "official_final")
        jaccard_sim_to_draft = jaccard_similarity_topics(doc_topics, topic_mapping, doc_id, "draft_fed")

        role_seed_similarities.append({
            "role": role,
            "seed": int(seed) if pd.notna(seed) else None,
            "tfidf_similarity_to_final": tfidf_sim_to_final,
            "tfidf_similarity_to_draft": tfidf_sim_to_draft,
            "tf_similarity_to_final": tf_sim_to_final,
            "tf_similarity_to_draft": tf_sim_to_draft,
            "jaccard_similarity_to_final": jaccard_sim_to_final,
            "jaccard_similarity_to_draft": jaccard_sim_to_draft,
        })

    baseline_similarities = [
        {
            "role": "draft_fed",
            "seed": None,
            "tfidf_similarity_to_final": cosine_similarity_tfidf(tfidf_df, "draft_fed", "official_final"),
            "tfidf_similarity_to_draft": 1.0,
            "tf_similarity_to_final": cosine_similarity_tf(tf_df, "draft_fed", "official_final"),
            "tf_similarity_to_draft": 1.0,
            "jaccard_similarity_to_final": jaccard_similarity_topics(doc_topics, topic_mapping, "draft_fed", "official_final"),
            "jaccard_similarity_to_draft": 1.0,
        },
        {
            "role": "draft_occ",
            "seed": None,
            "tfidf_similarity_to_final": cosine_similarity_tfidf(tfidf_df, "draft_occ", "official_final"),
            "tfidf_similarity_to_draft": cosine_similarity_tfidf(tfidf_df, "draft_occ", "draft_fed"),
            "tf_similarity_to_final": cosine_similarity_tf(tf_df, "draft_occ", "official_final"),
            "tf_similarity_to_draft": cosine_similarity_tf(tf_df, "draft_occ", "draft_fed"),
            "jaccard_similarity_to_final": jaccard_similarity_topics(doc_topics, topic_mapping, "draft_occ", "official_final"),
            "jaccard_similarity_to_draft": jaccard_similarity_topics(doc_topics, topic_mapping, "draft_occ", "draft_fed"),
        },
        {
            "role": "draft_fdic",
            "seed": None,
            "tfidf_similarity_to_final": cosine_similarity_tfidf(tfidf_df, "draft_fdic", "official_final"),
            "tfidf_similarity_to_draft": cosine_similarity_tfidf(tfidf_df, "draft_fdic", "draft_fed"),
            "tf_similarity_to_final": cosine_similarity_tf(tf_df, "draft_fdic", "official_final"),
            "tf_similarity_to_draft": cosine_similarity_tf(tf_df, "draft_fdic", "draft_fed"),
            "jaccard_similarity_to_final": jaccard_similarity_topics(doc_topics, topic_mapping, "draft_fdic", "official_final"),
            "jaccard_similarity_to_draft": jaccard_similarity_topics(doc_topics, topic_mapping, "draft_fdic", "draft_fed"),
        },
    ]

    all_similarities = role_seed_similarities + baseline_similarities
    sim_df = pd.DataFrame(all_similarities)

    role_tfidf_similarity_mc_df = sim_df[
        (sim_df["role"].isin(ROLES)) & (sim_df["seed"].notna())
    ][["role", "seed", "tfidf_similarity_to_final"]].copy()
    role_tfidf_similarity_mc_df = role_tfidf_similarity_mc_df.rename(columns={"tfidf_similarity_to_final": "similarity"})

    role_tf_similarity_mc_df = sim_df[
        (sim_df["role"].isin(ROLES)) & (sim_df["seed"].notna())
    ][["role", "seed", "tf_similarity_to_final"]].copy()
    role_tf_similarity_mc_df = role_tf_similarity_mc_df.rename(columns={"tf_similarity_to_final": "similarity"})

    role_jaccard_similarity_mc_df = sim_df[
        (sim_df["role"].isin(ROLES)) & (sim_df["seed"].notna())
    ][["role", "seed", "jaccard_similarity_to_final"]].copy()
    role_jaccard_similarity_mc_df = role_jaccard_similarity_mc_df.rename(columns={"jaccard_similarity_to_final": "similarity"})

    baseline_fed_tfidf = baseline_similarities[0]["tfidf_similarity_to_final"]
    baseline_occ_tfidf = baseline_similarities[1]["tfidf_similarity_to_final"]
    baseline_fdic_tfidf = baseline_similarities[2]["tfidf_similarity_to_final"]

    baseline_fed_tf = baseline_similarities[0]["tf_similarity_to_final"]
    baseline_occ_tf = baseline_similarities[1]["tf_similarity_to_final"]
    baseline_fdic_tf = baseline_similarities[2]["tf_similarity_to_final"]

    baseline_fed_jaccard = baseline_similarities[0]["jaccard_similarity_to_final"]
    baseline_occ_jaccard = baseline_similarities[1]["jaccard_similarity_to_final"]
    baseline_fdic_jaccard = baseline_similarities[2]["jaccard_similarity_to_final"]

    if not role_tfidf_similarity_mc_df.empty:
        print("Generating TF-IDF similarity plots...")

        with tqdm(total=4, desc="Generating TF-IDF plots") as pbar:
            order_df = (
                role_tfidf_similarity_mc_df.groupby("role", as_index=False)["similarity"].mean()
                .sort_values("similarity", ascending=False)
            )
            ordered_roles = list(order_df["role"].values)
            pbar.update(1)

        n_rows = len(ordered_roles)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(3, 2 * n_rows)), sharex=True)
        if n_rows == 1:
            axes = [axes]

        for ax, role_name in zip(axes, ordered_roles):
            vals = (
                role_tfidf_similarity_mc_df.loc[role_tfidf_similarity_mc_df["role"] == role_name, "similarity"]
                .astype(float)
                .dropna()
            )
            if len(vals) >= 2:
                sns.kdeplot(
                    vals,
                    ax=ax,
                    fill=True,
                    common_norm=False,
                    alpha=0.25,
                    linewidth=2,
                    color="#8b8b8b",
                )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylabel(role_name)
            if baseline_fed_tfidf is not None:
                ax.axvline(baseline_fed_tfidf, linestyle="--", color="black", linewidth=1.2)
            if baseline_occ_tfidf is not None:
                ax.axvline(baseline_occ_tfidf, linestyle=(0, (5, 5)), color="dimgray", linewidth=1.2)
            if baseline_fdic_tfidf is not None:
                ax.axvline(baseline_fdic_tfidf, linestyle=(0, (3, 3)), color="gray", linewidth=1.2)

        legend_lines: List[Line2D] = []
        legend_labels: List[str] = []
        if baseline_fed_tfidf is not None:
            legend_lines.append(Line2D([0], [0], color="black", lw=1.2, linestyle="--"))
            legend_labels.append("Draft FRS")
        if baseline_occ_tfidf is not None:
            legend_lines.append(Line2D([0], [0], color="dimgray", lw=1.2, linestyle=(0, (5, 5))))
            legend_labels.append("Draft OCC")
        if baseline_fdic_tfidf is not None:
            legend_lines.append(Line2D([0], [0], color="gray", lw=1.2, linestyle=(0, (3, 3))))
            legend_labels.append("Draft FDIC")

        axes[-1].set_xlabel("Topic-based TF-IDF similarity to official final (cosine)")
        if legend_lines:
            fig.legend(legend_lines, legend_labels, title="Draft baselines", loc="upper right")

        plt.tight_layout()
        OUTPUT_TFIDF_STACKED_PNG.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_TFIDF_STACKED_PNG, dpi=200)
        plt.savefig(OUTPUT_TFIDF_STACKED_SVG)
        plt.close()
        pbar.update(1)

        sim_summary = (
            role_tfidf_similarity_mc_df.groupby("role", as_index=False)
            .agg(mean=("similarity", "mean"), std=("similarity", "std"), n=("seed", "count"))
        )
        sim_summary["se"] = sim_summary["std"] / sim_summary["n"].clip(lower=1).pow(0.5)
        sim_summary["ci_95_margin"] = 1.96 * sim_summary["se"].fillna(0.0)
        sim_summary = sim_summary.sort_values("mean", ascending=False)

        OUTPUT_TFIDF_SIM_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        sim_summary.to_csv(OUTPUT_TFIDF_SIM_SUMMARY_CSV, index=False)

        plt.figure(figsize=(10, max(4, int(0.6 * len(sim_summary)))))
        ax = sns.barplot(
            data=sim_summary,
            y="role",
            x="mean",
            color="#8b8b8b",
            orient="h",
            errorbar=None,
        )
        for i, (_, row) in enumerate(sim_summary.iterrows()):
            ax.errorbar(
                x=float(row["mean"]),
                y=i,
                xerr=float(row["ci_95_margin"]),
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Topic-based TF-IDF similarity to official final (mean across seeds) with 95% CI")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(OUTPUT_TFIDF_SIM_SUMMARY_PNG, dpi=200)
        plt.savefig(OUTPUT_TFIDF_SIM_SUMMARY_SVG)
        plt.close()
        pbar.update(1)

    if not role_jaccard_similarity_mc_df.empty:
        print("Generating Jaccard similarity plots...")

        jaccard_sim_summary = (
            role_jaccard_similarity_mc_df.groupby("role", as_index=False)
            .agg(mean=("similarity", "mean"), std=("similarity", "std"), n=("seed", "count"))
        )
        jaccard_sim_summary["se"] = jaccard_sim_summary["std"] / jaccard_sim_summary["n"].clip(lower=1).pow(0.5)
        jaccard_sim_summary["ci_95_margin"] = 1.96 * jaccard_sim_summary["se"].fillna(0.0)
        jaccard_sim_summary = jaccard_sim_summary.sort_values("mean", ascending=False)

        OUTPUT_JACCARD_SIM_CSV.parent.mkdir(parents=True, exist_ok=True)
        jaccard_sim_summary.to_csv(OUTPUT_JACCARD_SIM_CSV, index=False)

        plt.figure(figsize=(10, max(4, int(0.6 * len(jaccard_sim_summary)))))
        ax = sns.barplot(
            data=jaccard_sim_summary,
            y="role",
            x="mean",
            color="#8b8b8b",
            orient="h",
            errorbar=None,
        )
        for i, (_, row) in enumerate(jaccard_sim_summary.iterrows()):
            ax.errorbar(
                x=float(row["mean"]),
                y=i,
                xerr=float(row["ci_95_margin"]),
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Topic overlap similarity to official final (mean across seeds) with 95% CI")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(OUTPUT_JACCARD_PNG, dpi=200)
        plt.savefig(OUTPUT_JACCARD_SVG)
        plt.close()

    if not role_tf_similarity_mc_df.empty:
        print("Generating TF similarity plots...")

        tf_sim_summary = (
            role_tf_similarity_mc_df.groupby("role", as_index=False)
            .agg(mean=("similarity", "mean"), std=("similarity", "std"), n=("seed", "count"))
        )
        tf_sim_summary["se"] = tf_sim_summary["std"] / tf_sim_summary["n"].clip(lower=1).pow(0.5)
        tf_sim_summary["ci_95_margin"] = 1.96 * tf_sim_summary["se"].fillna(0.0)
        tf_sim_summary = tf_sim_summary.sort_values("mean", ascending=False)

        OUTPUT_TF_SIM_CSV.parent.mkdir(parents=True, exist_ok=True)
        tf_sim_summary.to_csv(OUTPUT_TF_SIM_CSV, index=False)

        plt.figure(figsize=(10, max(4, int(0.6 * len(tf_sim_summary)))))
        ax = sns.barplot(
            data=tf_sim_summary,
            y="role",
            x="mean",
            color="#8b8b8b",
            orient="h",
            errorbar=None,
        )
        for i, (_, row) in enumerate(tf_sim_summary.iterrows()):
            ax.errorbar(
                x=float(row["mean"]),
                y=i,
                xerr=float(row["ci_95_margin"]),
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Topic frequency similarity to official final (mean across seeds) with 95% CI")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(OUTPUT_TF_PNG, dpi=200)
        plt.savefig(OUTPUT_TF_SVG)
        plt.close()

    print("Generating topic frequency plot...")

    top_topics = topic_frequency_df.head(20)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=top_topics,
        y="topic",
        x="frequency",
        color="#8b8b8b",
        orient="h"
    )

    for i, (_, row) in enumerate(top_topics.iterrows()):
        ax.text(
            row["frequency"] + 0.5,
            i,
            f"{row[\"document_percentage\"]:.1f}%",
            va="center",
            ha="left",
            fontsize=9
        )

    ax.set_xlabel("Total Frequency Across All Documents")
    ax.set_ylabel("Topic")
    ax.set_title("Top 20 Most Frequent Topics\n(Percentage shows documents containing this topic)")
    plt.tight_layout()

    OUTPUT_TOPIC_FREQUENCY_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_TOPIC_FREQUENCY_PNG, dpi=200)
    plt.savefig(OUTPUT_TOPIC_FREQUENCY_SVG)
    plt.close()
    print(f"Saved topic frequency plot: {OUTPUT_TOPIC_FREQUENCY_PNG}")

    print("Computing distinctive topics by role...")
    role_topics_df = compute_distinctive_topics_by_role(tfidf_df, trace_df, top_n=5)

    OUTPUT_ROLE_TOPICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    role_topics_df.to_csv(OUTPUT_ROLE_TOPICS_CSV, index=False)
    print(f"Saved role topics data: {OUTPUT_ROLE_TOPICS_CSV}")

    print("Generating distinctive topics by role plot...")
    n_roles = len(ROLES)
    n_cols = 3
    n_rows = (n_roles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for idx, role in enumerate(ROLES):
        ax = axes[idx]
        role_data = role_topics_df[role_topics_df["role"] == role].copy()

        if not role_data.empty:

            role_data = role_data.sort_values("avg_tfidf_score", ascending=True)

            bars = ax.barh(
                range(len(role_data)),
                role_data["avg_tfidf_score"],
                color="#8b8b8b",
                alpha=0.7
            )

            topic_labels = [topic[:40] + "..." if len(topic) > 40 else topic
                          for topic in role_data["topic"]]
            ax.set_yticks(range(len(role_data)))
            ax.set_yticklabels(topic_labels, fontsize=8)

            ax.set_xlabel("Avg TF-IDF Score", fontsize=9)
            ax.set_title(f"{role.capitalize()}", fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    for idx in range(len(ROLES), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    OUTPUT_ROLE_TOPICS_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_ROLE_TOPICS_PNG, dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_ROLE_TOPICS_SVG, bbox_inches="tight")
    plt.close()
    print(f"Saved distinctive topics by role plot: {OUTPUT_ROLE_TOPICS_PNG}")

    if not role_tfidf_similarity_mc_df.empty:
        role_groups = {
            rn: role_tfidf_similarity_mc_df.loc[role_tfidf_similarity_mc_df["role"] == rn, "similarity"].astype(float).dropna().values
            for rn in ROLES
        }
        r = len(ROLES)
        t_mat = np.zeros((r, r), dtype=float)
        for i, ri in enumerate(ROLES):
            xi = role_groups.get(ri, np.array([], dtype=float))
            for j, rj in enumerate(ROLES):
                if i == j:
                    t_mat[i, j] = 0.0
                    continue
                xj = role_groups.get(rj, np.array([], dtype=float))
                if len(xi) >= 2 and len(xj) >= 2:
                    t_stat, _ = ttest_ind(xi, xj, equal_var=False, nan_policy="omit")
                    t_mat[i, j] = float(t_stat) if np.isfinite(t_stat) else np.nan
                else:
                    t_mat[i, j] = np.nan

        t_df = pd.DataFrame(t_mat, index=ROLES, columns=ROLES)
        OUTPUT_TFIDF_TTEST_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
        t_df.to_csv(OUTPUT_TFIDF_TTEST_MATRIX_CSV)

        plt.figure(figsize=(max(6, int(0.6 * r)), max(4, int(0.6 * r))))
        cat_mat = np.where(
            np.isnan(t_mat),
            np.nan,
            np.where(t_mat > 1.645, 1.0, np.where(t_mat < -1.645, -1.0, 0.0)),
        )
        cat_df = pd.DataFrame(cat_mat, index=ROLES, columns=ROLES)
        cmap = ListedColormap(["#1f77b4", "#bfbfbf", "#d62728"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)
        sns.heatmap(
            cat_df,
            cmap=cmap,
            norm=norm,
            annot=t_df.round(2),
            fmt=".2f",
            cbar=False,
        )
        plt.title("Welch t-statistics across roles (topic-based TF-IDF similarity to official final)")
        plt.tight_layout()
        plt.savefig(OUTPUT_TFIDF_TTEST_HEATMAP_PNG, dpi=200)
        plt.savefig(OUTPUT_TFIDF_TTEST_HEATMAP_SVG)
        plt.close()
        pbar.update(1)

    print(f"\nSaved outputs:")
    print(f"  Topic cache → {TOPIC_CACHE_PATH}")
    print(f"  Merged topics → {MERGED_TOPICS_PATH}")
    print(f"  TF-IDF matrix → {TFIDF_MATRIX_PATH}")
    print(f"  TF matrix → {TF_MATRIX_PATH}")
    print(f"  Topic frequency data → {OUTPUT_TOPIC_FREQUENCY_CSV}")
    print(f"  Topic frequency plot → {OUTPUT_TOPIC_FREQUENCY_PNG}, {OUTPUT_TOPIC_FREQUENCY_SVG}")
    print(f"  Distinctive topics by role → {OUTPUT_ROLE_TOPICS_CSV}")
    print(f"  Distinctive topics plot → {OUTPUT_ROLE_TOPICS_PNG}, {OUTPUT_ROLE_TOPICS_SVG}")
    print(f"  TF-IDF similarity summary → {OUTPUT_TFIDF_SIM_SUMMARY_CSV}")
    print(f"  TF-IDF similarity plot → {OUTPUT_TFIDF_SIM_SUMMARY_PNG}, {OUTPUT_TFIDF_SIM_SUMMARY_SVG}")
    print(f"  TF-IDF stacked distributions → {OUTPUT_TFIDF_STACKED_PNG}, {OUTPUT_TFIDF_STACKED_SVG}")
    print(f"  TF similarity summary → {OUTPUT_TF_SIM_CSV}")
    print(f"  TF similarity plot → {OUTPUT_TF_PNG}, {OUTPUT_TF_SVG}")
    print(f"  Jaccard similarity summary → {OUTPUT_JACCARD_SIM_CSV}")
    print(f"  Jaccard similarity plot → {OUTPUT_JACCARD_PNG}, {OUTPUT_JACCARD_SVG}")
    print(f"  TF-IDF t-test matrix → {OUTPUT_TFIDF_TTEST_MATRIX_CSV}")
    print(f"  TF-IDF t-test heatmap → {OUTPUT_TFIDF_TTEST_HEATMAP_PNG}, {OUTPUT_TFIDF_TTEST_HEATMAP_SVG}")

if __name__ == "__main__":
    main()
