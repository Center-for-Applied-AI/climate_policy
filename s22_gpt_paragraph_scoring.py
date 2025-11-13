#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import openai
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
import instructor
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplot2tikz import save as tikz_save


DATA_DIR = Path("data/policies/parsed")
AFTER_FILE = "final.md"
BEFORE_FILES = {
    "fdic": "fdic.md",
    "occ": "occ.md",
}
ANALYSIS_RANGE = range(65, 84)

EMB_MODEL = "text-embedding-3-large"
TEXT_WORD_LIMIT = 80_000
BATCH_WORD_LIMIT = 80_000

MODEL_NAME = "gpt-4o-2024-08-06"
SEED = 42

EMBEDDINGS_JSON = Path("outputs/comment_embeddings.json")
OUTPUT_XLSX = Path("outputs/s22_paragraph_scores.xlsx")

PLOTS_DIR = Path("plots")
OUTPUT_BASE_HIST_OCC = PLOTS_DIR / "score_histogram_OCC_paragraphs"
OUTPUT_BASE_HIST_FDIC = PLOTS_DIR / "score_histogram_FDIC_paragraphs"
OUTPUT_BASE_HIST_FINAL = PLOTS_DIR / "score_histogram_FINAL_paragraphs"
GREY_COLOR = "#7f7f7f"

import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

CLIENT = instructor.patch(openai.OpenAI(api_key=openai.api_key))

PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")
WORD_PATTERN = re.compile(r"\w+")


def word_len(text: str) -> int:
    """Return Unicode word-count for batching and truncation.
    
    Args:
        text: Input text.
        
    Returns:
        Number of words.
    """
    return len(WORD_PATTERN.findall(text))


def truncate_by_words(text: str) -> tuple[str, int]:
    """Truncate text to TEXT_WORD_LIMIT words.
    
    Args:
        text: Input text.
        
    Returns:
        Tuple of (text_or_trimmed, flag).
    """
    words = WORD_PATTERN.findall(text)
    if len(words) <= TEXT_WORD_LIMIT:
        return text, 0
    return " ".join(words[:TEXT_WORD_LIMIT]), 1


def read_paragraphs(path: Path) -> list[str]:
    """Read Markdown, drop headings, split paragraphs on blank lines.
    
    Args:
        path: Path to markdown file.
        
    Returns:
        List of paragraph strings.
    """
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if not ln.lstrip().startswith("#")]
    return [p.strip() for p in PARAGRAPH_SPLIT_PATTERN.split("\n".join(lines)) if p.strip()]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embedding endpoint for a batch of texts.
    
    Args:
        texts: List of text strings.
        
    Returns:
        List of embedding vectors.
    """
    resp = openai.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_batches(texts: list[str]) -> tuple[list[list[float]], list[int]]:
    """Embed texts with batching by word-count.
    
    Returns embeddings and keep_flags.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        Tuple of (vectors, keep_flags).
    """
    vectors: list[list[float]] = []
    keep: list[int] = []

    def process_batch(batch_texts: list[str]) -> None:
        if not batch_texts:
            return
        
        vectors.extend(embed_batch(batch_texts))
        keep.extend([1] * len(batch_texts))

    acc_words = 0
    batch: list[str] = []
    for t in tqdm(texts, desc="Embedding", leave=False):
        wl = word_len(t)
        if acc_words + wl > BATCH_WORD_LIMIT:
            process_batch(batch)
            batch = [t]
            acc_words = wl
        else:
            batch.append(t)
            acc_words += wl
    process_batch(batch)
    return vectors, keep


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for every row-vector pair of a and b.
    
    Args:
        a: First matrix (m x d).
        b: Second matrix (n x d).
        
    Returns:
        Similarity matrix (m x n).
    """
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_n @ b_n.T


class ParagraphClassification(BaseModel):
    """Climate engagement classification for a paragraph."""

    score: int | None = Field(
        ..., description="The climate policy engagement score from 1 to 5. Null if not applicable."
    )
    explanation: str = Field(..., description="A brief explanation for the score, using quotes from the text.")


def classify_text(text: str, model: str = MODEL_NAME) -> dict:
    """Classify a paragraph using the s9 prompt without metadata extraction.
    
    Args:
        text: The text to classify.
        model: The OpenAI model to use.
        
    Returns:
        Dictionary with score and explanation.
    """
    prompt = (
        "You will receive a comment on a policy proposal addressed to a U.S. regulator.\n"
        "Your task is to analyze the comment and return a structured JSON object with a climate policy engagement score.\n\n"
        "Assign a climate policy engagement score from 1 to 5 according to the definitions below, "
        "and provide a brief explanation that uses quotes from the text.\n"
        "Score definitions:\n"
        "1 = Strong opposition to climate action by the regulator. Explicitly resists climate measures. "
        "May deny climate or climate risks.\n"
        "2 = Skeptical or hesitant. Questions the need for special treatment or warns about costs and unintended consequences.\n"
        "3 = Neutral. Takes no strong position for or against climate action.\n"
        "4 = Supportive. Backs climate actions of the regulator. May support other climate measures. "
        "May advocate for more incremental steps.\n"
        "5 = Strong advocate. Fully supports ambitious, binding climate targets and broad reforms. "
        "May seek to strengthen proposed initiatives.\n\n"
        "[TEXT]\n" + text
    )
    
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=ParagraphClassification,
        seed=SEED,
    )
    result = resp.model_dump()
    if result.get("score") is None:
        result["score"] = "na"
    return result


def save_all_formats(fig: plt.Figure, base_path: Path) -> None:
    """Save a figure to PNG, SVG, and TiKZ (.tex).
    
    Args:
        fig: The matplotlib figure to save.
        base_path: Base path without extension.
    """
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    tikz_save(str(base_path.with_suffix(".tex")), figure=fig)


def plot_histograms(paragraph_score_before_df: pd.DataFrame) -> None:
    """Plot histograms for OCC and FDIC paragraph scores.
    
    Args:
        paragraph_score_before_df: DataFrame with before paragraph scores.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["svg.fonttype"] = "none"

    occ_scores = paragraph_score_before_df.loc[paragraph_score_before_df["source"] == "OCC", "score"]
    fdic_scores = paragraph_score_before_df.loc[paragraph_score_before_df["source"] == "FDIC", "score"]

    bins = np.arange(1, 7) - 0.5

    fig_occ, ax_occ = plt.subplots(figsize=(6, 6))
    ax_occ.hist(occ_scores, bins=bins, color=GREY_COLOR, edgecolor="black")
    ax_occ.set_xlabel("Climate Engagement Score")
    ax_occ.set_ylabel("Count")
    ax_occ.set_xticks(range(1, 6))
    plt.tight_layout()
    save_all_formats(fig_occ, OUTPUT_BASE_HIST_OCC)

    fig_fdic, ax_fdic = plt.subplots(figsize=(6, 6))
    ax_fdic.hist(fdic_scores, bins=bins, color=GREY_COLOR, edgecolor="black")
    ax_fdic.set_xlabel("Climate Engagement Score")
    ax_fdic.set_ylabel("Count")
    ax_fdic.set_xticks(range(1, 6))
    plt.tight_layout()
    save_all_formats(fig_fdic, OUTPUT_BASE_HIST_FDIC)


def plot_histogram_final(paragraph_score_after_df: pd.DataFrame) -> None:
    """Plot histogram for final paragraph scores.
    
    Args:
        paragraph_score_after_df: DataFrame with after paragraph scores.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["svg.fonttype"] = "none"

    bins = np.arange(1, 7) - 0.5

    fig_f, ax_f = plt.subplots(figsize=(6, 6))
    ax_f.hist(paragraph_score_after_df["score"], bins=bins, color=GREY_COLOR, edgecolor="black")
    ax_f.set_xlabel("Climate Engagement Score")
    ax_f.set_ylabel("Count")
    ax_f.set_xticks(range(1, 6))
    plt.tight_layout()
    save_all_formats(fig_f, OUTPUT_BASE_HIST_FINAL)


def main() -> None:
    """Score before and after paragraphs with GPT and export visualizations."""
    after_all = read_paragraphs(DATA_DIR / AFTER_FILE)
    after_texts_all, _ = zip(*(truncate_by_words(p) for p in after_all))
    after_texts = [after_texts_all[i - 1] for i in ANALYSIS_RANGE]

    payload = None
    if EMBEDDINGS_JSON.exists():
        with EMBEDDINGS_JSON.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

    if payload and "after" in payload:
        par_nums = [int(i) for i in payload["after"].get("paragraph_nums", [])]
        if par_nums == [int(i) for i in ANALYSIS_RANGE]:
            after_vecs = np.array(payload["after"].get("embeddings", []))
        else:
            after_vecs_list, _ = embed_batches(after_texts)
            after_vecs = np.array(after_vecs_list)
    else:
        after_vecs_list, _ = embed_batches(after_texts)
        after_vecs = np.array(after_vecs_list)

    paragraph_score_records: list[dict] = []

    for r, final_par_num in tqdm(list(enumerate(ANALYSIS_RANGE)), desc="Scoring FINAL paragraphs"):
        text = after_texts[r]
        res = classify_text(text)
        paragraph_score_records.append(
            {
                "agency": "final",
                "type": "after",
                "paragraph_num": int(final_par_num),
                "text": text,
                "score": res.get("score"),
                "explanation": res.get("explanation"),
            }
        )

    for agency, before_name in BEFORE_FILES.items():
        before_paras = read_paragraphs(DATA_DIR / before_name)
        before_texts_all, _ = zip(*(truncate_by_words(p) for p in before_paras))
        before_vecs = np.array(embed_batches(list(before_texts_all))[0])

        sim_ab = cosine_sim_matrix(after_vecs, before_vecs)
        best_before_idx = np.argmax(sim_ab, axis=1)

        unique_before_indices = sorted(set(int(i) for i in best_before_idx))
        before_to_text: dict[int, str] = {i: before_paras[i] for i in unique_before_indices}

        to_score: list[tuple[str, str, int]] = []
        for r, final_par_num in enumerate(ANALYSIS_RANGE):
            b_idx = int(best_before_idx[r])
            to_score.append(("before", before_to_text[b_idx], int(final_par_num)))

        for kind, text, par_num in tqdm(to_score, desc=f"Scoring {agency.upper()} paragraphs"):
            res = classify_text(text)
            paragraph_score_records.append(
                {
                    "agency": agency,
                    "type": kind,
                    "paragraph_num": par_num,
                    "text": text,
                    "score": res.get("score"),
                    "explanation": res.get("explanation"),
                }
            )

    paragraph_score_df = pd.DataFrame.from_records(paragraph_score_records)
    paragraph_score_df.to_excel(OUTPUT_XLSX, index=False)

    paragraph_score_before_df = paragraph_score_df[
        (paragraph_score_df["type"] == "before")
        & (paragraph_score_df["score"].apply(lambda x: isinstance(x, (int, float))))
    ].copy()
    paragraph_score_before_df["score"] = pd.to_numeric(paragraph_score_before_df["score"])
    paragraph_score_before_df["source"] = paragraph_score_before_df["agency"].str.upper().replace(
        {"OCC": "OCC", "FDIC": "FDIC"}
    )

    plot_histograms(paragraph_score_before_df)

    paragraph_score_after_df = paragraph_score_df[
        (paragraph_score_df["type"] == "after")
        & (paragraph_score_df["agency"] == "final")
        & (paragraph_score_df["score"].apply(lambda x: isinstance(x, (int, float))))
    ].copy()
    if not paragraph_score_after_df.empty:
        paragraph_score_after_df["score"] = pd.to_numeric(paragraph_score_after_df["score"])
        plot_histogram_final(paragraph_score_after_df)

    print(f"Wrote {OUTPUT_XLSX.name} ({len(paragraph_score_df)} rows)")
    print(
        f"Histograms saved to {OUTPUT_BASE_HIST_OCC.with_suffix('.png')}, "
        f"{OUTPUT_BASE_HIST_FDIC.with_suffix('.png')}, and "
        f"{OUTPUT_BASE_HIST_FINAL.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
