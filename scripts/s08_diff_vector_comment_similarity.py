#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import openai
from tqdm.auto import tqdm


DATA_DIR = Path("data/policies/parsed")
COMMENT_DIRS = [
    Path("data/comments/occ/txt"),
    Path("data/comments/fdic/txt"),
    Path("data/comments/fed/txt"),
]
SOURCE_FILES = ["fdic.md", "occ.md", "fed.md"]
REFERENCE_FILE = "final.md"
ANALYSIS_RANGE = range(65, 84)
ANALYSIS_TOP = 3

EMB_MODEL = "text-embedding-3-large"
TEXT_WORD_LIMIT = 8_000
BATCH_WORD_LIMIT = 8_000

import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")
WORD_PATTERN = re.compile(r"\w+")


def word_len(t: str) -> int:
    """Return word count using Unicode word character pattern.
    
    Args:
        t: Input text.
        
    Returns:
        Number of words.
    """
    return len(WORD_PATTERN.findall(t))


def truncate(t: str) -> tuple[str, int]:
    """Truncate text to word limit.
    
    Args:
        t: Input text.
        
    Returns:
        Tuple of (truncated_text, truncation_flag).
    """
    words = WORD_PATTERN.findall(t)
    if len(words) <= TEXT_WORD_LIMIT:
        return t, 0
    return " ".join(words[:TEXT_WORD_LIMIT]), 1


def read_paragraphs(path: Path) -> list[str]:
    """Return paragraphs from a markdown file.
    
    Ignores Markdown heading lines and splits on blank lines (≥2 newlines).
    
    Args:
        path: Path to markdown file.
        
    Returns:
        List of paragraph strings.
    """
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if not ln.lstrip().startswith("#")]
    return [p.strip() for p in PARAGRAPH_SPLIT_PATTERN.split("\n".join(lines)) if p.strip()]


def _embed(batch: list[str]) -> list[list[float]]:
    """Embed a batch of texts.
    
    Args:
        batch: List of text strings.
        
    Returns:
        List of embedding vectors.
    """
    resp = openai.embeddings.create(model=EMB_MODEL, input=batch)
    return [d.embedding for d in resp.data]


def embed_batches(texts: list[str], *, allow_skip: bool = False) -> tuple[list[list[float]], list[int]]:
    """Embed texts in batches with error handling.
    
    Args:
        texts: List of text strings to embed.
        allow_skip: Whether to skip texts that fail to embed.
        
    Returns:
        Tuple of (embeddings, keep_flags).
    """
    vecs: list[list[float]] = []
    keep: list[int] = []

    def process_batch(batch_texts: list[str]):
        nonlocal vecs, keep
        if not batch_texts:
            return
        
        vecs.extend(_embed(batch_texts))
        keep.extend([1] * len(batch_texts))

    batch: list[str] = []
    words_accum = 0
    for txt in tqdm(texts, desc="Embedding"):
        wl = word_len(txt)
        if words_accum + wl > BATCH_WORD_LIMIT:
            process_batch(batch)
            batch = [txt]
            words_accum = wl
        else:
            batch.append(txt)
            words_accum += wl
    process_batch(batch)
    return vecs, keep


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between row vectors of two matrices.
    
    Args:
        a: First matrix (m x d).
        b: Second matrix (n x d).
        
    Returns:
        Similarity matrix (m x n).
    """
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_n @ b_n.T


def main() -> None:
    """Main entry point for diff-comment similarity analysis."""
    ref_paras = read_paragraphs(DATA_DIR / REFERENCE_FILE)
    finals, f_flags = zip(*(truncate(ref_paras[i - 1]) for i in ANALYSIS_RANGE))

    src_texts, src_meta, src_flags = [], [], []
    for sf in SOURCE_FILES:
        stem = Path(sf).stem
        for idx, p in enumerate(read_paragraphs(DATA_DIR / sf), 1):
            t, fl = truncate(p)
            src_texts.append(t)
            src_meta.append((stem, idx))
            src_flags.append(fl)

    para_emb, _ = embed_batches(list(finals) + src_texts, allow_skip=False)
    para_emb = np.array(para_emb)
    finals_emb = para_emb[: len(finals)]
    src_emb = para_emb[len(finals) :]

    sim_mat = cosine_sim(finals_emb, src_emb)
    top_idx = np.argsort(-sim_mat, axis=1)[:, :ANALYSIS_TOP]

    pairs = []
    for i, num in enumerate(ANALYSIS_RANGE):
        for rk in range(ANALYSIS_TOP):
            s_pos = int(top_idx[i, rk])
            stem, s_no = src_meta[s_pos]
            pairs.append(
                {
                    "final_num": num,
                    "match_rank": rk + 1,
                    "final_text": finals[i],
                    "final_trunc": f_flags[i],
                    "source_text": src_texts[s_pos],
                    "source_trunc": src_flags[s_pos],
                    "diff_vec": finals_emb[i] - src_emb[s_pos],
                }
            )

    c_paths: list[str] = []
    c_texts: list[str] = []
    c_flags: list[int] = []
    for cdir in COMMENT_DIRS:
        for p in cdir.rglob("*.txt"):
            raw = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw:
                continue
            t, fl = truncate(raw)
            c_paths.append(str(p))
            c_texts.append(t)
            c_flags.append(fl)

    cm_emb, cm_keep = embed_batches(c_texts, allow_skip=True)
    cm_paths = [p for p, k in zip(c_paths, cm_keep) if k]
    cm_flags = [f for f, k in zip(c_flags, cm_keep) if k]
    cm_emb = np.array(cm_emb)
    cm_norm = cm_emb / np.linalg.norm(cm_emb, axis=1, keepdims=True)

    recs: list[dict] = []
    for pr in tqdm(pairs, desc="Similarity"):
        dv_n = pr["diff_vec"] / np.linalg.norm(pr["diff_vec"])
        sims = dv_n @ cm_norm.T
        mean, std = float(sims.mean()), float(sims.std())
        top_i = int(np.argmax(sims))
        recs.append(
            {
                "final_paragraph_num": pr["final_num"],
                "match_rank": pr["match_rank"],
                "final_paragraph": pr["final_text"],
                "source_paragraph": pr["source_text"],
                "top_comment_path": cm_paths[top_i],
                "top_similarity": float(sims[top_i]),
                "mean_similarity": mean,
                "z_score": (float(sims[top_i]) - mean) / (std if std else 1),
                "truncated_final": pr["final_trunc"],
                "truncated_source": pr["source_trunc"],
                "truncated_comment": cm_flags[top_i],
            }
        )

    skipped = cm_keep.count(0)
    paragraph_comment_similarity_df = pd.DataFrame(recs)
    paragraph_comment_similarity_df.to_excel("outputs/diff_comment_similarity.xlsx", index=False)
    print(
        f"Wrote outputs/diff_comment_similarity.xlsx "
        f"({len(recs)} rows, skipped {skipped} comments)"
    )


if __name__ == "__main__":
    main()
