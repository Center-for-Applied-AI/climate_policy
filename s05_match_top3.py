#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import openai


import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

DATA_DIR = Path("data/policies/parsed")
SOURCE_FILES = ["fdic.md", "occ.md", "fed.md"]
REFERENCE_FILE = "final.md"
TOP_K = 3

PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")


def get_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 512,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Fetch embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed.
        model: OpenAI embedding model name.
        batch_size: Number of texts to process per API call.
        dimensions: Optional dimensionality for embeddings.
        
    Returns:
        List of embedding vectors.
    """
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        params = {"model": model, "input": batch}
        if dimensions is not None:
            params["dimensions"] = dimensions
        response = openai.embeddings.create(**params)
        embeddings.extend([r.embedding for r in response.data])
    return embeddings


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
    text = "\n".join(lines)
    return [p.strip() for p in PARAGRAPH_SPLIT_PATTERN.split(text) if p.strip()]


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between row vectors of two matrices.
    
    Args:
        a: First matrix (m x d).
        b: Second matrix (n x d).
        
    Returns:
        Similarity matrix (m x n).
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def main() -> None:
    """Main entry point for top-3 paragraph matching."""
    ref_paragraphs = read_paragraphs(DATA_DIR / REFERENCE_FILE)
    ref_embs = np.array(get_embeddings(ref_paragraphs))

    combined_src_paragraphs: list[dict[str, str | int]] = []
    combined_embs_list: list[np.ndarray] = []

    for src_name in SOURCE_FILES:
        src_path = DATA_DIR / src_name
        src_stem = src_path.stem

        src_paragraphs = read_paragraphs(src_path)
        src_embs = np.array(get_embeddings(src_paragraphs))

        for idx, (p, emb) in enumerate(zip(src_paragraphs, src_embs), start=1):
            combined_src_paragraphs.append(
                {
                    "source_file": src_stem,
                    "source_par_num": idx,
                    "text": p,
                }
            )
            combined_embs_list.append(emb)

        sim_mat = cosine_similarity_matrix(src_embs, ref_embs)
        topk_idx = np.argsort(-sim_mat, axis=1)[:, :TOP_K]
        topk_scores = np.take_along_axis(sim_mat, topk_idx, axis=1)

        data: dict[str, list] = {
            f"{src_stem}_paragraph_num": np.arange(1, len(src_paragraphs) + 1),
            f"{src_stem}_paragraph": src_paragraphs,
        }
        for k in range(TOP_K):
            data[f"final_paragraph_num_{k+1}"] = topk_idx[:, k] + 1
            data[f"final_paragraph_{k+1}"] = [ref_paragraphs[j] for j in topk_idx[:, k]]
            data[f"cosine_similarity_{k+1}"] = topk_scores[:, k]

        paragraph_match_df = pd.DataFrame(data)
        out_name = f"outputs/{src_stem}_top{TOP_K}_matches_vs_final.xlsx"
        paragraph_match_df.to_excel(out_name, index=False)
        print(f"Wrote {out_name} ({len(paragraph_match_df)} rows)")

    combined_embs = np.array(combined_embs_list)
    sim_ref_to_all = cosine_similarity_matrix(ref_embs, combined_embs)
    topk_idx_rev = np.argsort(-sim_ref_to_all, axis=1)[:, :TOP_K]
    topk_scores_rev = np.take_along_axis(sim_ref_to_all, topk_idx_rev, axis=1)

    records_rev: list[dict[str, str | int | float]] = []
    for j in range(sim_ref_to_all.shape[0]):
        record: dict[str, str | int | float] = {
            "final_paragraph_num": j + 1,
            "final_paragraph": ref_paragraphs[j],
        }
        for k in range(TOP_K):
            idx = topk_idx_rev[j, k]
            score = topk_scores_rev[j, k]
            src_info = combined_src_paragraphs[idx]

            record[f"match{k+1}_source_file"] = src_info["source_file"]
            record[f"match{k+1}_source_paragraph_num"] = src_info["source_par_num"]
            record[f"match{k+1}_source_paragraph"] = src_info["text"]
            record[f"cosine_similarity_{k+1}"] = score
        records_rev.append(record)

    paragraph_reverse_match_df = pd.DataFrame.from_records(records_rev)
    output_path = "outputs/final_top3_matches_from_sources.xlsx"
    paragraph_reverse_match_df.to_excel(output_path, index=False)
    print(f"Wrote {output_path} ({len(paragraph_reverse_match_df)} rows)")

    print("All Excel exports complete.")


if __name__ == "__main__":
    main()
