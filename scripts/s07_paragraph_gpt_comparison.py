#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re
from time import sleep

import numpy as np
import pandas as pd
import openai
from tqdm.auto import tqdm


import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

DATA_DIR = Path("data/policies/parsed")
SOURCE_FILES = ["fdic.md", "occ.md", "fed.md"]
REFERENCE_FILE = "final.md"
TOP_K = 10
ANALYSIS_RANGE = range(65, 84)
ANALYSIS_TOP = 3
MODEL_NAME = "gpt-4o-2024-08-06"

PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")
WORD_PATTERN = re.compile(r"\w+")


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
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i : i + batch_size]
        params: dict = {"model": model, "input": batch}
        if dimensions is not None:
            params["dimensions"] = dimensions
        resp = openai.embeddings.create(**params)
        embeddings.extend([r.embedding for r in resp.data])
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
    return [p.strip() for p in PARAGRAPH_SPLIT_PATTERN.split("\n".join(lines)) if p.strip()]


def word_count(text: str) -> int:
    """Return the number of words in text.
    
    Uses Unicode word character pattern.
    
    Args:
        text: Input text string.
        
    Returns:
        Number of words found.
    """
    return len(WORD_PATTERN.findall(text))


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


def analyse_pair(final_para: str, match_para: str) -> dict:
    """Analyze a pair of paragraphs using GPT-4o.
    
    Args:
        final_para: The final policy paragraph.
        match_para: The matched source paragraph.
        
    Returns:
        Dictionary with MoreClimateFriendly, MajorDifference, and Differences.
    """
    prompt = (
        "You will receive two policy paragraphs.\n"
        "Return ONLY a valid JSON object with exactly three keys: \n"
        "  MoreClimateFriendly (1/0/-1),\n"
        "  MajorDifference (1 if substantive policy change, else 0),\n"
        "  Differences (array of [short, detailed] pairs covering ALL linguistic differences).\n\n"
        "[FINAL PARAGRAPH]\n" + final_para + "\n\n[MATCHED PARAGRAPH]\n" + match_para
    )
    for attempt in range(3):
        resp = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    raise RuntimeError("GPT-4o analysis failed after 3 attempts")


def main() -> None:
    """Main entry point for paragraph comparison and GPT-4o analysis."""
    ref_paras = read_paragraphs(DATA_DIR / REFERENCE_FILE)
    ref_counts = [word_count(p) for p in ref_paras]
    ref_embs = np.array(get_embeddings(ref_paras))

    combined_src: list[dict] = []
    combined_embs: list[np.ndarray] = []

    for src in tqdm(SOURCE_FILES, desc="Sources"):
        stem = Path(src).stem
        paras = read_paragraphs(DATA_DIR / src)
        counts = [word_count(p) for p in paras]
        embs = np.array(get_embeddings(paras))
        
        for idx, (p, e, c) in enumerate(zip(paras, embs, counts), 1):
            combined_src.append(
                {"source_file": stem, "source_par_num": idx, "text": p, "word_count": c}
            )
            combined_embs.append(e)

        sim = cosine_similarity_matrix(embs, ref_embs)
        top_idx = np.argsort(-sim, axis=1)[:, :TOP_K]
        top_scores = np.take_along_axis(sim, top_idx, axis=1)
        
        paragraph_match_columns = [
            f"{stem}_paragraph_num",
            f"{stem}_paragraph",
            f"{stem}_paragraph_word_count",
        ]
        paragraph_match_data = {
            paragraph_match_columns[0]: list(range(1, len(paras) + 1)),
            paragraph_match_columns[1]: paras,
            paragraph_match_columns[2]: counts,
        }
        
        for k in range(TOP_K):
            paragraph_match_data[f"final_paragraph_num_{k+1}"] = top_idx[:, k] + 1
            paragraph_match_data[f"final_paragraph_{k+1}"] = [ref_paras[j] for j in top_idx[:, k]]
            paragraph_match_data[f"final_paragraph_word_count_{k+1}"] = [ref_counts[j] for j in top_idx[:, k]]
            paragraph_match_data[f"cosine_similarity_{k+1}"] = top_scores[:, k]
        
        paragraph_match_df = pd.DataFrame(paragraph_match_data)
        paragraph_match_df.to_excel(f"outputs/{stem}_top{TOP_K}_matches_vs_final.xlsx", index=False)

    sim_rev = cosine_similarity_matrix(ref_embs, np.array(combined_embs))
    rev_idx = np.argsort(-sim_rev, axis=1)[:, :TOP_K]

    pair_records: list[dict] = []
    diff_rows: list[dict] = []
    max_diff_len = 0

    for fn in tqdm(ANALYSIS_RANGE, desc="GPT-4o analysis"):
        i_final = fn - 1
        final_p = ref_paras[i_final]
        for rank in range(ANALYSIS_TOP):
            src_pos = rev_idx[i_final, rank]
            src_info = combined_src[src_pos]
            res = analyse_pair(final_p, src_info["text"])
            diffs = res.get("Differences", [])
            max_diff_len = max(max_diff_len, len(diffs))

            pair_rec = {
                "final_paragraph_num": fn,
                "final_paragraph": final_p,
                "match_rank": rank + 1,
                "source_file": src_info["source_file"],
                "source_paragraph_num": src_info["source_par_num"],
                "source_paragraph": src_info["text"],
                "MoreClimateFriendly": res.get("MoreClimateFriendly"),
                "MajorDifference": res.get("MajorDifference"),
            }
            for d_i, pair in enumerate(diffs, 1):
                if isinstance(pair, list) and len(pair) == 2:
                    short, detail = pair
                    pair_rec[f"difference_{d_i}"] = short
                    pair_rec[f"difference_explanation_{d_i}"] = detail
                    diff_rows.append(
                        {
                            "final_paragraph_num": fn,
                            "final_paragraph": final_p,
                            "match_rank": rank + 1,
                            "source_file": src_info["source_file"],
                            "source_paragraph_num": src_info["source_par_num"],
                            "source_paragraph": src_info["text"],
                            "difference_num": d_i,
                            "difference": short,
                            "difference_explanation": detail,
                            "MajorDifference": res.get("MajorDifference"),
                            "MoreClimateFriendly": res.get("MoreClimateFriendly"),
                        }
                    )
            pair_records.append(pair_rec)

    base_cols = [
        "final_paragraph_num",
        "final_paragraph",
        "match_rank",
        "source_file",
        "source_paragraph_num",
        "source_paragraph",
        "MoreClimateFriendly",
        "MajorDifference",
    ]
    diff_cols = []
    for d in range(1, max_diff_len + 1):
        diff_cols.extend([f"difference_{d}", f"difference_explanation_{d}"])

    paragraph_pair_df = pd.DataFrame(pair_records)
    for col in base_cols + diff_cols:
        if col not in paragraph_pair_df.columns:
            paragraph_pair_df[col] = ""
    paragraph_pair_df = paragraph_pair_df[base_cols + diff_cols]
    paragraph_pair_df.to_excel("outputs/side_by_side_policy_paragraphs.xlsx", index=False)

    paragraph_diff_df = pd.DataFrame(diff_rows)
    paragraph_diff_df.to_excel("outputs/side_by_side_differences.xlsx", index=False)

    print("Generated side_by_side_policy_paragraphs.xlsx and side_by_side_differences.xlsx")
    print("All tasks complete.")


if __name__ == "__main__":
    main()
