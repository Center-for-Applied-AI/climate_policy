#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import openai
from tqdm.auto import tqdm
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


DATA_DIR = Path("data/policies/parsed")
AFTER_FILE = "final.md"
BEFORE_FILES = {
    "fdic": "fdic.md",
    "occ": "occ.md",
    "fed": "fed.md",
}
COMMENT_DIRS = [
    Path("data/comments/occ/txt"),
    Path("data/comments/fdic/txt"),
    Path("data/comments/fed/txt"),
]
ANALYSIS_RANGE = range(65, 84)

EMB_MODEL = "text-embedding-3-large"
TEXT_WORD_LIMIT = 80_000
BATCH_WORD_LIMIT = 80_000

EMBEDDINGS_JSON = Path("outputs/comment_embeddings.json")
OUTPUT_XLSX = Path("outputs/before_after_comment_similarity_embeddings.xlsx")

MATCH_ALL_TO_ALL = True

import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")
WORD_PATTERN = re.compile(r"\w+")


def comment_agency(path_str: str) -> str:
    """Infer agency from path name.
    
    Args:
        path_str: Comment file path.
        
    Returns:
        Agency name (occ, fed, or fdic).
    """
    p = path_str.lower()
    if "comments/occ" in p:
        return "occ"
    if "comments/fed" in p:
        return "fed"
    return "fdic"


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
    
    Returns embeddings and keep_flags. Errors per-batch are decomposed recursively
    to singletons; failures are recorded as keep=0.
    
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


def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Remove characters illegal in Excel XML from object columns.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        A copy of the DataFrame with illegal characters removed from all object-dtype columns.
    """
    clean_df = df.copy()
    obj_cols = [c for c in clean_df.columns if pd.api.types.is_object_dtype(clean_df[c])]
    for col in obj_cols:
        clean_df[col] = clean_df[col].map(
            lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
        )
    return clean_df


def main() -> None:
    """Run Equations 2 and 3 similarity labeling and export outputs."""
    payload = None
    if EMBEDDINGS_JSON.exists():
        with EMBEDDINGS_JSON.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

    if payload and "after" in payload:
        par_nums = [int(i) for i in payload["after"].get("paragraph_nums", [])]
        if par_nums == [int(i) for i in ANALYSIS_RANGE]:
            after_texts = list(payload["after"].get("texts", []))
            after_vecs = np.array(payload["after"].get("embeddings", []))
        else:
            after_all = read_paragraphs(DATA_DIR / AFTER_FILE)
            after_texts_all, _ = zip(*(truncate_by_words(p) for p in after_all))
            after_texts = [after_texts_all[i - 1] for i in ANALYSIS_RANGE]
            after_vecs_list, _ = embed_batches(after_texts)
            after_vecs = np.array(after_vecs_list)
    else:
        after_all = read_paragraphs(DATA_DIR / AFTER_FILE)
        after_texts_all, _ = zip(*(truncate_by_words(p) for p in after_all))
        after_texts = [after_texts_all[i - 1] for i in ANALYSIS_RANGE]
        after_vecs_list, _ = embed_batches(after_texts)
        after_vecs = np.array(after_vecs_list)

    comment_paths: list[str] = []
    comment_texts: list[str] = []
    comment_agencies: list[str] = []
    comment_trim_flags: list[int] = []

    for cdir in COMMENT_DIRS:
        for p in tqdm(sorted(cdir.rglob("*.txt")), desc=f"Load {cdir.name}", leave=False):
            raw = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw:
                continue
            text, flag = truncate_by_words(raw)
            comment_paths.append(str(p))
            comment_texts.append(text)
            comment_agencies.append(comment_agency(str(p)))
            comment_trim_flags.append(flag)

    n_comments = len(comment_paths)
    loaded_map = {}
    if payload and "comments" in payload:
        for rec in payload["comments"]:
            loaded_map[str(rec.get("path", ""))] = rec

    comment_keep = [0] * n_comments
    dim_from_after = after_vecs.shape[1] if after_vecs.ndim == 2 else 1
    comment_norm = np.full((n_comments, dim_from_after), np.nan)

    to_embed_idx: list[int] = []
    to_embed_texts: list[str] = []
    for i, path in enumerate(comment_paths):
        rec = loaded_map.get(path)
        if rec and rec.get("embedding") is not None:
            vec = np.array(rec["embedding"], dtype=float)
            comment_norm[i] = vec
            comment_keep[i] = 1
        else:
            to_embed_idx.append(i)
            to_embed_texts.append(comment_texts[i])

    if to_embed_texts:
        new_vecs_list, new_keep = embed_batches(to_embed_texts)

        if new_vecs_list:
            new_mat = np.array(new_vecs_list)
            new_norm = new_mat / np.linalg.norm(new_mat, axis=1, keepdims=True)
            success_positions = [pos for pos, k in enumerate(new_keep) if k]
            for s, pos in enumerate(success_positions):
                idx = to_embed_idx[pos]
                comment_norm[idx] = new_norm[s]
                comment_keep[idx] = 1

    records: list[dict] = []

    before_store: dict[str, dict] = {}
    for agency, before_name in BEFORE_FILES.items():
        if payload and "before" in payload and agency in payload["before"]:
            bobj = payload["before"][agency]
            before_paras = read_paragraphs(DATA_DIR / before_name)
            before_texts = tuple(bobj.get("texts", before_paras))
            before_vecs = np.array(bobj.get("embeddings", []))
            if before_vecs.size == 0:
                before_texts, _ = zip(*(truncate_by_words(p) for p in before_paras))
                before_vecs = np.array(embed_batches(list(before_texts))[0])
        else:
            before_paras = read_paragraphs(DATA_DIR / before_name)
            before_texts, _ = zip(*(truncate_by_words(p) for p in before_paras))
            before_vecs = np.array(embed_batches(list(before_texts))[0])

        sim_ab = cosine_sim_matrix(after_vecs, before_vecs)
        best_before_idx = np.argmax(sim_ab, axis=1)

        before_store[agency] = {
            "texts": list(before_texts),
            "embeddings": before_vecs.tolist(),
            "best_map": {str(par): int(best_before_idx[i]) for i, par in enumerate(ANALYSIS_RANGE)},
        }

        if MATCH_ALL_TO_ALL:
            total_iters = len(ANALYSIS_RANGE) * n_comments
        else:
            total_iters = len(ANALYSIS_RANGE) * sum(1 for a in comment_agencies if a == agency)
        progress = tqdm(total=total_iters, desc=f"Similarity {agency}", leave=False)

        for r, final_par_num in enumerate(ANALYSIS_RANGE):
            after_vec = after_vecs[r]
            b_idx = best_before_idx[r]
            before_vec = before_vecs[b_idx]
            before_text = before_paras[b_idx]
            after_text = after_texts[r]

            av = after_vec / np.linalg.norm(after_vec)
            bv = before_vec / np.linalg.norm(before_vec)

            for c in range(n_comments):
                if not MATCH_ALL_TO_ALL and comment_agencies[c] != agency:
                    continue
                cv = comment_norm[c]
                if np.isnan(cv).any():
                    sim_a = np.nan
                    sim_b = np.nan
                else:
                    sim_a = float(av @ cv)
                    sim_b = float(bv @ cv)
                y = 1 if (sim_a > sim_b) else 0
                records.append(
                    {
                        "agency": agency,
                        "paragraph_num": int(final_par_num),
                        "comment_idx": int(c),
                        "comment_path": comment_paths[c],
                        "sim_after": sim_a,
                        "sim_before": sim_b,
                        "y_after_gt_before": int(y),
                        "after_text": after_text,
                        "before_text": before_text,
                        "comment_text": comment_texts[c] if comment_keep[c] else "N/A",
                    }
                )
                progress.update(1)

        progress.close()

    embeddings_payload = {
        "after": {
            "paragraph_nums": [int(i) for i in ANALYSIS_RANGE],
            "texts": after_texts,
            "embeddings": after_vecs.tolist(),
        },
        "before": before_store,
        "comments": [
            {
                "path": comment_paths[i],
                "agency": comment_agencies[i],
                "kept": int(comment_keep[i]),
                "embedding": None if np.isnan(comment_norm[i]).any() else comment_norm[i].tolist(),
            }
            for i in range(n_comments)
        ],
    }

    with EMBEDDINGS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(embeddings_payload, fh, ensure_ascii=False)

    paragraph_comment_similarity_df = pd.DataFrame.from_records(records)
    safe_df = sanitize_for_excel(paragraph_comment_similarity_df)
    safe_df.to_excel(OUTPUT_XLSX, index=False)

    print(f"Wrote {EMBEDDINGS_JSON.name}")
    print(f"Wrote {OUTPUT_XLSX.name} ({len(paragraph_comment_similarity_df)} rows)")


if __name__ == "__main__":
    main()
