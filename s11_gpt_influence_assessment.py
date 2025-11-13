#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import threading

import instructor
import openai
import pandas as pd
import tiktoken
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

CLIENT = instructor.patch(openai.OpenAI(api_key=openai.api_key))

MODEL_NAME = "gpt-4o-2024-08-06"
SEED = 42
MAX_WORKERS = 10

SCORES_FILE = Path("outputs/comment_climate_engagement_scores.xlsx")
OUTPUT_EXCEL_ALL = Path("outputs/before_after_comment_similarity_ALL.xlsx")
OUTPUT_JSON = Path("outputs/influence_results.json")
CACHE_JSON = Path("outputs/influence_cache.json")

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
MATCH_ALL_TO_ALL = True

PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")
WORD_PATTERN = re.compile(r"\w+")


class InfluenceAssessment(BaseModel):
    """Structured output for influence assessment."""

    influenced: str = Field(..., description="Either 'yes' or 'no'.")
    explanation: str = Field(..., description="Short explanation for the decision.")


def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Remove characters illegal in Excel XML from object columns.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Sanitized dataframe.
    """
    clean_df = df.copy()
    obj_cols = [c for c in clean_df.columns if pd.api.types.is_object_dtype(clean_df[c])]
    for col in obj_cols:
        clean_df[col] = clean_df[col].map(
            lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
        )
    return clean_df


def comment_agency(path_str: str) -> str:
    """Infer agency from path name.
    
    Args:
        path_str: Path string.
        
    Returns:
        Agency name (occ, fed, fdic).
    """
    p = str(path_str).lower()
    if "comments/occ" in p:
        return "occ"
    if "comments/fed" in p:
        return "fed"
    if "comments/fdic" in p:
        return "fdic"
    return "fdic"


def word_len(text: str) -> int:
    """Return word count using Unicode word character pattern.
    
    Args:
        text: Input text.
        
    Returns:
        Number of words.
    """
    return len(WORD_PATTERN.findall(text))


def truncate_by_words(text: str) -> tuple[str, int]:
    """Truncate text to word limit.
    
    Args:
        text: Input text.
        
    Returns:
        Tuple of (truncated_text, truncation_flag).
    """
    words = WORD_PATTERN.findall(text)
    if len(words) <= TEXT_WORD_LIMIT:
        return text, 0
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


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts.
    
    Args:
        texts: List of text strings.
        
    Returns:
        List of embedding vectors.
    """
    resp = openai.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_batches(texts: list[str]) -> tuple[list[list[float]], list[int]]:
    """Embed texts in batches with error handling.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        Tuple of (embeddings, keep_flags).
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


def build_prompt(before: str, after: str, comment: str) -> str:
    """Construct the assessment prompt.
    
    Args:
        before: Before paragraph text.
        after: After paragraph text.
        comment: Comment text.
        
    Returns:
        Formatted prompt string.
    """
    return (
        "You will receive a BEFORE paragraph from an agency draft, the AFTER paragraph from the final rule, "
        "and a PUBLIC COMMENT.\n\n"
        "Task: Decide whether the comment plausibly influenced the change from BEFORE to AFTER.\n"
        'If **not**, respond JSON {"influenced": "no", "explanation": ""}.\n'
        'If **yes**, respond JSON {"influenced": "yes", "explanation": "<why>"}.\n'
        "Return only valid JSON.\n\n"
        f"[BEFORE]\n{before}\n\n[AFTER]\n{after}\n\n[COMMENT]\n{comment}"
    )


def make_row_key(row: pd.Series) -> str:
    """Create a unique key for caching per agency×paragraph×comment.
    
    Args:
        row: DataFrame row.
        
    Returns:
        Unique key string.
    """
    return "|".join(
        [
            str(row.get("agency", "")),
            str(int(row.get("paragraph_num", -1)) if pd.notna(row.get("paragraph_num")) else -1),
            str(int(row.get("comment_idx", -1)) if pd.notna(row.get("comment_idx")) else -1),
            str(row.get("comment_path", "")),
        ]
    )


def main(plan_only: bool = False, limit: int | None = None, workers: int | None = None) -> None:
    """Run influence assessment over generated comment–paragraph pairs.
    
    Args:
        plan_only: If True, only compute and print planned token count.
        limit: Limit the number of uncached tasks processed.
        workers: Number of parallel workers for LLM calls.
    """
    if workers is None or workers <= 0:
        workers = MAX_WORKERS

    after_all = read_paragraphs(DATA_DIR / AFTER_FILE)
    after_texts_all, _ = zip(*(truncate_by_words(p) for p in after_all))
    after_texts = [after_texts_all[i - 1] for i in ANALYSIS_RANGE]
    after_vecs_list, _ = embed_batches(after_texts)
    after_vecs = np.array(after_vecs_list)

    comment_paths: list[str] = []
    comment_texts: list[str] = []
    comment_agencies: list[str] = []
    for cdir in COMMENT_DIRS:
        for p in tqdm(sorted(cdir.rglob("*.txt")), desc=f"Load {cdir.name}", leave=False):
            raw = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw:
                continue
            text, _ = truncate_by_words(raw)
            comment_paths.append(str(p))
            comment_texts.append(text)
            comment_agencies.append(comment_agency(str(p)))
    n_comments = len(comment_paths)

    records: list[dict] = []
    for agency, before_name in BEFORE_FILES.items():
        before_paras = read_paragraphs(DATA_DIR / before_name)
        before_texts, _ = zip(*(truncate_by_words(p) for p in before_paras))
        before_vecs = np.array(embed_batches(list(before_texts))[0])
        sim_ab = cosine_sim_matrix(after_vecs, before_vecs)
        best_before_idx = np.argmax(sim_ab, axis=1)

        progress_total = len(ANALYSIS_RANGE) * (
            n_comments if MATCH_ALL_TO_ALL else sum(1 for a in comment_agencies if a == agency)
        )
        progress = tqdm(total=progress_total, desc=f"Build pairs {agency}", leave=False)

        for r, final_par_num in enumerate(ANALYSIS_RANGE):
            b_idx = int(best_before_idx[r])
            before_text = before_paras[b_idx]
            after_text = after_texts[r]

            for c in range(n_comments):
                if not MATCH_ALL_TO_ALL and comment_agencies[c] != agency:
                    continue
                records.append(
                    {
                        "agency": agency,
                        "paragraph_num": int(final_par_num),
                        "comment_idx": int(c),
                        "comment_path": comment_paths[c],
                        "after_text": after_text,
                        "before_text": before_text,
                        "comment_text": comment_texts[c],
                    }
                )
                progress.update(1)
        progress.close()

    paragraph_comment_pair_df = pd.DataFrame.from_records(records)

    comment_score_df = pd.read_excel(SCORES_FILE)

    scores_keep_cols = [
        "comment_path",
        "score",
        "author_type",
        "author_organization_type",
    ]
    for col in scores_keep_cols:
        if col not in comment_score_df.columns:
            comment_score_df[col] = None

    paragraph_comment_merged_df = paragraph_comment_pair_df.merge(
        comment_score_df[scores_keep_cols], on="comment_path", how="left", suffixes=("", "_score")
    )

    cache: dict[str, dict[str, str]]
    if CACHE_JSON.exists():
        cache = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    else:
        cache = {}

    tasks: list[dict[str, str]] = []

    enc = tiktoken.get_encoding("o200k_base")
    total_tokens_all: int = 0
    total_tokens_paragraphs: int = 0
    total_tokens_comments: int = 0
    total_tokens_prompt_rest: int = 0

    for _, row in paragraph_comment_merged_df.iterrows():
        key = make_row_key(row)
        if key in cache:
            continue
        before_txt = str(row.get("before_text", ""))
        after_txt = str(row.get("after_text", ""))
        comment_txt = str(row.get("comment_text", ""))
        prompt = build_prompt(before_txt, after_txt, comment_txt)
        tasks.append({"key": key, "prompt": prompt})

        t_total = len(enc.encode(prompt))
        t_before = len(enc.encode(before_txt))
        t_after = len(enc.encode(after_txt))
        t_comment = len(enc.encode(comment_txt))
        t_pars = t_before + t_after
        t_rest = t_total - t_pars - t_comment
        total_tokens_all += t_total
        total_tokens_paragraphs += t_pars
        total_tokens_comments += t_comment
        total_tokens_prompt_rest += t_rest

    print(f"Planned submission prompts: {len(tasks)}")
    print("Planned tokens (approx):")
    print(f"- Prompt (instructions + labels): {total_tokens_prompt_rest}")
    print(f"- Paragraphs (BEFORE + AFTER): {total_tokens_paragraphs}")
    print(f"- Comments: {total_tokens_comments}")
    print(f"Total: {total_tokens_all}")

    if limit is not None and limit >= 0:
        tasks = tasks[:limit]
        print(f"Limiting to first {len(tasks)} tasks for this run")

    if plan_only:
        print("Plan-only mode: exiting before any LLM calls.")
        return

    results: dict[str, dict[str, str]] = {}
    results.update(cache)

    def call_llm(prompt: str) -> dict[str, str]:
        resp = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_model=InfluenceAssessment,
            seed=SEED,
        )
        out = resp.model_dump()
        return {
            "influenced": str(out.get("influenced", "no")),
            "explanation": str(out.get("explanation", "")),
        }

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_llm, job["prompt"]): job["key"] for job in tasks}
        with tqdm(total=len(futures), desc=f"Assess influence (workers={workers})") as bar:
            for fut in as_completed(futures):
                key = futures[fut]
                out = fut.result()
                with lock:
                    results[key] = out
                    if len(results) % 500 == 0:
                        Path(CACHE_JSON).write_text(
                            json.dumps(results, ensure_ascii=False), encoding="utf-8"
                        )
                bar.update(1)

    Path(CACHE_JSON).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    influenced_col: list[str] = []
    explanation_col: list[str] = []
    for _, row in paragraph_comment_merged_df.iterrows():
        key = make_row_key(row)
        rec = results.get(key, {"influenced": "no", "explanation": "missing"})
        influenced_col.append(rec.get("influenced", "no"))
        explanation_col.append(rec.get("explanation", ""))

    paragraph_comment_influence_df = paragraph_comment_merged_df.assign(
        gpt_influenced=influenced_col,
        gpt_explanation=explanation_col,
    )

    Path(OUTPUT_JSON).write_text(
        json.dumps(
            [
                {
                    **{k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()},
                }
                for row in paragraph_comment_influence_df.to_dict(orient="records")
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    safe_df = sanitize_for_excel(paragraph_comment_influence_df)
    with pd.ExcelWriter(OUTPUT_EXCEL_ALL) as writer:
        safe_df.to_excel(writer, index=False)

    print(f"Saved cache to {CACHE_JSON}")
    print(f"Saved results JSON to {OUTPUT_JSON}")
    print(f"Saved Excel to {OUTPUT_EXCEL_ALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM influence scoring.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only compute and print planned token count; do not call the LLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of uncached tasks processed in this run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for LLM calls (default 10)",
    )
    args = parser.parse_args()
    main(plan_only=args.plan_only, limit=args.limit, workers=args.workers)
