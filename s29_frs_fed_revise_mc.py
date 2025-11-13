#!/usr/bin/env python3
"""Run Monte Carlo role-conditioned revisions and similarity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple
import argparse
import json
import hashlib
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openai
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field, constr
from tqdm import tqdm
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

GLOBAL_RANDOM_SEED: int = 42
MODEL_NAME: str = "gpt-5"
WORD_LIMIT: int = 2000

NUM_SEEDS_DEFAULT: int = 100
SEED_START_DEFAULT: int = 1

CACHE_DIR_DEFAULT: Path = Path("outputs/s29_cache")
TRACE_XLSX_PATH: Path = Path("outputs/s29_role_trace_all_agencies_all_roles.xlsx")
SIM_XLSX_PATH: Path = Path("outputs/s29_role_similarity_all_agencies_all_roles.xlsx")
PROMPT_TEMPLATE_PATH: Path = Path("prompt_policy_proposal_v2.txt")

EMB_MODEL: str = "text-embedding-3-large"
BATCH_SIZE_DEFAULT: int = 10

SIM_MATRIX_AVG_CSV: Path = Path("outputs/s29_policy_similarity_matrix_avg.csv")
HEATMAP_AVG_PNG: Path = Path("plots/s29_policy_similarity_heatmap_avg.png")
HEATMAP_AVG_SVG: Path = Path("plots/s29_policy_similarity_heatmap_avg.svg")
RIDGE_PNG_PATH: Path = Path("plots/s29_similarity_ridge_official_final.png")
RIDGE_SVG_PATH: Path = Path("plots/s29_similarity_ridge_official_final.svg")

ORIGINAL_FINAL_PATH: Path = Path("data/policies/final.txt")
AGENCY_TO_DRAFT: Dict[str, Path] = {
    "fed": Path("data/policies/drafts/fed.txt"),
    "occ": Path("data/policies/drafts/occ.txt"),
    "fdic": Path("data/policies/drafts/fdic.txt"),
}

COMMENT_DIRS: List[Path] = [
    Path("data/comments/occ/txt"),
    Path("data/comments/fdic/txt"),
    Path("data/comments/fed/txt"),
]

ROLES: Dict[str, str] = {
    "monetary": (
        "to follow your mandate to maintain long run growth of the monetary and credit "
        "aggregates commensurate with the economy’s long run potential to increase production, "
        "so as to promote effectively the goals of maximum employment, stable prices, and "
        "moderate long-term interest rates."
    ),
    "banking": (
        "to ensure the safety and soundness of the banking system, and to make sure that the "
        "institutions you regulate and supervise understand and can effectively manage the "
        "material risks they face, including, climate-related financial risks."
    ),
    "bureaucrat": (
        "that of a risk-averse bureaucrat who avoids risks that could threaten personal career security."
    ),
    "nonpartisan": "that of a regulator who wants to appear non-partisan.",
    "american": "to advance the economic interests of the average American.",
    "democratic": "to align policy with Democratic party preferences.",
    "republican": "to align policy with Republican party preferences.",
    "wealth": "to protect the wealth of the top 1 percent of Americans.",
    "worldwide": "to promote the welfare of the average person worldwide.",
    "openai": "to act in accordance with the training you received from OpenAI.",
}

RPM_LIMIT: int = 15000
TPM_LIMIT: int = 40_000_000
OUTPUT_TOKENS_BUDGET_PER_CALL: int = 3000

_WORD_LIMIT_PATTERN = rf"^\s*$|^\S+(?:\s+\S+){{0,{WORD_LIMIT - 1}}}$"
WordLimitedStr = constr(pattern=_WORD_LIMIT_PATTERN)

class BatchRevisionOutput(BaseModel):
    """Structured output for revising a full policy with a batch of comments."""

    revised_policy: str = Field(
        ..., description="Revised policy text under 2,000 words."
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

_WORD_RE = __import__("re").compile(r"\w+")

def word_count(text: str) -> int:
    """Compute the number of word tokens in a string."""

    return len(_WORD_RE.findall(text))

def comment_agency_from_path(p: Path) -> str:
    """Infer the agency from a comment file path name."""

    s = str(p).lower()
    if "comments/occ" in s:
        return "occ"
    if "comments/fed" in s:
        return "fed"
    if "comments/fdic" in s:
        return "fdic"
    return "fdic"

def load_all_comments(dirs: List[Path]) -> List[Dict[str, str]]:
    """Load comments from multiple directories as a list of dicts with id and text."""

    items: List[Dict[str, str]] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.txt")):
            raw = read_text_file(p)
            if not raw:
                continue
            agency = comment_agency_from_path(p)
            items.append({
                "id": f"{agency}:{p.stem}",
                "text": raw,
            })
    return items

def iter_comment_batches(
    comments: List[Dict[str, str]], batch_size: int, seed: int
) -> Iterator[List[Dict[str, str]]]:
    """Yield shuffled batches of comments of a fixed size."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.RandomState(seed)
    order = np.arange(len(comments))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield [comments[i] for i in idx]

def build_batch_messages(
    template_text: str,
    role_text: str,
    policy_text: str,
    batch_comments: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Construct chat messages for the batch revision prompt."""

    comments_block_parts: List[str] = []
    for cm in batch_comments:
        comments_block_parts.append(
            "ID: "
            + str(cm["id"]).strip()
            + "\n"
            + str(cm["text"]).strip()
        )
    comments_block = "\n\n".join(comments_block_parts)
    prompt = template_text.format(
        role=role_text,
        policy=policy_text,
        comments=comments_block,
        words=word_count(policy_text),
    )
    return [
        {"role": "user", "content": prompt},
    ]

def compute_batch_cache_key(
    *,
    agency: str,
    role_name: str,
    model: str,
    prompt_text: str,
    seed: int,
) -> str:
    """Build a deterministic cache key including seed and full prompt content."""

    sha = hashlib.sha256()
    sha.update((agency + "|" + role_name + "|" + model + "|" + str(seed) + "|").encode("utf-8"))
    sha.update(prompt_text.encode("utf-8", errors="ignore"))
    return sha.hexdigest()

class RateLimiter:
    """Thread-safe rolling window rate limiter for RPM and TPM."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = int(rpm)
        self.tpm = int(tpm)
        self.request_times: deque = deque()
        self.token_events: deque = deque()
        self.tokens_in_window: int = 0
        self.lock = threading.Lock()

    def _prune(self, now: float) -> None:
        one_min_ago = now - 60.0
        while self.request_times and self.request_times[0] < one_min_ago:
            self.request_times.popleft()
        while self.token_events and self.token_events[0][0] < one_min_ago:
            ts, toks = self.token_events.popleft()
            self.tokens_in_window -= toks

    def acquire(self, tokens: int) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                if len(self.request_times) < self.rpm and (self.tokens_in_window + tokens) <= self.tpm:
                    self.request_times.append(now)
                    self.token_events.append((now, int(tokens)))
                    self.tokens_in_window += int(tokens)
                    return
                wait_req = 0.0
                if self.request_times:
                    wait_req = max(0.0, 60.0 - (now - self.request_times[0]))
                wait_tok = 0.0
                if self.token_events:
                    wait_tok = max(0.0, 60.0 - (now - self.token_events[0][0]))
                sleep_for = max(wait_req, wait_tok, 0.01)
            time.sleep(min(sleep_for, 1.0))

def estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate token count for a list of chat messages using o200k_base."""

    enc = tiktoken.get_encoding("o200k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(str(msg.get("content", ""))))
    return total

def embed_texts(
    texts: List[str], limiter: RateLimiter, model: str = EMB_MODEL
) -> List[Optional[List[float]]]:
    """Compute embeddings with rate limiting; logs and returns None on failure."""

    vectors: List[Optional[List[float]]] = []
    enc = tiktoken.get_encoding("o200k_base")
    for t in texts:
        try:
            toks = len(enc.encode(t)) + 1
            limiter.acquire(toks)
            resp = openai.embeddings.create(model=model, input=[t])
            vectors.append(resp.data[0].embedding)
        except Exception as e:
            print(f"Embedding failed: {e}")
            vectors.append(None)
    return vectors

def cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> Optional[float]:
    """Compute cosine similarity; return None if any vector is None or degenerate."""

    if vec_a is None or vec_b is None:
        return None
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float((a / na) @ (b / nb))

def plan_tokens_for_batches(
    template_text: str,
    role_text: str,
    initial_policy: str,
    batches: List[List[Dict[str, str]]],
    agency: str,
    role_name: str,
    seed: int,
) -> int:
    """Approximate total tokens for all batch prompts given seed."""

    enc = tiktoken.get_encoding("o200k_base")
    planned_total = 0
    policy_cursor = initial_policy
    for batch in batches:
        messages = build_batch_messages(
            template_text=template_text,
            role_text=role_text,
            policy_text=policy_cursor,
            batch_comments=batch,
        )
        prompt = messages[0]["content"]
        key = compute_batch_cache_key(
            agency=agency,
            role_name=role_name,
            model=MODEL_NAME,
            prompt_text=prompt,
            seed=seed,
        )
        planned_total += len(enc.encode(prompt))
        policy_cursor = policy_cursor
    return planned_total

def ensure_inputs_exist(agency: str) -> Tuple[Path, Path]:
    """Resolve and validate input draft and final policy paths for an agency."""

    draft_path = AGENCY_TO_DRAFT.get(agency)
    if draft_path is None:
        raise ValueError(f"Unsupported agency: {agency}")
    if not draft_path.exists():
        raise FileNotFoundError(
            f"Draft policy not found for agency {agency}: {draft_path}"
        )
    if not ORIGINAL_FINAL_PATH.exists():
        raise FileNotFoundError(f"Final policy file not found: {ORIGINAL_FINAL_PATH}")
    return draft_path, ORIGINAL_FINAL_PATH

def build_batches(
    comments: List[Dict[str, str]],
    batch_size: int,
    comment_limit: Optional[int],
    seed: int,
) -> List[List[Dict[str, str]]]:
    """Prepare shuffled comment batches with optional global limit."""

    subset = comments[: (comment_limit if comment_limit is not None else len(comments))]
    batches = list(iter_comment_batches(subset, batch_size=batch_size, seed=seed))
    return batches

def load_prompt_template(path: Path) -> str:
    """Load the batch prompt template from a file."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return read_text_file(path)

def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Drop characters illegal for Excel XML storage from object columns."""

    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    clean_df = df.copy()
    obj_cols = [c for c in clean_df.columns if pd.api.types.is_object_dtype(clean_df[c])]
    for col in obj_cols:
        clean_df[col] = clean_df[col].map(
            lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
        )
    return clean_df

def write_trace_output(output_path: Path, trace_df: pd.DataFrame) -> None:
    """Write the batch revision trace to an Excel file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        sanitize_for_excel(trace_df).to_excel(writer, index=False, sheet_name="trace")

def write_similarity_output(output_path: Path, similarity_df: pd.DataFrame) -> None:
    """Write the cosine similarity results to an Excel file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        similarity_df.to_excel(writer, index=False, sheet_name="similarities")

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Monte Carlo role-based revision and analysis."""

    parser = argparse.ArgumentParser(
        description="Monte Carlo role revisions."
    )
    parser.add_argument(
        "--agency",
        type=str,
        choices=["fed"],
        default="fed",
        help="Agency whose draft to revise (currently limited to ‘fed’).",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="all",
        help=(
            "Comma-separated list of roles to run (e.g., ‘monetary,banking’). Use ‘all’ to run every role."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Number of comments per batch.",
    )
    parser.add_argument(
        "--comment-limit",
        type=int,
        default=None,
        help="Optional cap on total comments used (after shuffling).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of batches processed per run.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=NUM_SEEDS_DEFAULT,
        help="Number of Monte Carlo seeds to run (seeds are 1..num_seeds).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=SEED_START_DEFAULT,
        help="Starting seed index (default 1).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only compute and print planned token count; do not call the LLM.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(CACHE_DIR_DEFAULT),
        help="Directory for JSON caches keyed by seed.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum parallel workers across role-seed runs (1 = sequential).",
    )
    return parser.parse_args()

def main() -> None:
    """Entry point for Monte Carlo role-based revision and averaged heatmap generation."""

    np.random.seed(GLOBAL_RANDOM_SEED)

    args = parse_args()
    agency: str = args.agency
    roles_arg: str = args.roles
    batch_size: int = int(args.batch_size)
    comment_limit: Optional[int] = args.comment_limit
    max_batches: Optional[int] = args.max_batches
    num_seeds: int = max(1, int(args.num_seeds))
    seed_start: int = int(args.seed_start)
    plan_only: bool = bool(args.plan_only)
    cache_dir: Path = Path(args.cache_dir)
    max_workers: int = max(1, int(args.max_workers))

    draft_path, final_path = ensure_inputs_exist(agency)
    original_draft = read_text_file(draft_path)
    original_final = read_text_file(final_path)
    occ_draft_path = AGENCY_TO_DRAFT["occ"]
    fdic_draft_path = AGENCY_TO_DRAFT["fdic"]
    if not occ_draft_path.exists():
        raise FileNotFoundError(f"OCC draft policy not found: {occ_draft_path}")
    if not fdic_draft_path.exists():
        raise FileNotFoundError(f"FDIC draft policy not found: {fdic_draft_path}")
    occ_draft_text = read_text_file(occ_draft_path)
    fdic_draft_text = read_text_file(fdic_draft_path)
    if roles_arg.strip().lower() == "all":
        roles_to_run: List[str] = list(ROLES.keys())
    else:
        roles_to_run = [r.strip() for r in roles_arg.split(",") if r.strip()]
        unknown = [r for r in roles_to_run if r not in ROLES]
        if unknown:
            raise ValueError(f"Unknown roles: {unknown}. Valid roles: {sorted(ROLES.keys())}")

    comments_all = load_all_comments(COMMENT_DIRS)
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)

    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds_list: List[int] = list(range(seed_start, seed_start + num_seeds))

    if plan_only:
        total_tokens = 0
        for seed_val in seeds_list:
            for role_name in roles_to_run:
                role_text = ROLES[role_name]
                batches_all = build_batches(
                    comments=comments_all,
                    batch_size=batch_size,
                    comment_limit=comment_limit,
                    seed=seed_val,
                )
                if max_batches is not None:
                    batches_all = batches_all[: max(0, max_batches)]
                planned_tokens = plan_tokens_for_batches(
                    template_text=prompt_template,
                    role_text=role_text,
                    initial_policy=original_draft,
                    batches=batches_all,
                    agency=agency,
                    role_name=role_name,
                    seed=seed_val,
                )
                total_tokens += planned_tokens
        print(f"Total planned prompt tokens (approx): {total_tokens}")
        print("Plan-only mode: exiting before any LLM calls.")
        return

    client = configure_openai_client()
    limiter = RateLimiter(rpm=RPM_LIMIT, tpm=TPM_LIMIT)

    ref_vecs = embed_texts(
        [original_final, original_draft, occ_draft_text, fdic_draft_text], limiter
    )
    ref_vec_true_final = ref_vecs[0]
    ref_vec_orig_draft = ref_vecs[1]
    ref_vec_occ_draft = ref_vecs[2]
    ref_vec_fdic_draft = ref_vecs[3]

    all_trace_records: List[Dict[str, object]] = []
    all_similarity_records: List[Dict[str, object]] = []

    io_lock = threading.Lock()
    seed_locks: Dict[int, threading.Lock] = {s: threading.Lock() for s in seeds_list}

    seed_cache_map: Dict[int, Dict[str, Dict[str, object]]] = {}
    for s in seeds_list:
        seed_cache_path = cache_dir / f"seed_{s}.json"
        if seed_cache_path.exists():
            seed_cache_map[s] = json.loads(seed_cache_path.read_text(encoding="utf-8"))
        else:
            seed_cache_map[s] = {}

    labels: List[str] = [
        "official_final",
        "draft_fed",
        "draft_occ",
        "draft_fdic",
    ] + [f"final_{rn}" for rn in roles_to_run]
    n_labels = len(labels)
    sum_matrix = np.zeros((n_labels, n_labels), dtype=float)
    count_matrix = np.zeros((n_labels, n_labels), dtype=int)

    role_index_map: Dict[str, int] = {labels[i]: i for i in range(n_labels)}

    def run_role_seed(role_name: str, seed_val: int) -> Tuple[List[Dict[str, object]], Dict[str, object], Optional[List[float]]]:
        role_text_local = ROLES[role_name]
        batches_all_local = build_batches(
            comments=comments_all,
            batch_size=batch_size,
            comment_limit=comment_limit,
            seed=seed_val,
        )
        if max_batches is not None:
            batches_all_local = batches_all_local[: max(0, max_batches)]

        trace_records_local: List[Dict[str, object]] = []
        policy_current_local = original_draft

        for batch_index, batch in enumerate(
            tqdm(batches_all_local, desc=f"Batches [{role_name} seed={seed_val}]"),
            start=1,
        ):
            messages = build_batch_messages(
                template_text=prompt_template,
                role_text=role_text_local,
                policy_text=policy_current_local,
                batch_comments=batch,
            )
            batch_comments_word_count = sum(word_count(str(c.get("text", ""))) for c in batch)
            prompt_text = messages[0]["content"]
            cache_key = compute_batch_cache_key(
                agency=agency,
                role_name=role_name,
                model=MODEL_NAME,
                prompt_text=prompt_text,
                seed=seed_val,
            )
            revised_policy: str
            with seed_locks[seed_val]:
                cached_obj = seed_cache_map[seed_val].get(cache_key)
            if cached_obj is not None:
                revised_policy = str(cached_obj.get("revised_policy", ""))
            else:
                prompt_tokens = estimate_prompt_tokens(messages) + OUTPUT_TOKENS_BUDGET_PER_CALL
                limiter.acquire(prompt_tokens)
                parsed = client.responses.parse(
                    model=MODEL_NAME,
                    input=messages,
                    text_format=BatchRevisionOutput,
                )
                result = parsed.output_parsed
                revised_policy = result.revised_policy
                with seed_locks[seed_val]:
                    seed_cache_map[seed_val][cache_key] = {
                        "revised_policy": revised_policy,
                        "agency": agency,
                        "role": role_name,
                        "batch_num": batch_index,
                        "prompt_sha": cache_key,
                        "seed": seed_val,
                    }
                    seed_cache_path = cache_dir / f"seed_{seed_val}.json"
                    seed_cache_path.write_text(
                        json.dumps(seed_cache_map[seed_val], ensure_ascii=False),
                        encoding="utf-8",
                    )

            n_words = word_count(revised_policy)
            print(f"[{role_name} seed={seed_val}] Batch {batch_index}: {n_words} words")

            step_vec = embed_texts([revised_policy], limiter)[0]
            step_sim_true_final = cosine_similarity(step_vec, ref_vec_true_final)
            step_sim_orig_draft = cosine_similarity(step_vec, ref_vec_orig_draft)

            record = {
                "agency": agency,
                "role": role_name,
                "seed": int(seed_val),
                "role_seed": f"{role_name}_{seed_val}",
                "batch_num": int(batch_index),
                "comment_ids": ",".join([str(c["id"]) for c in batch]),
                "batch_comments_words": int(batch_comments_word_count),
                "n_words": int(n_words),
                "similarity_to_official_final": float(step_sim_true_final)
                if step_sim_true_final is not None
                else None,
                "similarity_to_original_draft": float(step_sim_orig_draft)
                if step_sim_orig_draft is not None
                else None,
                "revised_policy": revised_policy,
            }
            trace_records_local.append(record)
            with io_lock:
                all_trace_records.append(record)
                tmp_df = pd.DataFrame.from_records(all_trace_records)
                write_trace_output(TRACE_XLSX_PATH, tmp_df)
            policy_current_local = revised_policy

        final_policy_local = policy_current_local
        final_vec = embed_texts([final_policy_local], limiter)[0]
        sim_final_vs_true_final = cosine_similarity(final_vec, ref_vec_true_final)
        sim_final_vs_original_draft = cosine_similarity(final_vec, ref_vec_orig_draft)

        sim_record_local: Dict[str, object] = {
            "agency": agency,
            "role": role_name,
            "seed": int(seed_val),
            "final_words": int(word_count(final_policy_local)),
            "similarity_to_official_final": float(sim_final_vs_true_final)
            if sim_final_vs_true_final is not None
            else None,
            "similarity_to_original_draft": float(sim_final_vs_original_draft)
            if sim_final_vs_original_draft is not None
            else None,
        }

        return trace_records_local, sim_record_local, final_vec

    tasks: List[Tuple[str, int]] = []
    for seed_val in seeds_list:
        for role_name in roles_to_run:
            tasks.append((role_name, seed_val))

    futures = []
    results: List[Tuple[List[Dict[str, object]], Dict[str, object], Optional[List[float]], str, int]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for role_name, seed_val in tasks:
            futures.append(executor.submit(run_role_seed, role_name, seed_val))
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Role-seed runs"):
            trace_records_local, sim_record_local, final_vec = fut.result()
            with io_lock:
                all_similarity_records.append(sim_record_local)
            results.append((trace_records_local, sim_record_local, final_vec, sim_record_local["role"], sim_record_local["seed"]))

    all_trace_df = pd.DataFrame.from_records(all_trace_records)
    all_sim_df = pd.DataFrame.from_records(all_similarity_records)

    base_sim_record = {
        "agency": agency,
        "role": "original_draft",
        "seed": None,
        "final_words": int(word_count(original_draft)),
        "similarity_to_official_final": (
            float(cosine_similarity(ref_vec_orig_draft, ref_vec_true_final))
            if cosine_similarity(ref_vec_orig_draft, ref_vec_true_final) is not None
            else None
        ),
        "similarity_to_original_draft": float(1.0),
    }
    occ_sim_record = {
        "agency": "occ",
        "role": "original_draft",
        "seed": None,
        "final_words": int(word_count(occ_draft_text)),
        "similarity_to_official_final": (
            float(cosine_similarity(ref_vec_occ_draft, ref_vec_true_final))
            if cosine_similarity(ref_vec_occ_draft, ref_vec_true_final) is not None
            else None
        ),
        "similarity_to_original_draft": (
            float(cosine_similarity(ref_vec_occ_draft, ref_vec_orig_draft))
            if cosine_similarity(ref_vec_occ_draft, ref_vec_orig_draft) is not None
            else None
        ),
    }
    fdic_sim_record = {
        "agency": "fdic",
        "role": "original_draft",
        "seed": None,
        "final_words": int(word_count(fdic_draft_text)),
        "similarity_to_official_final": (
            float(cosine_similarity(ref_vec_fdic_draft, ref_vec_true_final))
            if cosine_similarity(ref_vec_fdic_draft, ref_vec_true_final) is not None
            else None
        ),
        "similarity_to_original_draft": (
            float(cosine_similarity(ref_vec_fdic_draft, ref_vec_orig_draft))
            if cosine_similarity(ref_vec_fdic_draft, ref_vec_orig_draft) is not None
            else None
        ),
    }
    all_sim_df = pd.concat(
        [
            all_sim_df,
            pd.DataFrame.from_records([base_sim_record, occ_sim_record, fdic_sim_record]),
        ],
        ignore_index=True,
    )

    write_trace_output(TRACE_XLSX_PATH, all_trace_df)
    write_similarity_output(SIM_XLSX_PATH, all_sim_df)

    def build_vectors_for_seed(role_to_final_vec: Dict[str, Optional[List[float]]]) -> List[Optional[List[float]]]:
        vecs: List[Optional[List[float]]] = [
            ref_vec_true_final,
            ref_vec_orig_draft,
            ref_vec_occ_draft,
            ref_vec_fdic_draft,
        ]
        for rn in roles_to_run:
            vecs.append(role_to_final_vec.get(rn))
        return vecs

    seed_to_role_vecs: Dict[int, Dict[str, Optional[List[float]]]] = {s: {} for s in seeds_list}
    for _, sim_record_local, final_vec, role_name, seed_val in results:
        seed_to_role_vecs[int(seed_val)][str(role_name)] = final_vec

    for seed_val in seeds_list:
        role_vec_map = seed_to_role_vecs.get(int(seed_val), {})
        vecs = build_vectors_for_seed(role_vec_map)
        mat = np.full((n_labels, n_labels), np.nan, dtype=float)
        for i in range(n_labels):
            for j in range(n_labels):
                sim = cosine_similarity(vecs[i], vecs[j])
                if sim is not None:
                    mat[i, j] = float(sim)
                    sum_matrix[i, j] += float(sim)
                    count_matrix[i, j] += 1

    with np.errstate(invalid="ignore"):
        avg_matrix = np.divide(sum_matrix, count_matrix, out=np.full_like(sum_matrix, np.nan), where=count_matrix > 0)

    sim_df_avg = pd.DataFrame(avg_matrix, index=labels, columns=labels)
    SIM_MATRIX_AVG_CSV.parent.mkdir(parents=True, exist_ok=True)
    sim_df_avg.to_csv(SIM_MATRIX_AVG_CSV)

    HEATMAP_AVG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(6, int(0.6 * n_labels))
    fig_h = max(4, int(0.6 * n_labels))
    plt.figure(figsize=(fig_w, fig_h))
    vmin_val = float(np.nanmin(avg_matrix)) if np.isfinite(np.nanmin(avg_matrix)) else 0.0
    vmax_val = float(np.nanmax(avg_matrix)) if np.isfinite(np.nanmax(avg_matrix)) else 1.0
    sns.heatmap(sim_df_avg, vmin=vmin_val, vmax=vmax_val, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Average Cosine Similarity Between Policies Across Seeds")
    plt.tight_layout()
    plt.savefig(HEATMAP_AVG_PNG, dpi=200)
    plt.savefig(HEATMAP_AVG_SVG)
    plt.close()

    print(f"Saved consolidated trace → {TRACE_XLSX_PATH}")
    print(f"Saved consolidated similarities → {SIM_XLSX_PATH}")
    print(f"Saved average policy similarity matrix → {SIM_MATRIX_AVG_CSV}")
    print(f"Saved average heatmap → {HEATMAP_AVG_PNG}, {HEATMAP_AVG_SVG}")

    role_similarity_mc_df = (
        all_sim_df[
            (all_sim_df["role"].isin(roles_to_run)) & (all_sim_df["seed"].notna())
        ][["role", "seed", "similarity_to_official_final"]]
        .rename(columns={"similarity_to_official_final": "similarity"})
        .copy()
    )

    if not role_similarity_mc_df.empty:
        RIDGE_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        palette = sns.color_palette("husl", len(roles_to_run))
        for role_name, color in zip(roles_to_run, palette):
            vals = (
                role_similarity_mc_df.loc[
                    role_similarity_mc_df["role"] == role_name, "similarity"
                ]
                .astype(float)
                .dropna()
            )
            if len(vals) < 2:
                continue
            sns.kdeplot(
                vals,
                ax=ax,
                fill=True,
                common_norm=False,
                alpha=0.25,
                linewidth=2,
                color=color,
                label=role_name,
            )

        baseline_fed = float(base_sim_record["similarity_to_official_final"]) if base_sim_record["similarity_to_official_final"] is not None else None
        baseline_occ = float(occ_sim_record["similarity_to_official_final"]) if occ_sim_record["similarity_to_official_final"] is not None else None
        baseline_fdic = float(fdic_sim_record["similarity_to_official_final"]) if fdic_sim_record["similarity_to_official_final"] is not None else None

        custom_lines: List[Line2D] = []
        custom_labels: List[str] = []
        if baseline_fed is not None:
            ax.axvline(baseline_fed, linestyle="--", color="black", linewidth=1.5)
            custom_lines.append(Line2D([0], [0], color="black", lw=1.5, linestyle="--"))
            custom_labels.append("Draft FRS")
        if baseline_occ is not None:
            ax.axvline(baseline_occ, linestyle=(0, (5, 5)), color="dimgray", linewidth=1.5)
            custom_lines.append(Line2D([0], [0], color="dimgray", lw=1.5, linestyle=(0, (5, 5))))
            custom_labels.append("Draft OCC")
        if baseline_fdic is not None:
            ax.axvline(baseline_fdic, linestyle=(0, (3, 3)), color="gray", linewidth=1.5)
            custom_lines.append(Line2D([0], [0], color="gray", lw=1.5, linestyle=(0, (3, 3))))
            custom_labels.append("Draft FDIC")

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Cosine similarity to official final")
        ax.set_ylabel("")
        ax.set_title("Overlapping densities of final policy similarity across roles (MC seeds)")
        role_legend = ax.legend(title="Role", loc="upper left")
        ax.add_artist(role_legend)
        if custom_lines:
            ax.legend(custom_lines, custom_labels, title="Draft baselines", loc="upper right")
        plt.tight_layout()
        plt.savefig(RIDGE_PNG_PATH, dpi=200)
        plt.savefig(RIDGE_SVG_PATH)
        plt.close()
        print(f"Saved similarity ridge plot → {RIDGE_PNG_PATH}, {RIDGE_SVG_PATH}")

if __name__ == "__main__":
    main()
