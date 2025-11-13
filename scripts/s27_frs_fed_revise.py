#!/usr/bin/env python3
"""Revise policy drafts with role-conditioned GPT passes and record similarities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
import argparse
import json
import hashlib
import threading
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

SEED: int = 42
MODEL_NAME: str = "gpt-5"
WORD_LIMIT: int = 2000

TRACE_OUTPUT_DEFAULT: Path = Path("outputs/s27_role_trace.xlsx")
SIM_OUTPUT_DEFAULT: Path = Path("outputs/s27_role_similarity.xlsx")
CACHE_JSON_DEFAULT: Path = Path("outputs/s27_policy_revision_cache.json")
PROMPT_TEMPLATE_PATH: Path = Path("prompt_policy_proposal_v2.txt")

EMB_MODEL: str = "text-embedding-3-large"
BATCH_SIZE_DEFAULT: int = 10

SIM_MATRIX_CSV_PATH: Path = Path("outputs/s27_policy_similarity_matrix.csv")
HEATMAP_PNG_PATH: Path = Path("plots/s27_policy_similarity_heatmap.png")
HEATMAP_SVG_PATH: Path = Path("plots/s27_policy_similarity_heatmap.svg")

ORIGINAL_FINAL_PATH: Path = Path("data/policies/final.txt")
AGENCY_TO_DRAFT: dict[str, Path] = {
    "fed": Path("data/policies/drafts/fed.txt"),
    "occ": Path("data/policies/drafts/occ.txt"),
    "fdic": Path("data/policies/drafts/fdic.txt"),
}

COMMENT_DIRS: list[Path] = [
    Path("data/comments/occ/txt"),
    Path("data/comments/fdic/txt"),
    Path("data/comments/fed/txt"),
]

ROLES: dict[str, str] = {
    "monetary": (
        "to follow your mandate to maintain long run growth of the monetary and credit "
        "aggregates commensurate with the economy's long run potential to increase production, "
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
    """Read a UTF-8 text file and return its contents stripped.
    
    Args:
        path: Path to text file.
        
    Returns:
        Stripped file contents.
    """
    return path.read_text(encoding="utf-8", errors="ignore").strip()


WORD_PATTERN = __import__("re").compile(r"\w+")


def word_count(text: str) -> int:
    """Compute the number of word tokens in a string.
    
    Args:
        text: Input text.
        
    Returns:
        Number of words.
    """
    return len(WORD_PATTERN.findall(text))


def comment_agency_from_path(p: Path) -> str:
    """Infer the agency from a comment file path name.
    
    Args:
        p: Comment file path.
        
    Returns:
        Agency name (occ, fed, or fdic).
    """
    s = str(p).lower()
    if "comments/occ" in s:
        return "occ"
    if "comments/fed" in s:
        return "fed"
    if "comments/fdic" in s:
        return "fdic"
    return "fdic"


def load_all_comments(dirs: list[Path]) -> list[dict[str, str]]:
    """Load comments from multiple directories as a list of dicts with id and text.
    
    Args:
        dirs: List of directories to search.
        
    Returns:
        List of comment dictionaries with id and text keys.
    """
    items: list[dict[str, str]] = []
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
) -> str:
    """Build a deterministic cache key from the full prompt content and model.

    Including the full prompt ensures changes in template, role wording, policy text,
    comment order/content, and word count all invalidate the cache automatically.
    """

    sha = hashlib.sha256()
    sha.update((agency + "|" + role_name + "|" + model + "|").encode("utf-8"))
    sha.update(prompt_text.encode("utf-8", errors="ignore"))
    return sha.hexdigest()

def llm_revise_batch(
    client: openai.OpenAI,
    model: str,
    template_text: str,
    role_text: str,
    policy_text: str,
    batch_comments: List[Dict[str, str]],
) -> BatchRevisionOutput:
    """Call the LLM to revise a full policy from a batch of comments."""

    messages = build_batch_messages(
        template_text=template_text,
        role_text=role_text,
        policy_text=policy_text,
        batch_comments=batch_comments,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        response_model=BatchRevisionOutput,
        seed=SEED,
    )
    return response

def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Compute embeddings for a list of texts.

    If an input is too long or otherwise fails (e.g., context length error), returns None for that item
    instead of raising. This allows downstream similarity to yield None gracefully.
    """

    vectors: List[Optional[List[float]]] = []
    for t in texts:
        try:
            resp = openai.embeddings.create(model=EMB_MODEL, input=[t])
            vectors.append(resp.data[0].embedding)
        except Exception:
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
    cache: Dict[str, Dict[str, object]],
    agency: str,
    role_name: str,
) -> int:
    """Approximate total tokens for all uncached batch prompts, stepping through cache sequentially."""

    enc = tiktoken.get_encoding("o200k_base")
    planned_total = 0
    policy_cursor = initial_policy
    for i, batch in enumerate(batches, start=1):
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
        )
        if key in cache:
            cached = cache[key]
            policy_cursor = str(cached.get("revised_policy", ""))
            continue
        planned_total += len(enc.encode(prompt))
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
    """Parse CLI arguments for role-based batched revision."""

    parser = argparse.ArgumentParser(
        description="Role-conditioned policy revisions."
    )
    parser.add_argument(
        "--agency",
        type=str,
        choices=["fed"],
        default="fed",
        help="Agency whose draft to revise (currently limited to 'fed').",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="all",
        help=(
            "Comma-separated list of roles to run (e.g., 'monetary,banking'). "
            "Use 'all' to run every role (default)."
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
        help="Optional cap on number of batches processed.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only compute and print planned token count; do not call the LLM.",
    )
    parser.add_argument(
        "--cache-json",
        type=str,
        default=str(CACHE_JSON_DEFAULT),
        help="Path to JSON cache file for batch results.",
    )
    parser.add_argument(
        "--roles-workers",
        type=int,
        default=10,
        help="Number of roles to run in parallel (1 = sequential).",
    )
    return parser.parse_args()

def main() -> None:
    """Entry point for role-based batched policy revision and similarity analysis."""

    np.random.seed(SEED)

    args = parse_args()
    agency: str = args.agency
    roles_arg: str = args.roles
    batch_size: int = int(args.batch_size)
    comment_limit: Optional[int] = args.comment_limit
    max_batches: Optional[int] = args.max_batches
    plan_only: bool = bool(args.plan_only)
    cache_path: Path = Path(args.cache_json)
    roles_workers: int = max(1, int(args.roles_workers))

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

    roles_workers = min(roles_workers, len(roles_to_run))

    comments_all = load_all_comments(COMMENT_DIRS)
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)

    cache: Dict[str, Dict[str, object]]
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    if plan_only:
        total_tokens = 0
        for role_name in roles_to_run:
            role_text = ROLES[role_name]
            batches_all = build_batches(
                comments=comments_all, batch_size=batch_size, comment_limit=comment_limit, seed=SEED
            )
            if max_batches is not None:
                batches_all = batches_all[: max(0, max_batches)]
            planned_tokens = plan_tokens_for_batches(
                template_text=prompt_template,
                role_text=role_text,
                initial_policy=original_draft,
                batches=batches_all,
                cache=cache,
                agency=agency,
                role_name=role_name,
            )
            total_tokens += planned_tokens
            print(f"[{role_name}] Planned tokens (approx): {planned_tokens}")
        print(f"Total planned tokens (approx): {total_tokens}")
        print("Plan-only mode: exiting before any LLM calls.")
        return

    client = configure_openai_client()

    ref_vecs = embed_texts([original_final, original_draft, occ_draft_text, fdic_draft_text])
    ref_vec_true_final = ref_vecs[0]
    ref_vec_orig_draft = ref_vecs[1]
    ref_vec_occ_draft = ref_vecs[2]
    ref_vec_fdic_draft = ref_vecs[3]

    all_trace_records: List[Dict[str, object]] = []
    all_similarity_records: List[Dict[str, object]] = []

    cache_lock = threading.Lock()
    io_lock = threading.Lock()
    consolidated_trace_path = Path("outputs/s27_role_trace_all_agencies_all_roles.xlsx")
    role_final_vecs: Dict[str, Optional[List[float]]] = {}

    def run_role(role_name: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        role_text_local = ROLES[role_name]
        batches_all_local = build_batches(
            comments=comments_all, batch_size=batch_size, comment_limit=comment_limit, seed=SEED
        )
        if max_batches is not None:
            batches_all_local = batches_all_local[: max(0, max_batches)]

        trace_records_local: List[Dict[str, object]] = []
        policy_current_local = original_draft

        for i, batch in enumerate(tqdm(batches_all_local, desc=f"Batches [{role_name}]"), start=1):
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
            )
            revised_policy: str
            with cache_lock:
                cached_obj = cache.get(cache_key)
            if cached_obj is not None:
                revised_policy = str(cached_obj.get("revised_policy", ""))
            else:
                parsed = client.responses.parse(
                    model=MODEL_NAME,
                    input=messages,
                    text_format=BatchRevisionOutput,
                )
                result = parsed.output_parsed
                revised_policy = result.revised_policy
                with cache_lock:
                    cache[cache_key] = {
                        "revised_policy": revised_policy,
                        "agency": agency,
                        "role": role_name,
                        "batch_num": i,
                        "prompt_sha": cache_key,
                    }
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

            n_words = word_count(revised_policy)
            print(f"[{role_name}] Batch {i}: {n_words} words")

            step_vec = embed_texts([revised_policy])[0]
            step_sim_true_final = cosine_similarity(step_vec, ref_vec_true_final)
            step_sim_orig_draft = cosine_similarity(step_vec, ref_vec_orig_draft)

            record = {
                    "agency": agency,
                "role": role_name,
                "batch_num": i,
                "comment_ids": ",".join([str(c["id"]) for c in batch]),
                "batch_comments_words": int(batch_comments_word_count),
                "n_words": int(n_words),
                "similarity_to_official_final": float(step_sim_true_final) if step_sim_true_final is not None else None,
                "similarity_to_original_draft": float(step_sim_orig_draft) if step_sim_orig_draft is not None else None,
                "revised_policy": revised_policy,
            }
            trace_records_local.append(record)

            with io_lock:
                all_trace_records.append(record)
                tmp_df = pd.DataFrame.from_records(all_trace_records)
                write_trace_output(consolidated_trace_path, tmp_df)
            policy_current_local = revised_policy

        final_policy_local = policy_current_local
        final_vec = embed_texts([final_policy_local])[0]
        sim_final_vs_true_final = cosine_similarity(final_vec, ref_vec_true_final)
        sim_final_vs_original_draft = cosine_similarity(final_vec, ref_vec_orig_draft)

        sim_record_local: Dict[str, object] = {
                    "agency": agency,
            "role": role_name,
            "final_words": int(word_count(final_policy_local)),
            "similarity_to_official_final": float(sim_final_vs_true_final),
            "similarity_to_original_draft": float(sim_final_vs_original_draft),
        }

        with io_lock:
            role_final_vecs[role_name] = final_vec

        return trace_records_local, sim_record_local

    if roles_workers == 1:
        for role_name in roles_to_run:
            trace_records_local, sim_record_local = run_role(role_name)
            all_trace_records.extend(trace_records_local)
            all_similarity_records.append(sim_record_local)
    else:
        with ThreadPoolExecutor(max_workers=roles_workers) as executor:
            futures = {executor.submit(run_role, rn): rn for rn in roles_to_run}
            for fut in as_completed(futures):
                trace_records_local, sim_record_local = fut.result()
                all_trace_records.extend(trace_records_local)
                all_similarity_records.append(sim_record_local)

    all_trace_df = pd.DataFrame.from_records(all_trace_records)
    all_sim_df = pd.DataFrame.from_records(all_similarity_records)

    base_sim_record = {
                "agency": agency,
        "role": "original_draft",
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

    consolidated_trace = Path("outputs/s27_role_trace_all_agencies_all_roles.xlsx")
    consolidated_sim = Path("outputs/s27_role_similarity_all_agencies_all_roles.xlsx")

    write_trace_output(consolidated_trace, all_trace_df)
    write_similarity_output(consolidated_sim, all_sim_df)

    labels: List[str] = [
        "official_final",
        "draft_fed",
        "draft_occ",
        "draft_fdic",
    ]
    vectors: List[Optional[List[float]]] = [
        ref_vec_true_final,
        ref_vec_orig_draft,
        ref_vec_occ_draft,
        ref_vec_fdic_draft,
    ]
    for rn in roles_to_run:
        labels.append(f"final_{rn}")
        vectors.append(role_final_vecs.get(rn))

    n = len(labels)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            sim = cosine_similarity(vectors[i], vectors[j])
            matrix[i, j] = float(sim) if sim is not None else np.nan

    sim_df = pd.DataFrame(matrix, index=labels, columns=labels)
    SIM_MATRIX_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(SIM_MATRIX_CSV_PATH)

    HEATMAP_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(6, int(0.6 * n))
    fig_h = max(4, int(0.6 * n))
    plt.figure(figsize=(fig_w, fig_h))

    vmin_val = float(np.nanmin(matrix))
    vmax_val = float(np.nanmax(matrix))
    sns.heatmap(sim_df, vmin=vmin_val, vmax=vmax_val, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Cosine Similarity Between Policies")
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG_PATH, dpi=200)
    plt.savefig(HEATMAP_SVG_PATH)
    plt.close()

    print(f"Saved consolidated trace → {consolidated_trace}")
    print(f"Saved consolidated similarities → {consolidated_sim}")
    print(f"Saved policy similarity matrix → {SIM_MATRIX_CSV_PATH}")
    print(f"Saved policy similarity heatmap → {HEATMAP_PNG_PATH}, {HEATMAP_SVG_PATH}")

if __name__ == "__main__":
    main()
