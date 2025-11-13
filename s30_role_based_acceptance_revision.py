#!/usr/bin/env python3
"""Simulate role-based acceptance, single-shot revisions, and similarity reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import argparse
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openai
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns

SEED: int = 42
MODEL_NAME: str = "gpt-5"
WORD_LIMIT: int = 2000

PROMPT_TEMPLATE_PATH: Path = Path("prompt_policy_proposal_v2.txt")

EMB_MODEL: str = "text-embedding-3-large"

ACCEPTANCE_CSV_PATH: Path = Path("outputs/s30_comment_acceptance_all_roles.csv")
ACCEPTANCE_CACHE_JSON: Path = Path("outputs/s30_acceptance_cache.json")
ACCEPTANCE_RATE_PNG: Path = Path("plots/s30_acceptance_rates.png")
ACCEPTANCE_RATE_SVG: Path = Path("plots/s30_acceptance_rates.svg")

TRACE_XLSX_PATH: Path = Path("outputs/s30_role_trace_all_agencies_all_roles.xlsx")
SIM_XLSX_PATH: Path = Path("outputs/s30_role_similarity_all_agencies_all_roles.xlsx")
SIM_MATRIX_CSV_PATH: Path = Path("outputs/s30_policy_similarity_matrix.csv")
HEATMAP_PNG_PATH: Path = Path("plots/s30_policy_similarity_heatmap.png")
HEATMAP_SVG_PATH: Path = Path("plots/s30_policy_similarity_heatmap.svg")

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

class CommentAcceptanceOutput(BaseModel):
    """Structured acceptance decision for a single comment under a role."""

    accepted: bool = Field(..., description="True if any part of the comment is accepted.")
    proposed_change: str = Field(
        ..., description="Brief summary of accepted changes. Up to three sentences."
    )

class BatchRevisionOutput(BaseModel):
    """Structured output for revising a full policy once from proposals."""

    revised_policy: str = Field(..., description="Revised policy text under 2,000 words.")

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

def load_prompt_template(path: Path) -> str:
    """Load the revision prompt template from a file."""

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
    """Write the revision trace to an Excel file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        sanitize_for_excel(trace_df).to_excel(writer, index=False, sheet_name="trace")

def write_similarity_output(output_path: Path, similarity_df: pd.DataFrame) -> None:
    """Write the cosine similarity results to an Excel file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        similarity_df.to_excel(writer, index=False, sheet_name="similarities")

def compute_accept_cache_key(*, role_name: str, model: str, comment_id: str, comment_text: str) -> str:
    """Build a deterministic cache key for acceptance using role, model, id, and text hash."""

    sha = hashlib.sha256()
    sha.update((role_name + "|" + model + "|" + comment_id + "|").encode("utf-8"))
    sha.update(comment_text.encode("utf-8", errors="ignore"))
    return sha.hexdigest()

def acceptance_messages(role_text: str, comment_text: str) -> List[Dict[str, str]]:
    """Construct chat messages for the acceptance prompt."""

    prompt = (
        "You will be given a comment submitted in response to a policy proposal.\n\n"
        f"Your role is {role_text}\n\n"
        "Your task is to review the comment and decide if you would accept any of the comment’s proposals. "
        "The decision to accept or reject the proposal should be based on your role and nothing else. "
        "You may choose to accept parts of the comment or the comment in entirety. "
        "Act according to your role in full extent.\n\n"
        "Output two fields:\n"
        "accepted: True if any part of the comment is accepted. False if all of it is ignored.\n"
        "proposed_change: brief summary of the accepted changes. Up to three sentences.\n\n"
        "Comment:\n"
        f"{comment_text}"
    )
    return [{"role": "user", "content": prompt}]

def revision_messages(template_text: str, role_text: str, policy_text: str, proposals_block: str) -> List[Dict[str, str]]:
    """Construct chat messages for single-shot revision using accepted proposals as comments."""

    prompt = template_text.format(
        role=role_text,
        policy=policy_text,
        comments=proposals_block,
        words=word_count(policy_text),
    )
    return [{"role": "user", "content": prompt}]

def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Compute embeddings for a list of texts. Returns None for failures."""

    vectors: List[Optional[List[float]]] = []
    for t in texts:
        try:
            resp = openai.embeddings.create(model=EMB_MODEL, input=[t])
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

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for acceptance and single-shot revision by role."""

    parser = argparse.ArgumentParser(
        description="Role-based acceptance and single-shot revisions."
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
        "--comment-limit",
        type=int,
        default=None,
        help="Optional cap on total comments evaluated per role (after loading).",
    )
    parser.add_argument(
        "--roles-workers",
        type=int,
        default=1,
        help="Number of roles to run in parallel (1 = sequential).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Dry run: estimate prompt tokens for acceptance and revision; no API calls.",
    )
    return parser.parse_args()

def main() -> None:
    """Entry point for comment acceptance, proposal aggregation, and policy similarity analysis."""

    np.random.seed(SEED)

    args = parse_args()
    agency: str = args.agency
    roles_arg: str = args.roles
    comment_limit: Optional[int] = args.comment_limit
    roles_workers: int = max(1, int(args.roles_workers))
    plan_only: bool = bool(args.plan_only)

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
    if comment_limit is not None:
        comments_all = comments_all[: max(0, int(comment_limit))]

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)

    client = configure_openai_client()

    if ACCEPTANCE_CACHE_JSON.exists():
        acceptance_cache: Dict[str, Dict[str, object]] = json.loads(
            ACCEPTANCE_CACHE_JSON.read_text(encoding="utf-8")
        )
    else:
        acceptance_cache = {}
    acceptance_cache_lock = threading.Lock()

    def estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
        """Estimate token count for a list of chat messages using o200k_base."""

        enc = tiktoken.get_encoding("o200k_base")
        total = 0
        for msg in messages:
            total += len(enc.encode(str(msg.get("content", ""))))
        return total

    def proposals_from_cache_for_role(role_name: str) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for obj in acceptance_cache.values():
            try:
                if str(obj.get("role")) != role_name:
                    continue
                if not bool(obj.get("accepted", False)):
                    continue
                cid = str(obj.get("comment_id", "")).strip()
                pchg = str(obj.get("proposed_change", "")).strip()
                if cid and pchg:
                    items.append((cid, pchg))
            except Exception:
                continue
        return items

    if plan_only:

        acceptance_tokens_total: int = 0
        acceptance_tokens_by_role: Dict[str, int] = {rn: 0 for rn in roles_to_run}
        print("[plan-only] Estimating acceptance prompt tokens …")
        for rn in tqdm(roles_to_run, desc="Roles (acceptance)"):
            role_text = ROLES[rn]
            subtotal = 0
            for cm in tqdm(comments_all, desc=f"Comments [{rn}]", leave=False):
                msgs = acceptance_messages(role_text=role_text, comment_text=str(cm["text"]))
                subtotal += estimate_prompt_tokens(msgs)
            acceptance_tokens_by_role[rn] = subtotal
            acceptance_tokens_total += subtotal
        for rn in roles_to_run:
            print(f"[plan-only] Acceptance tokens for {rn} (approx): {acceptance_tokens_by_role[rn]}")
        print(f"[plan-only] Acceptance tokens total (approx): {acceptance_tokens_total}")

        revision_tokens_total: int = 0
        revision_tokens_by_role: Dict[str, int] = {rn: 0 for rn in roles_to_run}
        print("[plan-only] Estimating revision prompt tokens (using cached accepted proposals if available) …")
        for rn in roles_to_run:
            role_text = ROLES[rn]
            props = proposals_from_cache_for_role(rn)
            proposals_block = "\n\n".join(["ID: " + cid + "\n" + pchg for cid, pchg in props])
            msgs = revision_messages(
                template_text=prompt_template,
                role_text=role_text,
                policy_text=original_draft,
                proposals_block=proposals_block,
            )
            toks = estimate_prompt_tokens(msgs)
            revision_tokens_by_role[rn] = toks
            revision_tokens_total += toks
            print(
                f"[plan-only] Revision tokens for {rn} (approx): {toks} "
                f"(cached accepted proposals: {len(props)})"
            )
        print(f"[plan-only] Revision tokens total (approx): {revision_tokens_total}")

        grand_total = acceptance_tokens_total + revision_tokens_total
        print(f"[plan-only] Grand total planned prompt tokens (approx): {grand_total}")
        print("Plan-only mode: exiting before any LLM calls.")
        return

    def run_acceptance_for_role(role_name: str) -> List[Dict[str, object]]:
        role_text = ROLES[role_name]
        records_local: List[Dict[str, object]] = []
        local_updates: Dict[str, Dict[str, object]] = {}
        role_iter = tqdm(comments_all, desc=f"Acceptance [{role_name}]")
        for cm in role_iter:
            cm_id = str(cm["id"]).strip()
            cm_text = str(cm["text"]).strip()
            cache_key = compute_accept_cache_key(
                role_name=role_name,
                model=MODEL_NAME,
                comment_id=cm_id,
                comment_text=cm_text,
            )
            cached_obj: Optional[Dict[str, object]]
            with acceptance_cache_lock:
                cached_obj = acceptance_cache.get(cache_key)
            if cached_obj is not None:
                accepted_val = bool(cached_obj.get("accepted", False))
                proposed_change_val = str(cached_obj.get("proposed_change", ""))
            else:
                messages = acceptance_messages(role_text=role_text, comment_text=cm_text)
                parsed = client.responses.parse(
                    model=MODEL_NAME,
                    input=messages,
                    text_format=CommentAcceptanceOutput,
                )
                out = parsed.output_parsed
                accepted_val = bool(out.accepted)
                proposed_change_val = str(out.proposed_change)
                local_updates[cache_key] = {
                    "role": role_name,
                    "comment_id": cm_id,
                    "accepted": accepted_val,
                    "proposed_change": proposed_change_val,
                }

            records_local.append(
                {
                    "role": role_name,
                    "comment_id": cm_id,
                    "accepted": bool(accepted_val),
                    "proposed_change": proposed_change_val,
                }
            )

        if local_updates:
            with acceptance_cache_lock:
                acceptance_cache.update(local_updates)
                ACCEPTANCE_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
                ACCEPTANCE_CACHE_JSON.write_text(
                    json.dumps(acceptance_cache, ensure_ascii=False), encoding="utf-8"
                )
        return records_local

    acceptance_records: List[Dict[str, object]] = []
    if roles_workers == 1:
        for rn in roles_to_run:
            acceptance_records.extend(run_acceptance_for_role(rn))
    else:
        with ThreadPoolExecutor(max_workers=roles_workers) as executor:
            futures = {executor.submit(run_acceptance_for_role, rn): rn for rn in roles_to_run}
            for fut in as_completed(futures):
                acceptance_records.extend(fut.result())

    acceptance_by_comment_role_df = pd.DataFrame.from_records(acceptance_records)
    ACCEPTANCE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    acceptance_by_comment_role_df.to_csv(ACCEPTANCE_CSV_PATH, index=False)

    acceptance_rate_df = (
        acceptance_by_comment_role_df.groupby("role", as_index=False)["accepted"].mean()
        .rename(columns={"accepted": "acceptance_rate"})
    )
    ACCEPTANCE_RATE_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=acceptance_rate_df, x="role", y="acceptance_rate", color="#8b8b8b")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.title("Acceptance rate by role")
    plt.tight_layout()
    plt.savefig(ACCEPTANCE_RATE_PNG, dpi=200)
    plt.savefig(ACCEPTANCE_RATE_SVG)
    plt.close()

    ref_vecs = embed_texts([original_final, original_draft, occ_draft_text, fdic_draft_text])
    ref_vec_true_final = ref_vecs[0]
    ref_vec_orig_draft = ref_vecs[1]
    ref_vec_occ_draft = ref_vecs[2]
    ref_vec_fdic_draft = ref_vecs[3]

    all_trace_records: List[Dict[str, object]] = []
    all_similarity_records: List[Dict[str, object]] = []

    role_final_vecs: Dict[str, Optional[List[float]]] = {}

    def run_revision_for_role(role_name: str) -> Tuple[Dict[str, object], Dict[str, object], Optional[List[float]]]:
        role_text = ROLES[role_name]
        role_accept_df = acceptance_by_comment_role_df[
            acceptance_by_comment_role_df["role"] == role_name
        ]
        accepted_df = role_accept_df[role_accept_df["accepted"].astype(bool)]

        if accepted_df.empty:
            final_policy_text = original_draft
            final_vec = embed_texts([final_policy_text])[0]
            sim_final_vs_true_final = cosine_similarity(final_vec, ref_vec_true_final)
            sim_final_vs_original_draft = cosine_similarity(final_vec, ref_vec_orig_draft)
            trace_record = {
                "agency": agency,
                "role": role_name,
                "num_accepted_proposals": int(0),
                "proposals_words": int(0),
                "revised_policy_words": int(word_count(final_policy_text)),
                "revised_policy": final_policy_text,
            }
            sim_record = {
                "agency": agency,
                "role": role_name,
                "final_words": int(word_count(final_policy_text)),
                "similarity_to_official_final": float(sim_final_vs_true_final)
                if sim_final_vs_true_final is not None
                else None,
                "similarity_to_original_draft": float(sim_final_vs_original_draft)
                if sim_final_vs_original_draft is not None
                else None,
            }
            return trace_record, sim_record, final_vec

        proposals_parts: List[str] = []
        for _, row in accepted_df.iterrows():
            proposals_parts.append(
                "ID: " + str(row["comment_id"]).strip() + "\n" + str(row["proposed_change"]).strip()
            )
        proposals_block = "\n\n".join(proposals_parts)

        messages = revision_messages(
            template_text=prompt_template,
            role_text=role_text,
            policy_text=original_draft,
            proposals_block=proposals_block,
        )
        parsed = client.responses.parse(
            model=MODEL_NAME,
            input=messages,
            text_format=BatchRevisionOutput,
        )
        revised_policy = parsed.output_parsed.revised_policy

        proposals_words = word_count(proposals_block)
        final_vec = embed_texts([revised_policy])[0]
        sim_final_vs_true_final = cosine_similarity(final_vec, ref_vec_true_final)
        sim_final_vs_original_draft = cosine_similarity(final_vec, ref_vec_orig_draft)

        trace_record = {
            "agency": agency,
            "role": role_name,
            "num_accepted_proposals": int(len(accepted_df)),
            "proposals_words": int(proposals_words),
            "revised_policy_words": int(word_count(revised_policy)),
            "revised_policy": revised_policy,
        }
        sim_record = {
            "agency": agency,
            "role": role_name,
            "final_words": int(word_count(revised_policy)),
            "similarity_to_official_final": float(sim_final_vs_true_final)
            if sim_final_vs_true_final is not None
            else None,
            "similarity_to_original_draft": float(sim_final_vs_original_draft)
            if sim_final_vs_original_draft is not None
            else None,
        }
        return trace_record, sim_record, final_vec

    if roles_workers == 1:
        for rn in roles_to_run:
            tr, sr, fv = run_revision_for_role(rn)
            all_trace_records.append(tr)
            all_similarity_records.append(sr)
            role_final_vecs[rn] = fv
    else:
        with ThreadPoolExecutor(max_workers=roles_workers) as executor:
            futures = {executor.submit(run_revision_for_role, rn): rn for rn in roles_to_run}
            for fut in as_completed(futures):
                tr, sr, fv = fut.result()
                all_trace_records.append(tr)
                all_similarity_records.append(sr)
                role_final_vecs[sr["role"]] = fv

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

    write_trace_output(TRACE_XLSX_PATH, all_trace_df)
    write_similarity_output(SIM_XLSX_PATH, all_sim_df)

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
    plt.title("Cosine Similarity Between Policies (Accepted-Proposals Revision)")
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG_PATH, dpi=200)
    plt.savefig(HEATMAP_SVG_PATH)
    plt.close()

    print(f"Saved acceptance CSV → {ACCEPTANCE_CSV_PATH}")
    print(f"Saved acceptance rate plot → {ACCEPTANCE_RATE_PNG}, {ACCEPTANCE_RATE_SVG}")
    print(f"Saved consolidated trace → {TRACE_XLSX_PATH}")
    print(f"Saved consolidated similarities → {SIM_XLSX_PATH}")
    print(f"Saved policy similarity matrix → {SIM_MATRIX_CSV_PATH}")
    print(f"Saved policy similarity heatmap → {HEATMAP_PNG_PATH}, {HEATMAP_SVG_PATH}")

if __name__ == "__main__":
    main()
