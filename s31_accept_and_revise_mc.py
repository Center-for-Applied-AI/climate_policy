#!/usr/bin/env python3
"""Run Monte Carlo acceptance and revision sweeps with GPT-5."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import threading
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

GLOBAL_RANDOM_SEED: int = 42
MODEL_NAME: str = "gpt-5"
WORD_LIMIT: int = 2000

PROMPT_TEMPLATE_PATH: Path = Path("prompt_policy_proposal_v2.txt")

EMB_MODEL: str = "text-embedding-3-large"

TRACE_XLSX_PATH: Path = Path("outputs/s31_role_trace_all_agencies_all_roles.xlsx")
SIM_XLSX_PATH: Path = Path("outputs/s31_role_similarity_all_agencies_all_roles.xlsx")
SIM_MATRIX_AVG_CSV: Path = Path("outputs/s31_policy_similarity_matrix_avg.csv")
HEATMAP_AVG_PNG: Path = Path("plots/s31_policy_similarity_heatmap_avg.png")
HEATMAP_AVG_SVG: Path = Path("plots/s31_policy_similarity_heatmap_avg.svg")
STACKED_PNG_PATH: Path = Path("plots/s31_similarity_stacked_official_final.png")
STACKED_SVG_PATH: Path = Path("plots/s31_similarity_stacked_official_final.svg")
T_MATRIX_CSV_PATH: Path = Path("outputs/s31_role_t_matrix.csv")
T_MATRIX_PNG_PATH: Path = Path("plots/s31_role_t_matrix_heatmap.png")
T_MATRIX_SVG_PATH: Path = Path("plots/s31_role_t_matrix_heatmap.svg")
ACCEPTANCE_RATES_MC_CSV: Path = Path("outputs/s31_acceptance_rates_by_role_mc.csv")
ACCEPTANCE_RATES_MC_PNG: Path = Path("plots/s31_acceptance_rates_with_ci.png")
ACCEPTANCE_RATES_MC_SVG: Path = Path("plots/s31_acceptance_rates_with_ci.svg")
SIMILARITY_RATES_MC_CSV: Path = Path("outputs/s31_similarity_to_final_by_role_mc.csv")
SIMILARITY_RATES_MC_PNG: Path = Path("plots/s31_similarity_to_final_with_ci.png")
SIMILARITY_RATES_MC_SVG: Path = Path("plots/s31_similarity_to_final_with_ci.svg")

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
    "french_cheese": "to act in the interest of French cheese producers.",
    "cinderella": "to act in the interests of Cinderella.",
}

class CommentAcceptanceOutput(BaseModel):
    """Structured acceptance decision for a single comment under a role.

    Attributes
    ----------
    accepted
        True if any part of the comment is accepted.
    proposed_change
        Brief summary of accepted changes. Up to three sentences.
    """

    accepted: bool = Field(..., description="True if any part of the comment is accepted.")
    proposed_change: str = Field(
        ..., description="Brief summary of accepted changes. Up to three sentences."
    )

class BatchRevisionOutput(BaseModel):
    """Structured output for revising a full policy once from proposals.

    Attributes
    ----------
    revised_policy
        Revised policy text under 2,000 words.
    """

    revised_policy: str = Field(..., description="Revised policy text under 2,000 words.")

def configure_openai_client() -> OpenAI:
    """Configure OpenAI client using environment variable.

    Returns
    -------
    OpenAI
        Configured client instance.
    """
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    client = OpenAI(api_key=openai.api_key)
    return client

def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file and return its contents stripped.

    Parameters
    ----------
    path
        Path to a text file.

    Returns
    -------
    str
        File contents stripped of surrounding whitespace.
    """

    return path.read_text(encoding="utf-8", errors="ignore").strip()

_WORD_RE = __import__("re").compile(r"\w+")

def word_count(text: str) -> int:
    """Compute the number of word tokens in a string.

    Parameters
    ----------
    text
        Input string.

    Returns
    -------
    int
        Number of word tokens.
    """

    return len(_WORD_RE.findall(text))

def comment_agency_from_path(p: Path) -> str:
    """Infer the agency from a comment file path name.

    Parameters
    ----------
    p
        Path to a comment file.

    Returns
    -------
    str
        Agency code inferred from the path.
    """

    s = str(p).lower()
    if "comments/occ" in s:
        return "occ"
    if "comments/fed" in s:
        return "fed"
    if "comments/fdic" in s:
        return "fdic"
    return "fdic"

def load_all_comments(dirs: List[Path]) -> List[Dict[str, str]]:
    """Load comments from multiple directories.

    Parameters
    ----------
    dirs
        List of directories with .txt files.

    Returns
    -------
    List[Dict[str, str]]
        Each item has keys "id" and "text".
    """

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
    """Resolve and validate input draft and final policy paths for an agency.

    Parameters
    ----------
    agency
        Agency code.

    Returns
    -------
    Tuple[Path, Path]
        Paths to the draft policy and the official final policy.
    """

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
    """Load the revision prompt template from a file.

    Parameters
    ----------
    path
        Path to the prompt template file.

    Returns
    -------
    str
        Template contents.
    """

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return read_text_file(path)

def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Drop characters illegal for Excel XML storage from object columns.

    Parameters
    ----------
    df
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """

    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    clean_df = df.copy()
    obj_cols = [c for c in clean_df.columns if pd.api.types.is_object_dtype(clean_df[c])]
    for col in obj_cols:
        clean_df[col] = clean_df[col].map(
            lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
        )
    return clean_df

def write_trace_output(output_path: Path, trace_df: pd.DataFrame) -> None:
    """Write the revision trace to an Excel file.

    Parameters
    ----------
    output_path
        Destination path.
    trace_df
        Trace DataFrame.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        sanitize_for_excel(trace_df).to_excel(writer, index=False, sheet_name="trace")

def write_similarity_output(output_path: Path, similarity_df: pd.DataFrame) -> None:
    """Write the cosine similarity results to an Excel file.

    Parameters
    ----------
    output_path
        Destination path.
    similarity_df
        Similarity DataFrame.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        similarity_df.to_excel(writer, index=False, sheet_name="similarities")

def acceptance_messages(role_text: str, comment_text: str) -> List[Dict[str, str]]:
    """Construct chat messages for the acceptance prompt.

    Parameters
    ----------
    role_text
        Role description to condition the decision.
    comment_text
        Full comment text.

    Returns
    -------
    List[Dict[str, str]]
        Single-message list for the LLM API.
    """

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
    """Construct chat messages for single-shot revision using accepted proposals as comments.

    Parameters
    ----------
    template_text
        Prompt template with placeholders.
    role_text
        Role description.
    policy_text
        Initial policy text.
    proposals_block
        Concatenated accepted proposals, formatted as ID and summary pairs.

    Returns
    -------
    List[Dict[str, str]]
        Single-message list for the LLM API.
    """

    prompt = template_text.format(
        role=role_text,
        policy=policy_text,
        comments=proposals_block,
        words=word_count(policy_text),
    )
    return [{"role": "user", "content": prompt}]

def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Compute embeddings for a list of texts.

    Parameters
    ----------
    texts
        List of input texts.

    Returns
    -------
    List[Optional[List[float]]]
        Embedding vectors, or None if a call failed.
    """

    vectors: List[Optional[List[float]]] = []
    for t in texts:
        resp = openai.embeddings.create(model=EMB_MODEL, input=[t])
        vectors.append(resp.data[0].embedding)
    return vectors

def cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> Optional[float]:
    """Compute cosine similarity.

    Parameters
    ----------
    vec_a
        First vector or None.
    vec_b
        Second vector or None.

    Returns
    -------
    Optional[float]
        Cosine similarity in [0, 1], or None if invalid.
    """

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
    """Parse CLI arguments for MC runs of s30 logic and analysis.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Monte Carlo acceptance and revisions (GPT-5)."
    )
    parser.add_argument(
        "--agency",
        type=str,
        choices=["fed"],
        default="fed",
        help="Agency whose draft to revise (currently limited to \"fed\").",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="all",
        help=(
            "Comma-separated list of roles to run (e.g., \"monetary,banking\"). Use \"all\" to run every role."
        ),
    )
    parser.add_argument(
        "--comment-limit",
        type=int,
        default=None,
        help="Optional cap on total comments evaluated per role (after loading).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of Monte Carlo seeds to run (seeds are 1..num_seeds).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Starting seed index (default 1).",
    )
    parser.add_argument(
        "--accept-workers",
        type=int,
        default=50,
        help="Maximum parallel acceptance calls per role (comments at a time).",
    )
    parser.add_argument(
        "--plots-from-cache",
        action="store_true",
        help=(
            "Skip all LLM calls; read cached outputs (per-seed acceptance CSVs, consolidated "
            "similarities XLSX, and average matrix CSV) to regenerate plots only."
        ),
    )
    return parser.parse_args()

def main() -> None:
    """Entry point for MC acceptance, single-shot revision, and similarity analysis."""

    np.random.seed(GLOBAL_RANDOM_SEED)

    args = parse_args()
    agency: str = args.agency
    roles_arg: str = args.roles
    comment_limit: Optional[int] = args.comment_limit
    num_seeds: int = max(1, int(args.num_seeds))
    seed_start: int = int(args.seed_start)
    accept_workers: int = max(1, int(args.accept_workers))
    plots_from_cache: bool = bool(args.plots_from_cache)

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

    if plots_from_cache:

        role_similarity_mc_df = pd.DataFrame()
        baseline_fed = None
        baseline_occ = None
        baseline_fdic = None
        if SIM_XLSX_PATH.exists():
            sim_all_df = pd.read_excel(SIM_XLSX_PATH, sheet_name="similarities")
            base_rows = sim_all_df[sim_all_df["role"] == "original_draft"].copy()
            def _first_val(df: pd.DataFrame) -> Optional[float]:
                s = df["similarity_to_official_final"].astype(float).dropna()
                return float(s.iloc[0]) if not s.empty else None
            baseline_fed = _first_val(base_rows[base_rows["agency"] == agency])
            baseline_occ = _first_val(base_rows[base_rows["agency"] == "occ"])
            baseline_fdic = _first_val(base_rows[base_rows["agency"] == "fdic"])
            role_similarity_mc_df = (
                sim_all_df[
                    (sim_all_df["role"].isin(roles_to_run)) & (sim_all_df["seed"].notna())
                ][["role", "seed", "similarity_to_official_final"]]
                .rename(columns={"similarity_to_official_final": "similarity"})
                .copy()
            )

        if not role_similarity_mc_df.empty:
            order_df = (
                role_similarity_mc_df.groupby("role", as_index=False)["similarity"].mean()
                .sort_values("similarity", ascending=False)
            )
            ordered_roles = list(order_df["role"].values)
            n_rows = len(ordered_roles)
            fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(3, 2 * n_rows)), sharex=True)
            if n_rows == 1:
                axes = [axes]
            for ax, role_name in zip(axes, ordered_roles):
                vals = (
                    role_similarity_mc_df.loc[role_similarity_mc_df["role"] == role_name, "similarity"]
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
                ax.set_xlim(0.8, 1.0)
                ax.set_ylabel(role_name)
                if baseline_fed is not None:
                    ax.axvline(baseline_fed, linestyle="--", color="black", linewidth=1.2)
                if baseline_occ is not None:
                    ax.axvline(baseline_occ, linestyle=(0, (5, 5)), color="dimgray", linewidth=1.2)
                if baseline_fdic is not None:
                    ax.axvline(baseline_fdic, linestyle=(0, (3, 3)), color="gray", linewidth=1.2)

            legend_lines: List[Line2D] = []
            legend_labels: List[str] = []
            if baseline_fed is not None:
                legend_lines.append(Line2D([0], [0], color="black", lw=1.2, linestyle="--"))
                legend_labels.append("Draft FRS")
            if baseline_occ is not None:
                legend_lines.append(Line2D([0], [0], color="dimgray", lw=1.2, linestyle=(0, (5, 5))))
                legend_labels.append("Draft OCC")
            if baseline_fdic is not None:
                legend_lines.append(Line2D([0], [0], color="gray", lw=1.2, linestyle=(0, (3, 3))))
                legend_labels.append("Draft FDIC")
            axes[-1].set_xlabel("Cosine similarity to official final")
            if legend_lines:
                fig.legend(legend_lines, legend_labels, title="Draft baselines", loc="upper right")
            plt.tight_layout()
            STACKED_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(STACKED_PNG_PATH, dpi=200)
            plt.savefig(STACKED_SVG_PATH)
            plt.close()

            role_groups = {
                rn: role_similarity_mc_df.loc[role_similarity_mc_df["role"] == rn, "similarity"].astype(float).dropna().values
                for rn in roles_to_run
            }
            r = len(roles_to_run)
            t_mat = np.zeros((r, r), dtype=float)
            for i, ri in enumerate(roles_to_run):
                xi = role_groups.get(ri, np.array([], dtype=float))
                for j, rj in enumerate(roles_to_run):
                    if i == j:
                        t_mat[i, j] = 0.0
                        continue
                    xj = role_groups.get(rj, np.array([], dtype=float))
                    if len(xi) >= 2 and len(xj) >= 2:
                        t_stat, _ = ttest_ind(xi, xj, equal_var=False, nan_policy="omit")
                        t_mat[i, j] = float(t_stat) if np.isfinite(t_stat) else np.nan
                    else:
                        t_mat[i, j] = np.nan
            t_df = pd.DataFrame(t_mat, index=roles_to_run, columns=roles_to_run)
            T_MATRIX_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            t_df.to_csv(T_MATRIX_CSV_PATH)
            plt.figure(figsize=(max(6, int(0.6 * r)), max(4, int(0.6 * r))))
            cat_mat = np.where(
                np.isnan(t_mat),
                np.nan,
                np.where(t_mat > 1.645, 1.0, np.where(t_mat < -1.645, -1.0, 0.0)),
            )
            cat_df = pd.DataFrame(cat_mat, index=roles_to_run, columns=roles_to_run)
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
            plt.title("Welch t-statistics across roles (similarity to official final)")
            plt.tight_layout()
            plt.savefig(T_MATRIX_PNG_PATH, dpi=200)
            plt.savefig(T_MATRIX_SVG_PATH)
            plt.close()

        seeds_list: List[int] = list(range(seed_start, seed_start + num_seeds))
        acceptance_rates_records_cache: List[Dict[str, object]] = []
        for seed_val in seeds_list:
            acc_path = Path(f"outputs/s31_comment_acceptance_all_roles_seed_{seed_val}.csv")
            if not acc_path.exists():
                continue
            acc_df_seed = pd.read_csv(acc_path)
            ar_seed = (
                acc_df_seed.groupby("role", as_index=False)["accepted"].mean()
                .rename(columns={"accepted": "acceptance_rate"})
            )
            ar_seed["seed"] = int(seed_val)
            for _, r in ar_seed.iterrows():
                acceptance_rates_records_cache.append(
                    {
                        "role": str(r["role"]),
                        "seed": int(r["seed"]),
                        "acceptance_rate": float(r["acceptance_rate"]),
                    }
                )
        if acceptance_rates_records_cache:
            acc_df = pd.DataFrame.from_records(acceptance_rates_records_cache)
            acc_summary = (
                acc_df.groupby("role", as_index=False)
                .agg(mean=("acceptance_rate", "mean"), std=("acceptance_rate", "std"), n=("seed", "count"))
            )
            acc_summary["se"] = acc_summary["std"] / acc_summary["n"].clip(lower=1).pow(0.5)
            acc_summary["ci_95_margin"] = 1.96 * acc_summary["se"].fillna(0.0)
            acc_summary = acc_summary.sort_values("mean", ascending=False)
            ACCEPTANCE_RATES_MC_CSV.parent.mkdir(parents=True, exist_ok=True)
            acc_summary.to_csv(ACCEPTANCE_RATES_MC_CSV, index=False)
            plt.figure(figsize=(10, max(4, int(0.6 * len(acc_summary)))))
            ax = sns.barplot(
                data=acc_summary,
                y="role",
                x="mean",
                color="#8b8b8b",
                orient="h",
                errorbar=None,
            )
            for i, (_, row) in enumerate(acc_summary.iterrows()):
                ax.errorbar(
                    x=float(row["mean"]),
                    y=i,
                    xerr=float(row["ci_95_margin"]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=3,
                )
            ax.set_xlim(0.8, 1.0)
            ax.set_xlabel("Acceptance rate (mean across seeds) with 95% CI")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(ACCEPTANCE_RATES_MC_PNG, dpi=200)
            plt.savefig(ACCEPTANCE_RATES_MC_SVG)
            plt.close()

        if not role_similarity_mc_df.empty:
            sim_summary = (
                role_similarity_mc_df.groupby("role", as_index=False)
                .agg(mean=("similarity", "mean"), std=("similarity", "std"), n=("seed", "count"))
            )
            sim_summary["se"] = sim_summary["std"] / sim_summary["n"].clip(lower=1).pow(0.5)
            sim_summary["ci_95_margin"] = 1.96 * sim_summary["se"].fillna(0.0)
            sim_summary = sim_summary.sort_values("mean", ascending=False)
            SIMILARITY_RATES_MC_CSV.parent.mkdir(parents=True, exist_ok=True)
            sim_summary.to_csv(SIMILARITY_RATES_MC_CSV, index=False)
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
            ax.set_xlim(0.8, 1.0)
            ax.set_xlabel("Similarity to official final (mean across seeds) with 95% CI")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(SIMILARITY_RATES_MC_PNG, dpi=200)
            plt.savefig(SIMILARITY_RATES_MC_SVG)
            plt.close()

        if SIM_MATRIX_AVG_CSV.exists():
            sim_df_avg_cached = pd.read_csv(SIM_MATRIX_AVG_CSV, index_col=0)
            HEATMAP_AVG_PNG.parent.mkdir(parents=True, exist_ok=True)
            n_labels_cached = sim_df_avg_cached.shape[0]
            fig_w = max(6, int(0.6 * n_labels_cached))
            fig_h = max(4, int(0.6 * n_labels_cached))
            plt.figure(figsize=(fig_w, fig_h))
            vmin_val = float(np.nanmin(sim_df_avg_cached.values)) if np.isfinite(np.nanmin(sim_df_avg_cached.values)) else 0.0
            vmax_val = float(np.nanmax(sim_df_avg_cached.values)) if np.isfinite(np.nanmax(sim_df_avg_cached.values)) else 1.0
            sns.heatmap(sim_df_avg_cached, vmin=vmin_val, vmax=vmax_val, cmap="viridis", annot=True, fmt=".2f")
            plt.title("Average Cosine Similarity Between Policies Across Seeds (s31)")
            plt.tight_layout()
            plt.savefig(HEATMAP_AVG_PNG, dpi=200)
            plt.savefig(HEATMAP_AVG_SVG)
            plt.close()

        print(f"Saved stacked similarity distributions → {STACKED_PNG_PATH}, {STACKED_SVG_PATH}")
        if T_MATRIX_CSV_PATH.exists():
            print(f"Saved t-statistics matrix → {T_MATRIX_CSV_PATH}")
            print(f"Saved t-statistics heatmap → {T_MATRIX_PNG_PATH}, {T_MATRIX_SVG_PATH}")
        if ACCEPTANCE_RATES_MC_CSV.exists():
            print(f"Saved acceptance rates with CIs → {ACCEPTANCE_RATES_MC_CSV}")
            print(f"Saved acceptance rates plot → {ACCEPTANCE_RATES_MC_PNG}, {ACCEPTANCE_RATES_MC_SVG}")
        if SIM_MATRIX_AVG_CSV.exists():
            print(f"Saved average heatmap → {HEATMAP_AVG_PNG}, {HEATMAP_AVG_SVG}")
        return

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    client = configure_openai_client()
    ref_vecs = embed_texts([original_final, original_draft, occ_draft_text, fdic_draft_text])
    ref_vec_true_final = ref_vecs[0]
    ref_vec_orig_draft = ref_vecs[1]
    ref_vec_occ_draft = ref_vecs[2]
    ref_vec_fdic_draft = ref_vecs[3]

    all_trace_records: List[Dict[str, object]] = []
    all_similarity_records: List[Dict[str, object]] = []

    labels: List[str] = [
        "official_final",
        "draft_fed",
        "draft_occ",
        "draft_fdic",
    ] + [f"final_{rn}" for rn in roles_to_run]
    n_labels = len(labels)
    sum_matrix = np.zeros((n_labels, n_labels), dtype=float)
    count_matrix = np.zeros((n_labels, n_labels), dtype=int)

    def run_acceptance_for_role(role_name: str, seed_val: int) -> List[Dict[str, object]]:
        role_text = ROLES[role_name]
        acceptance_cache_path = Path(f"outputs/s31_acceptance_cache_seed_{seed_val}.json")
        if acceptance_cache_path.exists():
            acceptance_cache: Dict[str, Dict[str, object]] = json.loads(
                acceptance_cache_path.read_text(encoding="utf-8")
            )
        else:
            acceptance_cache = {}
        acceptance_cache_lock = threading.Lock()

        def compute_accept_cache_key(*, model: str, comment_id: str, comment_text: str) -> str:
            sha = __import__("hashlib").sha256()
            sha.update((role_name + "|" + model + "|" + comment_id + "|").encode("utf-8"))
            sha.update(comment_text.encode("utf-8", errors="ignore"))
            return sha.hexdigest()

        def process_comment(cm: Dict[str, str]) -> Dict[str, object]:
            cm_id = str(cm["id"]).strip()
            cm_text = str(cm["text"]).strip()
            cache_key = compute_accept_cache_key(
                model=MODEL_NAME,
                comment_id=cm_id,
                comment_text=cm_text,
            )
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
                with acceptance_cache_lock:
                    acceptance_cache[cache_key] = {
                        "role": role_name,
                        "comment_id": cm_id,
                        "accepted": accepted_val,
                        "proposed_change": proposed_change_val,
                    }
            return {
                "role": role_name,
                "comment_id": cm_id,
                "accepted": bool(accepted_val),
                "proposed_change": proposed_change_val,
            }

        records_local: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=accept_workers) as executor:
            futures = [executor.submit(process_comment, cm) for cm in comments_all]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Acceptance [{role_name}]"):
                rec = _.result()
                records_local.append(rec)

        acceptance_cache_path.parent.mkdir(parents=True, exist_ok=True)
        acceptance_cache_path.write_text(
            json.dumps(acceptance_cache, ensure_ascii=False), encoding="utf-8"
        )
        return records_local

    def run_revision_for_role(role_name: str, acceptance_df: pd.DataFrame) -> Tuple[Dict[str, object], Dict[str, object], Optional[List[float]]]:
        role_text = ROLES[role_name]
        role_accept_df = acceptance_df[acceptance_df["role"] == role_name]
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

    seeds_list: List[int] = list(range(seed_start, seed_start + num_seeds))
    seed_to_role_vecs: Dict[int, Dict[str, Optional[List[float]]]] = {s: {} for s in seeds_list}
    acceptance_rates_records: List[Dict[str, object]] = []

    for seed_val in tqdm(seeds_list, desc="Seeds"):
        np.random.seed(seed_val)

        acceptance_records: List[Dict[str, object]] = []
        for rn in roles_to_run:
            acceptance_records.extend(run_acceptance_for_role(rn, seed_val))

        acceptance_by_comment_role_df = pd.DataFrame.from_records(acceptance_records)
        acceptance_csv_path = Path(f"outputs/s31_comment_acceptance_all_roles_seed_{seed_val}.csv")
        acceptance_csv_path.parent.mkdir(parents=True, exist_ok=True)
        acceptance_by_comment_role_df.to_csv(acceptance_csv_path, index=False)

        ar_seed = (
            acceptance_by_comment_role_df.groupby("role", as_index=False)["accepted"].mean()
            .rename(columns={"accepted": "acceptance_rate"})
        )
        ar_seed["seed"] = int(seed_val)
        for _, r in ar_seed.iterrows():
            acceptance_rates_records.append(
                {
                    "role": str(r["role"]),
                    "seed": int(r["seed"]),
                    "acceptance_rate": float(r["acceptance_rate"]),
                }
            )

        for rn in roles_to_run:
            tr, sr, fv = run_revision_for_role(rn, acceptance_by_comment_role_df)
            tr_seeded = dict(tr)
            tr_seeded["seed"] = int(seed_val)
            sr_seeded = dict(sr)
            sr_seeded["seed"] = int(seed_val)
            all_trace_records.append(tr_seeded)
            all_similarity_records.append(sr_seeded)
            seed_to_role_vecs[int(seed_val)][str(rn)] = fv

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
        avg_matrix = np.divide(
            sum_matrix,
            count_matrix,
            out=np.full_like(sum_matrix, np.nan),
            where=count_matrix > 0,
        )

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
    plt.title("Average Cosine Similarity Between Policies Across Seeds (s31)")
    plt.tight_layout()
    plt.savefig(HEATMAP_AVG_PNG, dpi=200)
    plt.savefig(HEATMAP_AVG_SVG)
    plt.close()

    role_similarity_mc_df = (
        all_sim_df[
            (all_sim_df["role"].isin(roles_to_run)) & (all_sim_df["seed"].notna())
        ][["role", "seed", "similarity_to_official_final"]]
        .rename(columns={"similarity_to_official_final": "similarity"})
        .copy()
    )

    if not role_similarity_mc_df.empty:
        STACKED_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)

        order_df = (
            role_similarity_mc_df.groupby("role", as_index=False)["similarity"].mean()
            .sort_values("similarity", ascending=False)
        )
        ordered_roles = list(order_df["role"].values)

        n_rows = len(ordered_roles)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(3, 2 * n_rows)), sharex=True)
        if n_rows == 1:
            axes = [axes]

        baseline_fed = float(base_sim_record["similarity_to_official_final"]) if base_sim_record["similarity_to_official_final"] is not None else None
        baseline_occ = float(occ_sim_record["similarity_to_official_final"]) if occ_sim_record["similarity_to_official_final"] is not None else None
        baseline_fdic = float(fdic_sim_record["similarity_to_official_final"]) if fdic_sim_record["similarity_to_official_final"] is not None else None

        for ax, role_name in zip(axes, ordered_roles):
            vals = (
                role_similarity_mc_df.loc[
                    role_similarity_mc_df["role"] == role_name, "similarity"
                ]
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
            ax.set_xlim(0.8, 1.0)
            ax.set_ylabel(role_name)
            if baseline_fed is not None:
                ax.axvline(baseline_fed, linestyle="--", color="black", linewidth=1.2)
            if baseline_occ is not None:
                ax.axvline(baseline_occ, linestyle=(0, (5, 5)), color="dimgray", linewidth=1.2)
            if baseline_fdic is not None:
                ax.axvline(baseline_fdic, linestyle=(0, (3, 3)), color="gray", linewidth=1.2)

        legend_lines: List[Line2D] = []
        legend_labels: List[str] = []
        if baseline_fed is not None:
            legend_lines.append(Line2D([0], [0], color="black", lw=1.2, linestyle="--"))
            legend_labels.append("Draft FRS")
        if baseline_occ is not None:
            legend_lines.append(Line2D([0], [0], color="dimgray", lw=1.2, linestyle=(0, (5, 5))))
            legend_labels.append("Draft OCC")
        if baseline_fdic is not None:
            legend_lines.append(Line2D([0], [0], color="gray", lw=1.2, linestyle=(0, (3, 3))))
            legend_labels.append("Draft FDIC")
        axes[-1].set_xlabel("Cosine similarity to official final")
        if legend_lines:
            fig.legend(legend_lines, legend_labels, title="Draft baselines", loc="upper right")
        plt.tight_layout()
        plt.savefig(STACKED_PNG_PATH, dpi=200)
        plt.savefig(STACKED_SVG_PATH)
        plt.close()

        sim_summary = (
            role_similarity_mc_df.groupby("role", as_index=False)
            .agg(mean=("similarity", "mean"), std=("similarity", "std"), n=("seed", "count"))
        )
        sim_summary["se"] = sim_summary["std"] / sim_summary["n"].clip(lower=1).pow(0.5)
        sim_summary["ci_95_margin"] = 1.96 * sim_summary["se"].fillna(0.0)
        sim_summary = sim_summary.sort_values("mean", ascending=False)
        SIMILARITY_RATES_MC_CSV.parent.mkdir(parents=True, exist_ok=True)
        sim_summary.to_csv(SIMILARITY_RATES_MC_CSV, index=False)

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
        ax.set_xlim(0.8, 1.0)
        ax.set_xlabel("Similarity to official final (mean across seeds) with 95% CI")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(SIMILARITY_RATES_MC_PNG, dpi=200)
        plt.savefig(SIMILARITY_RATES_MC_SVG)
        plt.close()

    if not role_similarity_mc_df.empty:
        role_groups = {
            rn: role_similarity_mc_df.loc[role_similarity_mc_df["role"] == rn, "similarity"].astype(float).dropna().values
            for rn in roles_to_run
        }
        r = len(roles_to_run)
        t_mat = np.zeros((r, r), dtype=float)
        for i, ri in enumerate(roles_to_run):
            xi = role_groups.get(ri, np.array([], dtype=float))
            for j, rj in enumerate(roles_to_run):
                if i == j:
                    t_mat[i, j] = 0.0
                    continue
                xj = role_groups.get(rj, np.array([], dtype=float))
                if len(xi) >= 2 and len(xj) >= 2:
                    t_stat, _ = ttest_ind(xi, xj, equal_var=False, nan_policy="omit")
                    t_mat[i, j] = float(t_stat) if np.isfinite(t_stat) else np.nan
                else:
                    t_mat[i, j] = np.nan
        t_df = pd.DataFrame(t_mat, index=roles_to_run, columns=roles_to_run)
        T_MATRIX_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        t_df.to_csv(T_MATRIX_CSV_PATH)

        plt.figure(figsize=(max(6, int(0.6 * r)), max(4, int(0.6 * r))))
        cat_mat = np.where(
            np.isnan(t_mat),
            np.nan,
            np.where(t_mat > 1.645, 1.0, np.where(t_mat < -1.645, -1.0, 0.0)),
        )
        cat_df = pd.DataFrame(cat_mat, index=roles_to_run, columns=roles_to_run)
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
        plt.title("Welch t-statistics across roles (similarity to official final)")
        plt.tight_layout()
        T_MATRIX_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(T_MATRIX_PNG_PATH, dpi=200)
        plt.savefig(T_MATRIX_SVG_PATH)
        plt.close()

    if acceptance_rates_records:
        acc_df = pd.DataFrame.from_records(acceptance_rates_records)
        acc_summary = (
            acc_df.groupby("role", as_index=False)
            .agg(mean=("acceptance_rate", "mean"), std=("acceptance_rate", "std"), n=("seed", "count"))
        )
        acc_summary["se"] = acc_summary["std"] / acc_summary["n"].clip(lower=1).pow(0.5)
        acc_summary["ci_95_margin"] = 1.96 * acc_summary["se"].fillna(0.0)
        acc_summary = acc_summary.sort_values("mean", ascending=False)

        ACCEPTANCE_RATES_MC_CSV.parent.mkdir(parents=True, exist_ok=True)
        acc_summary.to_csv(ACCEPTANCE_RATES_MC_CSV, index=False)

        plt.figure(figsize=(10, max(4, int(0.6 * len(acc_summary)))))
        ax = sns.barplot(
            data=acc_summary,
            y="role",
            x="mean",
            color="#8b8b8b",
            orient="h",
            errorbar=None,
        )
        for i, (_, row) in enumerate(acc_summary.iterrows()):
            ax.errorbar(
                x=float(row["mean"]),
                y=i,
                xerr=float(row["ci_95_margin"]),
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
        ax.set_xlim(0.8, 1.0)
        ax.set_xlabel("Acceptance rate (mean across seeds) with 95% CI")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(ACCEPTANCE_RATES_MC_PNG, dpi=200)
        plt.savefig(ACCEPTANCE_RATES_MC_SVG)
        plt.close()

    print(f"Saved consolidated trace → {TRACE_XLSX_PATH}")
    print(f"Saved consolidated similarities → {SIM_XLSX_PATH}")
    print(f"Saved average policy similarity matrix → {SIM_MATRIX_AVG_CSV}")
    print(f"Saved average heatmap → {HEATMAP_AVG_PNG}, {HEATMAP_AVG_SVG}")
    print(f"Saved stacked similarity distributions → {STACKED_PNG_PATH}, {STACKED_SVG_PATH}")
    if T_MATRIX_CSV_PATH.exists():
        print(f"Saved t-statistics matrix → {T_MATRIX_CSV_PATH}")
        print(f"Saved t-statistics heatmap → {T_MATRIX_PNG_PATH}, {T_MATRIX_SVG_PATH}")
    if ACCEPTANCE_RATES_MC_CSV.exists():
        print(f"Saved acceptance rates with CIs → {ACCEPTANCE_RATES_MC_CSV}")
        print(f"Saved acceptance rates plot → {ACCEPTANCE_RATES_MC_PNG}, {ACCEPTANCE_RATES_MC_SVG}")

if __name__ == "__main__":
    main()
