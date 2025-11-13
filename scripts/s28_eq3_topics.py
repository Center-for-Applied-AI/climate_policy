#!/usr/bin/env python3
"""Augment EQ3 regressions with significant topic dummies."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Tuple

import pandas as pd
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

SEED = 42
DATA_FILE = Path("outputs/before_after_comment_similarity_embeddings.xlsx")
TOPICS_FILE = Path("outputs/comment_topics.xlsx")
TOPICS_SHEET = "topics_dummies"
TOPIC_OLS_FILE = Path("outputs/ols_topics_results.xlsx")
OUT_FILE = Path("outputs/regression_eq3_topics.xlsx")
CLUSTER_VAR = "paragraph_num"

def load_eq3_base() -> pd.DataFrame:
    """Load similarity table baseline with identifiers and controls.

    Returns:
        pd.DataFrame: Rows at paragraph level containing outcome and identifiers.
    """
    df = pd.read_excel(DATA_FILE)
    df = df.loc[:, [
        "y_after_gt_before",
        "paragraph_num",
        "comment_idx",
        "agency",
        "comment_path",
    ]].copy()
    df = df.dropna(subset=["y_after_gt_before"])
    df["y_after_gt_before"] = df["y_after_gt_before"].astype(int)
    df["comment_path"] = df["comment_path"].astype(str)
    return df

def load_topics_dummies() -> pd.DataFrame:
    """Load topic dummy matrix keyed by comment_path.

    Returns:
        pd.DataFrame: One row per comment with dummy columns for each topic.
    """
    df = pd.read_excel(TOPICS_FILE, sheet_name=TOPICS_SHEET)
    df["comment_path"] = df["comment_path"].astype(str)

    cols = [c for c in df.columns if c not in {"score"}]
    df = df.loc[:, cols].copy()
    return df

def load_significant_topics() -> Tuple[List[str], Dict[str, str]]:
    """Identify significant topics (p < 0.05) and retrieve topic renaming map.

    Returns:
        Tuple[List[str], Dict[str, str]]: (significant topic names after renaming, rename_map)
    """
    ols_df = pd.read_excel(TOPIC_OLS_FILE, sheet_name="ols_summary")
    rename_df = pd.read_excel(TOPIC_OLS_FILE, sheet_name="topic_renaming")

    p_columns = [c for c in ols_df.columns if str(c).strip().lower() in {"p>|t|", "pvalue", "p"}]
    if not p_columns:
        raise ValueError("Could not locate p-value column in ols_summary. Expected one of: P>|t|, pvalue, p")
    p_col = p_columns[0]

    var_col = "variable" if "variable" in ols_df.columns else ols_df.columns[0]
    sig = ols_df[(ols_df[var_col] != "const") & (ols_df[p_col] < 0.05)].copy()
    sig_topics: List[str] = sig[var_col].astype(str).tolist()

    if {"original_topic", "renamed_topic"}.issubset(set(rename_df.columns)):
        rename_map = dict(
            zip(rename_df["original_topic"].astype(str), rename_df["renamed_topic"].astype(str))
        )
    else:
        rename_map = {}

    return sig_topics, rename_map

def apply_topic_renaming(dummies_df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """Apply topic renaming and collapse duplicates to match s17's modeling frame.

    Args:
        dummies_df: Wide topic dummies keyed by comment_path.
        rename_map: Mapping from original_topic to renamed_topic.

    Returns:
        pd.DataFrame: Dummies with columns renamed and duplicates collapsed (clipped to 1).
    """
    if not rename_map:
        return dummies_df

    id_cols = ["comment_path"]
    topic_cols = [c for c in dummies_df.columns if c not in id_cols]
    renamed = dummies_df.rename(columns={k: v for k, v in rename_map.items() if k in topic_cols})
    id_part = renamed[id_cols].copy()
    topics_part = renamed.drop(columns=id_cols)

    if topics_part.shape[1] == 0:
        return renamed

    collapsed = topics_part.T.groupby(level=0).sum().T.clip(upper=1)
    result = pd.concat([id_part, collapsed], axis=1)
    return result

def make_specs(rhs_vars_expr: str) -> List[Dict[str, object]]:
    """Create exactly three specifications as requested:

    (1) Agency FE = No, Paragraph FE = No
    (2) Agency FE = Yes, Paragraph FE = No
    (3) Agency FE = Yes, Paragraph FE = Yes

    Comment FE is set to No for all three.
    """
    specs: List[Dict[str, object]] = []

    rhs1 = f"1 + {rhs_vars_expr}"
    specs.append({
        "label": "None",
        "formula": f"y_after_gt_before ~ {rhs1}",
        "par_fe": False,
        "com_fe": False,
    })

    rhs2 = f"1 + {rhs_vars_expr} + C(agency)"
    specs.append({
        "label": "Agency",
        "formula": f"y_after_gt_before ~ {rhs2}",
        "par_fe": False,
        "com_fe": False,
    })

    rhs3 = f"1 + {rhs_vars_expr} + C(agency) + C(paragraph_num)"
    specs.append({
        "label": "Agency",
        "formula": f"y_after_gt_before ~ {rhs3}",
        "par_fe": True,
        "com_fe": False,
    })

    return specs

def star(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def run_ols_multi(formula: str, data: pd.DataFrame, var_names: List[str], clustered: bool):
    """Fit OLS with HC1 or paragraph-clustered SEs and return results for multiple variables.

    Returns:
        Tuple[Dict[str, Tuple[float, float, float]], float, int]:
            mapping var -> (coef, se, pval), R², nobs
    """
    fit_kw = (
        {"cov_type": "cluster", "cov_kwds": {"groups": data[CLUSTER_VAR]}}
        if clustered
        else {"cov_type": "HC1"}
    )
    try:
        fit = smf.ols(formula, data=data).fit(**fit_kw)
        out: Dict[str, Tuple[float, float, float]] = {}
        for v in var_names:
            out[v] = (
                float(fit.params.get(v, float("nan"))),
                float(fit.bse.get(v, float("nan"))),
                float(fit.pvalues.get(v, float("nan"))),
            )
        r2 = float(fit.rsquared)
        nobs = int(getattr(fit, "nobs", len(data)))
    except Exception as e:
        print(f"OLS failed for clustered={clustered}: {e}")
        out = {v: (float("nan"), float("nan"), float("nan")) for v in var_names}
        r2 = float("nan")
        nobs = len(data)
    return out, r2, nobs

def run_ols_cluster_two_multi(formula: str, data: pd.DataFrame, var_names: List[str]):
    try:
        fit_kw = {
            "cov_type": "cluster",
            "cov_kwds": {"groups": [data[CLUSTER_VAR], data["comment_idx"]]},
        }
        fit = smf.ols(formula, data=data).fit(**fit_kw)
        out: Dict[str, Tuple[float, float, float]] = {}
        for v in var_names:
            out[v] = (
                float(fit.params.get(v, float("nan"))),
                float(fit.bse.get(v, float("nan"))),
                float(fit.pvalues.get(v, float("nan"))),
            )
        r2 = float(fit.rsquared)
        nobs = int(getattr(fit, "nobs", len(data)))
    except Exception as e:
        print(f"OLS two-way cluster failed: {e}")
        out = {v: (float("nan"), float("nan"), float("nan")) for v in var_names}
        r2 = float("nan")
        nobs = len(data)
    return out, r2, nobs

def run_logit_multi(formula: str, data: pd.DataFrame, var_names: List[str], se_kind: str):
    if se_kind == "hc1":
        fit_kw = {"cov_type": "HC1", "disp": False, "method": "lbfgs", "maxiter": 100}
    elif se_kind == "cluster_par":
        fit_kw = {
            "cov_type": "cluster",
            "cov_kwds": {"groups": data[CLUSTER_VAR]},
            "disp": False,
            "method": "lbfgs",
            "maxiter": 100,
        }
    else:
        fit_kw = {
            "cov_type": "cluster",
            "cov_kwds": {"groups": [data[CLUSTER_VAR], data["comment_idx"]]},
            "disp": False,
            "method": "lbfgs",
            "maxiter": 100,
        }
    try:
        fit = smf.logit(formula, data=data).fit(**fit_kw)
        out: Dict[str, Tuple[float, float, float]] = {}
        for v in var_names:
            out[v] = (
                float(fit.params.get(v, float("nan"))),
                float(fit.bse.get(v, float("nan"))),
                float(fit.pvalues.get(v, float("nan"))),
            )
        r2 = float(getattr(fit, "prsquared", float("nan")))
        nobs = int(getattr(fit, "nobs", len(data)))
    except Exception as e:
        print(f"Logit failed with se_kind={se_kind}: {e}")
        out = {v: (float("nan"), float("nan"), float("nan")) for v in var_names}
        r2 = float("nan")
        nobs = len(data)
    return out, r2, nobs

def build_blank_multi(var_labels: List[str], kind: str, ncols: int) -> pd.DataFrame:
    rows: List[str] = []
    for lbl in var_labels:
        rows.append(lbl)
        rows.append(f"(se) {lbl}")
    rows.extend([
        "Controls",
        "Paragraph FE",
        "Comment FE",
        "SE type",
        "Observations",
        "R-squared" if kind == "ols" else "Pseudo R²",
    ])
    return pd.DataFrame(index=rows, columns=[f"({i})" for i in range(1, ncols + 1)])

def main() -> None:
    df_base = load_eq3_base()
    topics_df = load_topics_dummies()
    sig_topics, rename_map = load_significant_topics()
    topics_df = apply_topic_renaming(topics_df, rename_map)

    merged = df_base.merge(topics_df, on="comment_path", how="left")

    topic_columns = [c for c in merged.columns if c not in {
        "y_after_gt_before",
        "paragraph_num",
        "comment_idx",
        "agency",
        "comment_path",
    }]

    present_sig_topics = [t for t in sig_topics if t in topic_columns]
    if not present_sig_topics:
        raise ValueError("No significant topics found in merged frame. Check inputs and renaming map.")

    def to_safe_var(name: str, used: set) -> str:
        base = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        if re.match(r"^[0-9]", base):
            base = f"t_{base}"
        if base == "":
            base = "t_var"
        candidate = base
        idx = 2
        while candidate in used:
            candidate = f"{base}_{idx}"
            idx += 1
        used.add(candidate)
        return candidate

    used_safe: set = set()
    topic_to_safe: Dict[str, str] = {t: to_safe_var(t, used_safe) for t in present_sig_topics}
    merged = merged.rename(columns=topic_to_safe)
    safe_vars: List[str] = [topic_to_safe[t] for t in present_sig_topics]
    display_labels: List[str] = present_sig_topics

    se_variants = [("Cluster (paragraph & comment)", "cluster_par_com")]

    rhs_expr = " + ".join(safe_vars)
    specs = make_specs(rhs_expr)
    ncols = len(specs) * len(se_variants)
    tables = {"ols": build_blank_multi(display_labels, "ols", ncols)}

    def run_with_se(formula: str, data: pd.DataFrame, se_kind: str):

        return run_ols_cluster_two_multi(formula, data, safe_vars)

    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xl:
        col_counter = 1
        for spec in tqdm(specs, desc="Eq.3 all topics (3 specs)"):
            for se_label, se_kind in se_variants:
                col = f"({col_counter})"
                col_counter += 1
                results_map, r2, nobs = run_with_se(spec["formula"], merged, se_kind)
                for disp_lbl, safe_var in zip(display_labels, safe_vars):
                    coef, se, pval = results_map.get(safe_var, (float("nan"), float("nan"), float("nan")))
                    tables["ols"].at[disp_lbl, col] = f"{coef:.3f}{star(pval)}"
                    tables["ols"].at[f"(se) {disp_lbl}", col] = f"({se:.3f})"
                tables["ols"].at["Controls", col] = spec["label"]
                tables["ols"].at["Paragraph FE", col] = "Yes" if spec["par_fe"] else "No"
                tables["ols"].at["Comment FE", col] = "Yes" if spec["com_fe"] else "No"
                tables["ols"].at["SE type", col] = se_label
                tables["ols"].at["Observations", col] = nobs
                tables["ols"].at[tables["ols"].index[-1], col] = f"{r2:.3f}"

        tables["ols"].to_excel(xl, sheet_name="OLS (all topics, 3 specs)")

    def _escape_latex(text: str) -> str:
        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
        )

    def _fmt_int(n: int) -> str:
        try:
            return f"{int(n):,}"
        except Exception:
            return ""

    tex_lines: List[str] = []
    tex_lines.append("\\begin{tabular}{lccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("\\toprule")
    tex_lines.append("  & (1) & (2) & (3) \\")
    tex_lines.append("\\midrule")

    col_names = ["(1)", "(2)", "(3)"]
    for disp_lbl in display_labels:
        row_vals = [tables["ols"].get((disp_lbl, c), tables["ols"].at[disp_lbl, c]) for c in col_names]
        row_vals = ["" if pd.isna(v) else str(v) for v in row_vals]
        tex_lines.append(
            f"{_escape_latex(disp_lbl)} & {row_vals[0]} & {row_vals[1]} & {row_vals[2]} \\")
        se_lbl = f"(se) {disp_lbl}"
        se_vals = [tables["ols"].get((se_lbl, c), tables["ols"].at[se_lbl, c]) for c in col_names]
        se_vals = ["" if pd.isna(v) else str(v) for v in se_vals]
        tex_lines.append(
            f" & {se_vals[0]} & {se_vals[1]} & {se_vals[2]} \\")
        tex_lines.append("&&&\\")

    agency_vals = ["No", "Yes", "Yes"]
    par_vals = ["No", "No", "Yes"]
    tex_lines.append(
        f"Agency FE & {agency_vals[0]} & {agency_vals[1]} & {agency_vals[2]} \\")
    tex_lines.append(
        f"Policy paragraph FE & {par_vals[0]} & {par_vals[1]} & {par_vals[2]} \\")
    tex_lines.append("%Cluster & paragraph \\& comment & paragraph \\& comment & paragraph \\& comment \\")
    tex_lines.append("&&&\\")

    obs_vals = [tables["ols"].at["Observations", c] for c in col_names]
    r2_vals = [tables["ols"].at[tables["ols"].index[-1], c] for c in col_names]
    obs_vals = ["" if pd.isna(v) else _fmt_int(v) for v in obs_vals]
    r2_vals = ["" if pd.isna(v) else str(v) for v in r2_vals]
    tex_lines.append(
        f"Observations & {obs_vals[0]} & {obs_vals[1]} & {obs_vals[2]} \\")
    tex_lines.append(
        f"R-squared & {r2_vals[0]} & {r2_vals[1]} & {r2_vals[2]} \\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")

    OUT_TEX = Path("outputs/regression_eq3_topics_table.tex")
    OUT_TEX.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Saved LaTeX table → {OUT_TEX}")

    print(f"Results saved → {OUT_FILE}")

if __name__ == "__main__":
    main()
