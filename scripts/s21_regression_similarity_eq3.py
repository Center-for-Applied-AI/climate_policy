#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


DATA_FILE = Path("outputs/before_after_comment_similarity_embeddings.xlsx")
SCORES_FILE = Path("outputs/comment_climate_engagement_scores.xlsx")
OUT_FILE = Path("outputs/regression_influence_summary_embeddings.xlsx")
CLUSTER_VAR = "paragraph_num"


def infer_agency(p: str) -> str:
    """Infer agency from path string.
    
    Args:
        p: Path string.
        
    Returns:
        Agency name (occ, fed, or fdic).
    """
    pl = str(p).lower()
    if "comments/occ" in pl:
        return "occ"
    if "comments/fed" in pl:
        return "fed"
    return "fdic"


def load_data() -> pd.DataFrame:
    """Load similarity table and merge climate engagement scores with fallback by filename+agency.
    
    Returns:
        Merged dataframe with y_after_gt_before, paragraph_num, comment_idx, agency, comment_path, and score.
    """
    paragraph_comment_similarity_df = pd.read_excel(DATA_FILE)
    comment_score_df = pd.read_excel(SCORES_FILE)

    paragraph_comment_similarity_df["comment_file"] = (
        paragraph_comment_similarity_df["comment_path"].astype(str).str.extract(r"([^/]+\.txt)$", expand=False)
    )
    paragraph_comment_similarity_df["comment_agency"] = paragraph_comment_similarity_df["comment_path"].map(
        infer_agency
    )
    
    if "comment_file" not in comment_score_df.columns:
        comment_score_df["comment_file"] = comment_score_df["comment_path"].astype(str).str.extract(
            r"([^/]+\.txt)$", expand=False
        )

    if "comment_agency" not in comment_score_df.columns:
        comment_score_df["comment_agency"] = comment_score_df["comment_path"].map(infer_agency)

    merged = paragraph_comment_similarity_df.merge(
        comment_score_df[["comment_path", "score"]], on="comment_path", how="left"
    )

    miss = merged["score"].isna()
    if miss.any():
        left_fb = paragraph_comment_similarity_df.loc[miss, ["comment_file", "comment_agency"]].copy()
        left_fb["__idx__"] = left_fb.index
        right_fb = comment_score_df[["comment_file", "comment_agency", "score"]].drop_duplicates()
        fb = left_fb.merge(right_fb, on=["comment_file", "comment_agency"], how="left")
        merged.loc[fb["__idx__"], "score"] = fb["score"].values

    miss2 = merged["score"].isna()
    if miss2.any():
        left_fb2 = paragraph_comment_similarity_df.loc[miss2, ["comment_file"]].copy()
        left_fb2["__idx__"] = left_fb2.index
        right_fb2 = comment_score_df[["comment_file", "score"]].drop_duplicates()
        fb2 = left_fb2.merge(right_fb2, on=["comment_file"], how="left")
        merged.loc[fb2["__idx__"], "score"] = fb2["score"].values

    result_columns = [
        "y_after_gt_before",
        "paragraph_num",
        "comment_idx",
        "agency",
        "comment_path",
        "score",
    ]
    paragraph_comment_score_df = merged.loc[:, result_columns].copy()

    paragraph_comment_score_df = paragraph_comment_score_df.dropna(subset=["y_after_gt_before", "score"])
    paragraph_comment_score_df["y_after_gt_before"] = paragraph_comment_score_df["y_after_gt_before"].astype(int)
    return paragraph_comment_score_df


def make_specs() -> list[dict]:
    """Create the cross of requested specifications (controls × FE).
    
    Returns:
        List of specification dictionaries.
    """
    specs: list[dict] = []

    controls: list[tuple[str, str]] = [
        ("None", ""),
        ("Agency", " + C(agency)"),
    ]

    par_fe_opts = [False, True]
    com_fe_opts = [False, True]

    for ctrl_label, ctrl_rhs in controls:
        for par_fe in par_fe_opts:
            for com_fe in com_fe_opts:
                rhs = "1 + score" + ctrl_rhs
                if par_fe:
                    rhs += " + C(paragraph_num)"
                if com_fe:
                    rhs += " + C(comment_idx)"
                specs.append(
                    {
                        "label": ctrl_label,
                        "formula": f"y_after_gt_before ~ {rhs}",
                        "par_fe": par_fe,
                        "com_fe": com_fe,
                    }
                )
    return specs


def star(p: float) -> str:
    """Return significance stars based on p-value.
    
    Args:
        p: P-value.
        
    Returns:
        String with stars (*** for p<0.01, ** for p<0.05, * for p<0.10).
    """
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def run_ols(formula: str, data: pd.DataFrame, clustered: bool) -> tuple[float, float, float, float, int]:
    """Fit OLS with HC1 or paragraph-clustered SEs.
    
    Args:
        formula: Regression formula.
        data: Input dataframe.
        clustered: Whether to use clustered standard errors.
        
    Returns:
        Tuple of (coef, se, p, R², nobs).
    """
    fit_kw = (
        {"cov_type": "cluster", "cov_kwds": {"groups": data[CLUSTER_VAR]}}
        if clustered
        else {"cov_type": "HC1"}
    )
    
    fit = smf.ols(formula, data=data).fit(**fit_kw)
    coef = float(fit.params["score"]) if "score" in fit.params else float("nan")
    se = float(fit.bse["score"]) if "score" in fit.bse else float("nan")
    pval = float(fit.pvalues["score"]) if "score" in fit.pvalues else float("nan")
    r2 = float(fit.rsquared)
    nobs = int(getattr(fit, "nobs", len(data)))
    
    return coef, se, pval, r2, nobs


def build_blank(kind: str, ncols: int) -> pd.DataFrame:
    """Build blank results table.
    
    Args:
        kind: Either "ols" or "logit".
        ncols: Number of columns.
        
    Returns:
        Empty dataframe with appropriate row labels.
    """
    rows = [
        "Climate engagement score",
        "",
        "Controls",
        "Paragraph FE",
        "Comment FE",
        "SE type",
        "Observations",
        "R-squared" if kind == "ols" else "Pseudo R²",
    ]
    return pd.DataFrame(index=rows, columns=[f"({i})" for i in range(1, ncols + 1)])


def main() -> None:
    """Estimate EQ3 regressions relating cosine winners to comment scores."""
    paragraph_comment_score_df = load_data()
    specs = make_specs()

    se_variants = [
        ("HC1", "hc1"),
        ("Cluster (paragraph)", "cluster_par"),
        ("Cluster (paragraph & comment)", "cluster_par_com"),
    ]

    ncols = len(specs) * len(se_variants)
    tables = {
        "ols": build_blank("ols", ncols),
        "logit": build_blank("logit", ncols),
    }

    def run_with_se_kind(
        model: str, formula: str, data: pd.DataFrame, se_kind: str
    ) -> tuple[float, float, float, float, int]:
        if model == "ols":
            if se_kind == "hc1":
                return run_ols(formula, data, clustered=False)
            if se_kind == "cluster_par":
                return run_ols(formula, data, clustered=True)
            if se_kind == "cluster_par_com":
                fit_kw = {
                    "cov_type": "cluster",
                    "cov_kwds": {"groups": [data[CLUSTER_VAR], data["comment_idx"]]},
                }
                
                fit = smf.ols(formula, data=data).fit(**fit_kw)
                coef = float(fit.params.get("score", float("nan")))
                se = float(fit.bse.get("score", float("nan")))
                pval = float(fit.pvalues.get("score", float("nan")))
                r2 = float(fit.rsquared)
                nobs = int(getattr(fit, "nobs", len(data)))
                
                return coef, se, pval, r2, nobs
        else:
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
            
            fit = smf.logit(formula, data=data).fit(**fit_kw)
            coef = float(fit.params.get("score", float("nan")))
            se = float(fit.bse.get("score", float("nan")))
            pval = float(fit.pvalues.get("score", float("nan")))
            r2 = float(getattr(fit, "prsquared", float("nan")))
            nobs = int(getattr(fit, "nobs", len(data)))
            
            return coef, se, pval, r2, nobs
        
        return run_ols(formula, data, clustered=False)

    col_counter = 1
    for spec in specs:
        for se_label, se_kind in se_variants:
            col = f"({col_counter})"
            col_counter += 1
            
            model_table_pairs = [("ols", tables["ols"]), ("logit", tables["logit"])]
            
            for kind, tbl in model_table_pairs:
                coef, se, pval, r2, nobs = run_with_se_kind(kind, spec["formula"], paragraph_comment_score_df, se_kind)
                tbl.at["Climate engagement score", col] = f"{coef:.3f}{star(pval)}"
                tbl.at["", col] = f"({se:.3f})"
                tbl.at["Controls", col] = spec["label"]
                tbl.at["Paragraph FE", col] = "Yes" if spec["par_fe"] else "No"
                tbl.at["Comment FE", col] = "Yes" if spec["com_fe"] else "No"
                tbl.at["SE type", col] = se_label
                tbl.at["Observations", col] = nobs
                tbl.at[tbl.index[-1], col] = f"{r2:.3f}"

    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xl:
        tables["ols"].to_excel(xl, sheet_name="OLS results")
        tables["logit"].to_excel(xl, sheet_name="Logit results")

    print(f"Results saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
