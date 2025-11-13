#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


DATA_FILE = "outputs/before_after_comment_similarity_ALL.xlsx"
OUT_FILE = "outputs/regression_influence_summary.xlsx"
CLUSTER_PAR = "paragraph_num"
CLUSTER_COMM = "comment_idx"


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


def run(model: str, formula: str, data: pd.DataFrame, se_kind: str) -> tuple[float, float, float, float]:
    """Run regression and return coefficient, standard error, p-value, and fit statistic.
    
    Args:
        model: Either "ols" or "logit".
        formula: Regression formula string.
        data: Input dataframe.
        se_kind: Standard error type (HC1, cluster_par, cluster_par_com).
        
    Returns:
        Tuple of (coefficient, standard_error, p_value, fit_statistic).
    """
    if model == "ols":
        mod = smf.ols
        if se_kind == "HC1":
            fit_kw = {"cov_type": "HC1"}
        elif se_kind == "cluster_par":
            fit_kw = {"cov_type": "cluster", "cov_kwds": {"groups": data[CLUSTER_PAR]}}
        else:
            fit_kw = {"cov_type": "cluster", "cov_kwds": {"groups": [data[CLUSTER_PAR], data[CLUSTER_COMM]]}}
    else:
        mod = smf.logit
        base = {"disp": False, "method": "lbfgs", "maxiter": 100}
        if se_kind == "HC1":
            fit_kw = {"cov_type": "HC1", **base}
        elif se_kind == "cluster_par":
            fit_kw = {"cov_type": "cluster", "cov_kwds": {"groups": data[CLUSTER_PAR]}, **base}
        else:
            fit_kw = {"cov_type": "cluster", "cov_kwds": {"groups": [data[CLUSTER_PAR], data[CLUSTER_COMM]]}, **base}

    fit = mod(formula, data=data).fit(**fit_kw)
    coef = fit.params.get("score", float("nan"))
    se = fit.bse.get("score", float("nan")) if hasattr(fit, "bse") else float("nan")
    pval = fit.pvalues.get("score", float("nan")) if hasattr(fit, "pvalues") else float("nan")
    fit_stat = fit.rsquared if model == "ols" else getattr(fit, "prsquared", float("nan"))
    
    return coef, se, pval, fit_stat


def build_blank(kind: str, ncols: int) -> pd.DataFrame:
    """Build blank results table.
    
    Args:
        kind: Either "ols" or "logit".
        ncols: Number of columns.
        
    Returns:
        Empty dataframe with appropriate row labels.
    """
    rows = [
        "Climate score",
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
    """Run regressions linking GPT influence labels to engagement scores."""
    paragraph_comment_influence_df = pd.read_excel(DATA_FILE)
    paragraph_comment_influence_df["author_type"] = paragraph_comment_influence_df["author_type"].fillna("Unknown")
    paragraph_comment_influence_df["author_organization_type"] = paragraph_comment_influence_df[
        "author_organization_type"
    ].fillna("Unknown")

    paragraph_comment_influence_df = paragraph_comment_influence_df.assign(
        gpt_influenced=lambda d: d["gpt_influenced"].map({"yes": 1, "no": 0}), score=lambda d: d["score"]
    )

    specs: list[dict] = [
        {"label": "None", "formula": "gpt_influenced ~ 1 + score", "par_fe": False, "agency": False},
        {"label": "Agency", "formula": "gpt_influenced ~ 1 + score + C(agency)", "par_fe": False, "agency": True},
        {
            "label": "Paragraph FE",
            "formula": "gpt_influenced ~ 1 + score + C(paragraph_num)",
            "par_fe": True,
            "agency": False,
        },
        {
            "label": "Agency + Paragraph FE",
            "formula": "gpt_influenced ~ 1 + score + C(agency) + C(paragraph_num)",
            "par_fe": True,
            "agency": True,
        },
    ]

    ncols = len(specs) * 3
    tables = {"ols": build_blank("ols", ncols), "logit": build_blank("logit", ncols)}

    se_configs = [
        ("HC1", "HC1"),
        ("cluster_par", "Cluster (paragraph)"),
        ("cluster_par_com", "Cluster 2W (par, com)"),
    ]

    for j, spec in enumerate(specs):
        for k, (se_key, se_label) in enumerate(se_configs, start=1):
            col = f"({3*j + k})"
            for kind in ("ols", "logit"):
                coef, se, pval, fitstat = run(kind, spec["formula"], paragraph_comment_influence_df, se_key)
                tbl = tables[kind]

                tbl.at["Climate score", col] = f"{coef:.3f}{star(pval)}"
                tbl.at["", col] = f"({se:.3f})"
                tbl.at["Controls", col] = spec["label"]
                tbl.at["Paragraph FE", col] = "Yes" if spec["par_fe"] else "No"
                tbl.at["Comment FE", col] = "No"
                tbl.at["SE type", col] = se_label
                tbl.at["Observations", col] = len(paragraph_comment_influence_df)
                tbl.at[tbl.index[-1], col] = f"{fitstat:.3f}"

    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xl:
        tables["ols"].to_excel(xl, sheet_name="OLS results")
        tables["logit"].to_excel(xl, sheet_name="Logit results")

    print(f"\nResults (HC1, paragraph-clustered, two-way-clustered) saved to {OUT_FILE}")
    print("\nOLS preview:\n", tables["ols"].iloc[:5, :6])
    print("\nLogit preview:\n", tables["logit"].iloc[:5, :6])


if __name__ == "__main__":
    main()
