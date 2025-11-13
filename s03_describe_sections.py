#!/usr/bin/env python3
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import openai


import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

DATA_DIR = Path("data/policies/parsed")
SOURCE_FILES = ["fdic.md", "occ.md", "fed.md"]
REFERENCE_FILE = "final.md"
FIGSIZE = (10, 9)
PAR_LABEL_FONTSIZE = 7
MAX_BAR_HEIGHT = 0.30

PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")


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
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        params = {"model": model, "input": batch}
        if dimensions is not None:
            params["dimensions"] = dimensions
        response = openai.embeddings.create(**params)
        embeddings.extend([r.embedding for r in response.data])
    return embeddings


def read_paragraphs(path: Path) -> list[str]:
    """Return a list of paragraphs from a markdown file.
    
    Ignores Markdown heading lines and splits on blank lines (≥2 newlines).
    
    Args:
        path: Path to markdown file.
        
    Returns:
        List of paragraph strings.
    """
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if not ln.lstrip().startswith("#")]

    text = "\n".join(lines)
    return [p.strip() for p in PARAGRAPH_SPLIT_PATTERN.split(text) if p.strip()]


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


def plot_heat_with_bars(
    mat: np.ndarray,
    src_word_counts: list[int],
    ref_word_counts: list[int],
    title: str,
    out_file: str,
):
    """Create heat-map with marginal bar-plots showing paragraph lengths.
    
    Args:
        mat: Similarity matrix.
        src_word_counts: Word counts for source paragraphs.
        ref_word_counts: Word counts for reference paragraphs.
        title: Plot title.
        out_file: Output file path.
    """
    m, n = mat.shape
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1 - MAX_BAR_HEIGHT, MAX_BAR_HEIGHT],
        height_ratios=[MAX_BAR_HEIGHT, 1 - MAX_BAR_HEIGHT],
        wspace=0.05,
        hspace=0.05,
    )
    fig = plt.figure(figsize=FIGSIZE)

    ax_heat  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_heat)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heat)

    im = ax_heat.imshow(mat, origin="lower", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xlabel("final.md paragraph #")
    ax_heat.set_ylabel("source file paragraph #")
    ax_heat.set_title(title, pad=10)

    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_yticks(np.arange(m))
    ax_heat.set_xticklabels(
        [str(i + 1) for i in range(n)], rotation=90, fontsize=PAR_LABEL_FONTSIZE
    )
    ax_heat.set_yticklabels(
        [str(i + 1) for i in range(m)], fontsize=PAR_LABEL_FONTSIZE
    )

    ax_top.bar(
        np.arange(n),
        ref_word_counts,
        color="grey",
        edgecolor="black",
        linewidth=0.3,
    )
    ax_top.set_xlim(-0.5, n - 0.5)
    ax_top.set_ylim(0, max(ref_word_counts) * 1.1)
    ax_top.axis("off")

    ax_right.barh(
        np.arange(m),
        src_word_counts,
        color="grey",
        edgecolor="black",
        linewidth=0.3,
    )
    ax_right.set_ylim(-0.5, m - 0.5)
    ax_right.set_xlim(0, max(src_word_counts) * 1.1)
    ax_right.axis("off")

    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("cosine similarity", rotation=270, labelpad=15)

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def save_similarity_xlsx(
    mat: np.ndarray,
    src_paragraphs: list[str],
    ref_paragraphs: list[str],
    src_doc_stem: str,
    excel_name: str,
):
    """Flatten the similarity matrix into a table and save as .xlsx.
    
    Args:
        mat: Similarity matrix.
        src_paragraphs: Source paragraph texts.
        ref_paragraphs: Reference paragraph texts.
        src_doc_stem: Source document stem name.
        excel_name: Output Excel file path.
    """
    m, n = mat.shape
    records = [
        {
            f"paragraph_{src_doc_stem}_num": i + 1,
            "paragraph_final_num": j + 1,
            f"paragraph_{src_doc_stem}": src_paragraphs[i],
            "paragraph_final": ref_paragraphs[j],
            "cosine_similarity": mat[i, j],
        }
        for i in range(m)
        for j in range(n)
    ]
    pd.DataFrame.from_records(records).to_excel(excel_name, index=False)


def main():
    """Main entry point for paragraph similarity analysis."""
    ref_paragraphs  = read_paragraphs(DATA_DIR / REFERENCE_FILE)
    ref_word_counts = [len(p.split()) for p in ref_paragraphs]
    ref_embs        = np.array(get_embeddings(ref_paragraphs))

    for src_name in SOURCE_FILES:
        src_path        = DATA_DIR / src_name
        src_stem        = src_path.stem
        src_paragraphs  = read_paragraphs(src_path)
        src_word_counts = [len(p.split()) for p in src_paragraphs]
        src_embs        = np.array(get_embeddings(src_paragraphs))

        sim_mat = cosine_similarity_matrix(src_embs, ref_embs)

        png_name = f"plots/{src_stem}_vs_final_paragraphs.png"
        plot_heat_with_bars(
            sim_mat,
            src_word_counts,
            ref_word_counts,
            title=f"{src_name} ↔ final.md (paragraphs – embeddings)",
            out_file=png_name,
        )

        xlsx_name = f"outputs/{src_stem}_vs_final_paragraphs.xlsx"
        save_similarity_xlsx(
            sim_mat,
            src_paragraphs,
            ref_paragraphs,
            src_stem,
            xlsx_name,
        )

    print("PNG heat-maps and XLSX files written successfully.")


if __name__ == "__main__":
    main()
