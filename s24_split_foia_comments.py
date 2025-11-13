#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm


SEED: int = 42
DEFAULT_EXCEL_PATH: Path = Path("data/foia/split_comments.xlsx")
DEFAULT_PDF1_PATH: Path = Path("data/foia/pdfs/FOIA-2025-00979.pdf")
DEFAULT_PDF2_PATH: Path = Path("data/foia/pdfs/FOIA-2025-00979 (2).pdf")
DEFAULT_OUTPUT_DIR: Path = Path("data/comments/fed/pdf")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for command-line options.
    
    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Split FOIA PDFs into per-comment files.")
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to the Excel with split instructions.",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel sheet name or index to read.",
    )
    parser.add_argument(
        "--pdf1-path",
        type=Path,
        default=DEFAULT_PDF1_PATH,
        help="Path to the first FOIA PDF volume.",
    )
    parser.add_argument(
        "--pdf2-path",
        type=Path,
        default=DEFAULT_PDF2_PATH,
        help="Path to the second FOIA PDF volume (continuation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write per-comment PDFs.",
    )
    parser.add_argument(
        "--start-col",
        type=str,
        default=None,
        help="Column name for start page (1-index). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--end-col",
        type=str,
        default=None,
        help="Column name for end page (1-index, inclusive). Auto-detected if omitted or computed from next start.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=None,
        help="Optional column name for comment id. If omitted, ids 1..N will be assigned.",
    )
    parser.add_argument(
        "--page-indexing",
        type=str,
        choices=["one", "zero"],
        default="one",
        help="Whether page numbers in Excel are 1-indexed or 0-indexed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit number of comments to process for quick validation.",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Print Excel schema and first rows, then exit.",
    )
    return parser


def read_split_instructions(excel_path: Path, sheet: str | int) -> pd.DataFrame:
    """Load split instructions from Excel or CSV.
    
    Args:
        excel_path: Path to Excel/CSV file with split metadata.
        sheet: Sheet name or index to read (Excel only).
        
    Returns:
        A pandas DataFrame containing the instructions.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel/CSV not found at {excel_path}")
    suffix = excel_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(excel_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(excel_path, sheet_name=sheet)

    return pd.read_excel(excel_path, sheet_name=sheet)


def normalize_column_name(name: str) -> str:
    """Normalize column name for matching.
    
    Args:
        name: Column name.
        
    Returns:
        Normalized column name.
    """
    return name.strip().lower().replace(" ", "_")


def autodetect_columns(
    df: pd.DataFrame,
    start_col: str | None,
    end_col: str | None,
    id_col: str | None,
    file_col: str | None = None,
) -> tuple[str, str | None, str | None, str | None]:
    """Autodetect relevant column names if not provided.
    
    Args:
        df: The DataFrame containing split metadata.
        start_col: Explicit start column name if provided.
        end_col: Explicit end column name if provided.
        id_col: Explicit id column name if provided.
        file_col: Explicit file column name if provided.
        
    Returns:
        Tuple of (start_col_name, end_col_name_or_None, id_col_name_or_None, file_col_name_or_None).
    """
    normalized_map = {normalize_column_name(c): c for c in df.columns}
    
    start_candidates = [
        "start_page",
        "start",
        "first_page",
        "first",
        "page_start",
        "from",
        "begin",
    ]
    end_candidates = [
        "end_page",
        "end",
        "last_page",
        "last",
        "page_end",
        "to",
        "finish",
    ]
    id_candidates = [
        "id",
        "comment_id",
        "commentid",
        "num",
        "index",
    ]
    file_candidates = [
        "file_id",
        "file",
        "volume",
        "pdf_id",
    ]

    resolved_start = start_col
    if resolved_start is None:
        for cand in start_candidates:
            if cand in normalized_map:
                resolved_start = normalized_map[cand]
                break
        if resolved_start is None:
            for nc, orig in normalized_map.items():
                if "start" in nc or "first" in nc or "from" in nc:
                    resolved_start = orig
                    break

    resolved_end = end_col
    if resolved_end is None:
        for cand in end_candidates:
            if cand in normalized_map:
                resolved_end = normalized_map[cand]
                break
        if resolved_end is None:
            for nc, orig in normalized_map.items():
                if "end" in nc or "last" in nc or "to" in nc:
                    resolved_end = orig
                    break

    resolved_id = id_col
    if resolved_id is None:
        for cand in id_candidates:
            if cand in normalized_map:
                resolved_id = normalized_map[cand]
                break

    resolved_file = file_col
    if resolved_file is None:
        for cand in file_candidates:
            if cand in normalized_map:
                resolved_file = normalized_map[cand]
                break

    if resolved_start is None:
        raise ValueError("Unable to detect start page column. Specify --start-col explicitly.")

    return resolved_start, resolved_end, resolved_id, resolved_file


def compute_end_pages_if_missing_grouped(
    df: pd.DataFrame,
    start_col: str,
    end_col: str | None,
    file_col: str | None,
    total_pages_by_file: dict | None,
) -> tuple[pd.Series, pd.Series]:
    """Return start and end page series; infer end pages when missing, grouped by file.
    
    Args:
        df: Input DataFrame with split metadata.
        start_col: Detected start page column name.
        end_col: Detected end page column name, or None if not present.
        file_col: Column name identifying which PDF volume a row belongs to.
        total_pages_by_file: Mapping from file identifier to total pages for that file.
        
    Returns:
        A pair of (start_pages_series, end_pages_series) aligned to df rows.
    """
    starts = df[start_col].astype("Int64").copy()

    if end_col is not None and end_col in df.columns:
        ends = df[end_col].astype("Int64").copy()
        return starts, ends

    if file_col is None:
        working = df[[start_col]].copy()
        working["__next_start"] = working[start_col].shift(-1)
        inferred_end = working["__next_start"].astype("Int64") - 1
        last_total = None
        if total_pages_by_file:
            last_total = sum(int(v) for v in total_pages_by_file.values())
        if pd.isna(inferred_end.iloc[-1]):
            inferred_end.iloc[-1] = last_total if last_total is not None else starts.max()
        inferred_end = inferred_end.astype("Int64")
        return starts, inferred_end

    ends = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    for file_value, group in df.groupby(file_col, sort=False):
        group_sorted = group.sort_values(start_col)
        next_start = group_sorted[start_col].shift(-1)
        end_inferred = next_start.astype("Int64") - 1
        if len(group_sorted) > 0:
            last_idx = group_sorted.index[-1]
            total_for_file = None
            if total_pages_by_file and file_value in total_pages_by_file:
                total_for_file = int(total_pages_by_file[file_value])
            end_inferred.loc[last_idx] = total_for_file if total_for_file is not None else group_sorted[start_col].max()
        ends.loc[group_sorted.index] = end_inferred.astype("Int64")
    return starts, ends


def validate_ranges(start_pages: pd.Series, end_pages: pd.Series, total_pages: int) -> None:
    """Validate page ranges for consistency and bounds.
    
    Args:
        start_pages: Series of start pages.
        end_pages: Series of end pages.
        total_pages: Total available pages.
    """
    if len(start_pages) != len(end_pages):
        raise ValueError("Start and end series have different lengths")

    if (start_pages.isna() | end_pages.isna()).any():
        raise ValueError("Start or end pages contain missing values")

    if (start_pages <= 0).any() or (end_pages <= 0).any():
        raise ValueError("Page numbers must be positive when 1-indexed")

    if (start_pages > end_pages).any():
        raise ValueError("Some ranges have start page greater than end page")

    if (end_pages > total_pages).any():
        raise ValueError("Some ranges exceed the total number of pages")


def compute_total_pages(pdf1_path: Path, pdf2_path: Path) -> tuple[int, int, int]:
    """Return (pages_pdf1, pages_pdf2, total_pages).
    
    Args:
        pdf1_path: Path to first PDF.
        pdf2_path: Path to second PDF.
        
    Returns:
        Tuple of (pages1, pages2, total).
    """
    reader1 = PdfReader(str(pdf1_path))
    reader2 = PdfReader(str(pdf2_path))
    pages1 = len(reader1.pages)
    pages2 = len(reader2.pages)
    return pages1, pages2, pages1 + pages2


def get_page_from_global_index(
    global_page_zero_index: int,
    reader1: PdfReader,
    reader2: PdfReader,
    pages1: int,
):
    """Get page from global index across two PDFs.
    
    Args:
        global_page_zero_index: Global 0-indexed page number.
        reader1: First PDF reader.
        reader2: Second PDF reader.
        pages1: Number of pages in first PDF.
        
    Returns:
        Page object.
    """
    if global_page_zero_index < pages1:
        return reader1.pages[global_page_zero_index]
    return reader2.pages[global_page_zero_index - pages1]


def split_and_write(
    df: pd.DataFrame,
    start_pages: pd.Series,
    end_pages: pd.Series,
    id_series: pd.Series | None,
    file_series: pd.Series | None,
    page_indexing: str,
    pdf1_path: Path,
    pdf2_path: Path,
    output_dir: Path,
    limit: int | None,
) -> list[Path]:
    """Perform the splitting and write per-comment PDFs.
    
    Args:
        df: DataFrame containing original rows for reference.
        start_pages: Series of start page numbers.
        end_pages: Series of end page numbers.
        id_series: Series containing comment ids; if None, ids are assigned 1..N.
        file_series: Series containing file identifiers.
        page_indexing: "one" if 1-indexed in Excel, "zero" if 0-indexed.
        pdf1_path: Path to first volume.
        pdf2_path: Path to second volume.
        output_dir: Destination directory.
        limit: Optional row limit to process.
        
    Returns:
        List of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    reader1 = PdfReader(str(pdf1_path))
    reader2 = PdfReader(str(pdf2_path))
    pages1 = len(reader1.pages)

    zero_based_offset = 0 if page_indexing == "zero" else 1

    num_rows = len(df)
    if limit is not None:
        num_rows = min(num_rows, limit)

    written_files: list[Path] = []
    iterator = range(num_rows)

    for row_index in tqdm(iterator, desc="Splitting comments", unit="comment"):
        start_page_excel = int(start_pages.iloc[row_index])
        end_page_excel = int(end_pages.iloc[row_index])

        start_zero = start_page_excel - zero_based_offset
        end_zero = end_page_excel - zero_based_offset

        writer = PdfWriter()
        if file_series is not None:
            file_value = file_series.iloc[row_index]
            if int(file_value) == 1:
                for idx in range(start_zero, end_zero + 1):
                    writer.add_page(reader1.pages[idx])
            elif int(file_value) == 2:
                for idx in range(start_zero, end_zero + 1):
                    writer.add_page(reader2.pages[idx])
            else:
                raise ValueError(f"Unsupported file id: {file_value}")
        else:
            for global_zero in range(start_zero, end_zero + 1):
                page = get_page_from_global_index(global_zero, reader1, reader2, pages1)
                writer.add_page(page)

        if id_series is not None:
            comment_id = str(id_series.iloc[row_index])
        else:
            comment_id = str(row_index + 1)

        out_path = output_dir / f"{comment_id}.pdf"
        with out_path.open("wb") as f:
            writer.write(f)
        written_files.append(out_path)

    return written_files


def main(argv: list[str] | None = None) -> int:
    """Main entry point for FOIA PDF splitting.
    
    Args:
        argv: Command-line arguments.
        
    Returns:
        Exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    split_instruction_df = read_split_instructions(args.excel_path, args.sheet)

    if args.peek:
        print("Columns:", list(split_instruction_df.columns))
        print("Shape:", split_instruction_df.shape)
        print(split_instruction_df.head(20).to_string(index=False))
        return 0

    pages1, pages2, total_pages = compute_total_pages(args.pdf1_path, args.pdf2_path)

    start_col, end_col, id_col, file_col = autodetect_columns(
        split_instruction_df, args.start_col, args.end_col, args.id_col
    )

    total_by_file = {1: pages1, 2: pages2}

    starts, ends = compute_end_pages_if_missing_grouped(
        split_instruction_df, start_col, end_col, file_col, total_by_file
    )

    working_df = split_instruction_df.copy()
    working_df["__start"] = starts.astype(int)
    working_df["__end"] = ends.astype(int)
    sort_cols = [c for c in ([file_col] if file_col else [])] + ["__start", "__end"]
    working_df = working_df.sort_values(sort_cols).reset_index(drop=True)

    starts_sorted = working_df["__start"]
    ends_sorted = working_df["__end"]

    total_for_validation = total_pages if file_col is None else max(pages1, pages2)
    validate_ranges(starts_sorted, ends_sorted, total_for_validation)

    id_series = None
    if id_col is not None and id_col in working_df.columns:
        id_series = working_df[id_col]
    file_series = None
    if file_col is not None and file_col in working_df.columns:
        file_series = working_df[file_col]

    written = split_and_write(
        working_df,
        starts_sorted,
        ends_sorted,
        id_series,
        file_series,
        args.page_indexing,
        args.pdf1_path,
        args.pdf2_path,
        args.output_dir,
        args.limit,
    )

    print(f"Written {len(written)} files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
