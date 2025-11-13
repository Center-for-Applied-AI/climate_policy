#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv

import pymupdf


NGRAM_SIZE = 25


def count_pages(pdf_path):
    """Return the number of pages in a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file.
        
    Returns:
        Number of pages in the PDF.
    """
    doc = pymupdf.open(pdf_path)
    try:
        count = doc.page_count
    finally:
        doc.close()
    return count


def extract_text(pdf_path):
    """Extract full text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file.
        
    Returns:
        Extracted text as a single string.
    """
    doc = pymupdf.open(pdf_path)
    texts = []
    try:
        for page in doc:
            texts.append(page.get_text())
    finally:
        doc.close()
    return "\n".join(texts)


def generate_ngrams(words, n):
    """Yield n-word tuples (n-grams) from a list of words.
    
    Args:
        words: List of words.
        n: Size of n-grams.
        
    Yields:
        Tuples of n consecutive words.
    """
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i+n])


def main():
    """Main entry point for PDF processing."""
    parser = argparse.ArgumentParser(
        description="Count pages, extract text, and detect overlaps among PDF files"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="data/comments/fdic/pdf",
        help="Directory to search for PDF files"
    )
    parser.add_argument(
        "--parsed-dir",
        default="data/comments/fdic/txt",
        help="Directory to save extracted text files"
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        default="outputs/pdf_page_counts.csv",
        help="Path to output CSV file for page counts"
    )
    parser.add_argument(
        "--overlaps-csv",
        default="outputs/overlaps.csv",
        help="Path to output CSV file for overlaps"
    )
    parser.add_argument(
        "--md-file",
        default="data/temp/initial/fdic.md",
        help="Markdown file to filter overlaps against"
    )
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    md_file_path = Path(args.md_file)
    if md_file_path.exists():
        md_text = md_file_path.read_text(encoding="utf-8")
    else:
        md_text = ""

    records = []
    texts = {}

    directory = Path(args.directory)
    for pdf_file in directory.rglob("*.pdf"):
        pages = count_pages(str(pdf_file))
        records.append({"file_path": str(pdf_file), "len_pages": pages})

        full_text = extract_text(str(pdf_file))
        txt_name = pdf_file.stem + ".txt"
        txt_path = parsed_dir / txt_name
        txt_path.write_text(full_text, encoding="utf-8")

        words = full_text.split()
        texts[str(pdf_file)] = words

    records.sort(key=lambda rec: rec["len_pages"], reverse=True)
    
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file_path", "len_pages"])
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"Saved sorted page counts to \"{args.output_csv}\"")

    ngram_map = {}
    for pdf_path, words in texts.items():
        seen = set()
        for ng in generate_ngrams(words, NGRAM_SIZE):
            if ng in seen:
                continue
            seen.add(ng)
            ngram_map.setdefault(ng, set()).add(pdf_path)

    raw_overlaps = []
    for ng, paths in ngram_map.items():
        if len(paths) > 1:
            seq_text = " ".join(ng)
            plist = list(paths)
            for i in range(len(plist)):
                for j in range(i+1, len(plist)):
                    raw_overlaps.append((plist[i], plist[j], seq_text))

    filtered = [o for o in raw_overlaps if o[2] not in md_text]

    overlaps_csv_path = Path(args.overlaps_csv)
    overlaps_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with overlaps_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file_path_1", "file_path_2", "overlapped_text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for f1, f2, seq in filtered:
            writer.writerow({"file_path_1": f1, "file_path_2": f2, "overlapped_text": seq})
    print(f"Saved {len(filtered)} filtered overlaps to \"{args.overlaps_csv}\"")


if __name__ == "__main__":
    main()
