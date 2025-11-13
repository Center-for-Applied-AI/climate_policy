#!/usr/bin/env python3
import csv
from pathlib import Path
import time

import requests
import pdfplumber
import pandas as pd
from tqdm.auto import tqdm


CSV_FILE = Path("data/metadata/download_links.csv")
PDF_DIR = Path("data/comments/occ/pdf")
PARSED_DIR = Path("data/comments/occ/txt")
DOWNLOAD_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.5


def download_pdf(url: str, filepath: Path) -> bool:
    """Download a PDF file from URL and save to filepath.
    
    Args:
        url: URL to download from.
        filepath: Path to save PDF to.
        
    Returns:
        True if successful, False otherwise.
    """
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    return True


def parse_pdf_with_pdfplumber(pdf_path: Path, txt_path: Path) -> bool:
    """Parse PDF using pdfplumber library and save as text.
    
    Args:
        pdf_path: Path to PDF file.
        txt_path: Path to save extracted text.
        
    Returns:
        True if successful, False otherwise.
    """
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    full_text = "\n\n".join(text_parts)
    txt_path.write_text(full_text, encoding="utf-8")
    return True


def get_file_lengths() -> list[dict]:
    """Get file lengths for all processed files.
    
    Returns:
        List of dictionaries with file statistics.
    """
    summary_data = []

    for pdf_file in PDF_DIR.glob("*.pdf"):
        tracking_number = pdf_file.stem
        pdf_size = pdf_file.stat().st_size
        txt_file = PARSED_DIR / f"{tracking_number}.txt"

        if txt_file.exists():
            txt_size = txt_file.stat().st_size
            txt_length = len(txt_file.read_text(encoding="utf-8"))
        else:
            txt_size = 0
            txt_length = 0

        summary_data.append(
            {
                "tracking_number": tracking_number,
                "pdf_size_bytes": pdf_size,
                "txt_size_bytes": txt_size,
                "txt_length_chars": txt_length,
            }
        )

    return summary_data


def generate_summary_excel() -> None:
    """Generate Excel summary with file lengths."""
    print("Generating summary Excel file...")
    summary_data = get_file_lengths()

    if not summary_data:
        print("No files found to summarize")
        return

    file_summary_df = pd.DataFrame(summary_data)
    output_file = Path("outputs/upd_occ_summary.xlsx")
    file_summary_df.to_excel(output_file, index=False)
    print(f"Generated summary: {output_file}")
    print(f"  - Total files processed: {len(summary_data)}")
    print(f"  - Total PDF size: {file_summary_df['pdf_size_bytes'].sum():,} bytes")
    print(f"  - Total text size: {file_summary_df['txt_size_bytes'].sum():,} bytes")
    print(f"  - Total characters: {file_summary_df['txt_length_chars'].sum():,}")


def process_csv() -> None:
    """Process the CSV file and download/parse PDFs."""
    print(f"Reading CSV file: {CSV_FILE}")
    
    downloaded_count = 0
    parsed_count = 0
    failed_downloads = 0
    failed_parses = 0
    
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in tqdm(rows, desc="Processing comments"):
        tracking_number = row.get("Tracking Number", "").strip()
        attachment_files = row.get("Attachment Files", "").strip()
        if not tracking_number or not attachment_files:
            continue

        urls = [url.strip() for url in attachment_files.split(",") if url.strip()]
        for url in urls:
            pdf_path = PDF_DIR / f"{tracking_number}.pdf"
            txt_path = PARSED_DIR / f"{tracking_number}.txt"

            if pdf_path.exists() and txt_path.exists():
                continue

            if not pdf_path.exists():
                download_pdf(url, pdf_path)
                downloaded_count += 1

            parse_pdf_with_pdfplumber(pdf_path, txt_path)
            parsed_count += 1
            
            time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    print(f"\n=== Summary ===")
    print(f"PDFs downloaded: {downloaded_count}")
    print(f"PDFs parsed: {parsed_count}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Failed parses: {failed_parses}")

    generate_summary_excel()


def main() -> None:
    """Main entry point for OCC comment download and parsing."""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting PDF download and parsing process...")
    process_csv()
    print("Process completed!")


if __name__ == "__main__":
    main()
