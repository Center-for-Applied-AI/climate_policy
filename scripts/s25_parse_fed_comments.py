#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SEED: int = 42
USE_MPS_DEFAULT: bool = True
USE_CUDA_DEFAULT: bool = True
MARKER_MODULE_CANDIDATES: list[list[str]] = [
    ["-m", "marker.scripts.convert_single"],
    ["-m", "marker.convert_single"],
    ["-m", "convert_single"],
    ["-m", "marker"],
]


def set_seed(seed: int) -> None:
    """Set global RNG seeds.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def configure_device(enable_mps: bool, enable_cuda: bool) -> None:
    """Configure acceleration device.
    
    Args:
        enable_mps: Whether to attempt Apple MPS acceleration.
        enable_cuda: Whether to attempt CUDA acceleration.
    """
    if enable_cuda and torch.cuda.is_available():
        torch.set_default_device("cuda")
        return
    if enable_mps:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if torch.backends.mps.is_available():
            torch.set_default_device("mps")


def build_jobs(input_dir: Path, output_dir: Path) -> list[dict]:
    """Build conversion jobs from input directory.
    
    Args:
        input_dir: Directory containing source PDF files.
        output_dir: Directory where .txt outputs will be written.
        
    Returns:
        A list of job dictionaries with keys: id, pdf_path, txt_path.
    """
    pdf_paths: list[Path] = sorted(input_dir.glob("*.pdf"))
    jobs: list[dict] = []
    for pdf_path in pdf_paths:
        comment_id = pdf_path.stem
        txt_path = output_dir / f"{comment_id}.txt"
        jobs.append({"id": comment_id, "pdf_path": pdf_path, "txt_path": txt_path})
    return jobs


def create_pdf_converter(config_overrides: dict | None = None):
    """Create a PdfConverter with optional configuration overrides.
    
    Args:
        config_overrides: Configuration overrides to pass to ConfigParser.
        
    Returns:
        Initialized PdfConverter.
    """
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    base_config = {"output_format": "markdown"}
    if config_overrides:
        base_config.update(config_overrides)

    parser = ConfigParser(base_config)
    return PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service(),
    )


def convert_pdf_to_text_python(pdf_path: Path) -> str:
    """Convert a single PDF into text using Marker Python API with fallbacks.
    
    Args:
        pdf_path: Path to the input PDF file.
        
    Returns:
        Extracted markdown text.
    """
    from marker.output import text_from_rendered
    from marker.models import create_model_dict
    from marker.converters.ocr import OCRConverter

    attempts: list[dict] = [
        {},
        {"force_ocr": True},
        {"no_tables": True},
        {"force_ocr": True, "no_tables": True},
    ]
    last_exc: Exception | None = None
    for cfg in attempts:
        converter = create_pdf_converter(cfg)
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)
        return text

    ocr_converter = OCRConverter(artifact_dict=create_model_dict())
    rendered = ocr_converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def run_marker_cli(pdf_path: Path, work_dir: Path) -> Path:
    """Run Marker CLI to convert a single PDF into markdown.
    
    Args:
        pdf_path: Input PDF path.
        work_dir: Directory where Marker will write outputs.
        
    Returns:
        Path to the produced markdown file.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []

    base_args = [
        "--output_dir",
        str(work_dir),
        "--output_format",
        "markdown",
        "--disable_image_extraction",
        str(pdf_path),
    ]

    venv_marker_single = Path(sys.executable).with_name("marker_single")
    exe_candidates: list[list[str]] = []
    if venv_marker_single.exists():
        exe_candidates.append([str(venv_marker_single)])

    for exe in exe_candidates:
        variant_args: list[list[str]] = [
            base_args,
            [*base_args[:-1], "--force_ocr", base_args[-1]],
            [*base_args[:-1], "--no_tables", base_args[-1]],
            [*base_args[:-1], "--force_ocr", "--no_tables", base_args[-1]],
        ]
        for exe_args in variant_args:
            result = subprocess.run(
                [*exe, *exe_args],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            if result.returncode == 0:
                md_candidates = list(work_dir.rglob("*.md"))
                if not md_candidates:
                    errors.append(
                        f"Succeeded with {' '.join(exe)} {' '.join(exe_args)} but no .md produced in {work_dir}"
                    )
                    continue
                return md_candidates[0]
            errors.append(
                f"{' '.join(exe)} {' '.join(exe_args)} failed (code {result.returncode}): {result.stderr.strip()}"
            )

    for launch in MARKER_MODULE_CANDIDATES:
        variant_args = [
            base_args,
            [*base_args[:-1], "--force_ocr", base_args[-1]],
            [*base_args[:-1], "--no_tables", base_args[-1]],
            [*base_args[:-1], "--force_ocr", "--no_tables", base_args[-1]],
        ]
        for exe_args in variant_args:
            cmd = [sys.executable, *launch, *exe_args]
            result = subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                md_candidates = list(work_dir.rglob("*.md"))
                if not md_candidates:
                    errors.append(
                        f"Succeeded with {' '.join(launch)} {' '.join(exe_args)} but no .md produced in {work_dir}"
                    )
                    continue
                return md_candidates[0]
            errors.append(
                f"{' '.join(launch)} {' '.join(exe_args)} failed (code {result.returncode}): {result.stderr.strip()}"
            )
    joined = "\n\n".join(errors)
    raise RuntimeError(f"Marker CLI failed for {pdf_path} with all candidates. Errors:\n{joined}")


def write_text(output_path: Path, text: str) -> None:
    """Write text to a file path.
    
    Args:
        output_path: Destination .txt file path.
        text: Text content to write.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def process_job(job: dict) -> tuple[str, bool, str | None]:
    """Process a single job dict and return status.
    
    Args:
        job: Dictionary with keys id, pdf_path, txt_path.
        
    Returns:
        Tuple of (id, success, error_message_or_none).
    """
    job_id = job["id"]
    pdf_path: Path = job["pdf_path"]
    txt_path: Path = job["txt_path"]
    work_dir = txt_path.parent / f"_{job_id}_marker"
    
    text = convert_pdf_to_text_python(pdf_path)
    write_text(txt_path, text)
    return job_id, True, None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    default_input = Path(__file__).resolve().parent / "data" / "comments" / "fed" / "pdf"
    default_output = Path(__file__).resolve().parent / "data" / "comments" / "fed" / "txt"

    parser = argparse.ArgumentParser(description="Convert FED PDFs to text.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=default_input,
        help="Directory with input PDFs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output,
        help="Directory to write .txt outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of PDFs to process",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already have .txt outputs",
    )
    parser.add_argument(
        "--no_mps",
        action="store_true",
        help="Disable Apple MPS acceleration",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA acceleration",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Deprecated. Parallel execution removed; runs single-worker only.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point to run conversion pipeline."""
    set_seed(SEED)

    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    limit: int | None = args.limit
    skip_existing: bool = args.skip_existing
    use_mps: bool = USE_MPS_DEFAULT and (not args.no_mps)
    use_cuda: bool = USE_CUDA_DEFAULT and (not args.no_cuda)

    output_dir.mkdir(parents=True, exist_ok=True)

    configure_device(use_mps, use_cuda)

    jobs = build_jobs(input_dir=input_dir, output_dir=output_dir)
    if skip_existing:
        jobs = [j for j in jobs if not j["txt_path"].exists()]
    if limit is not None:
        jobs = jobs[: max(0, limit)]

    for job in tqdm(jobs, desc="Converting PDFs", unit="pdf"):
        _ = process_job(job)


if __name__ == "__main__":
    main()
