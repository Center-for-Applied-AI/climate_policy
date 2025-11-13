#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

import pandas as pd
from tqdm.auto import tqdm
import openai
import instructor
from pydantic import BaseModel, Field


COMMENT_DIRS = [
    Path("data/comments/occ/txt"),
    Path("data/comments/fdic/txt"),
]
OUTPUT_FILE = Path("outputs/comments_id_and_content.csv")
ENCODING = "utf-8"
MODEL_NAME = "gpt-4o-2024-08-06"
MAX_WORKERS = 4
SEED = 42

import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

CLIENT = instructor.patch(openai.OpenAI(api_key=openai.api_key))

PARAGRAPH_BREAK_PATTERN = re.compile(r"(\r?\n){2,}")
NEWLINE_PATTERN = re.compile(r"(\r?\n)+")


class CommentSummary(BaseModel):
    """Data model for the LLM summary of a comment."""

    summary: str = Field(..., description="Summary of the comment, up to 200 words.")


def clean_comment_text(text: str) -> str:
    """Remove single newlines within sentences, preserving paragraph breaks.
    
    Args:
        text: The original comment text.
        
    Returns:
        Cleaned comment text with in-sentence newlines replaced by spaces, paragraph breaks preserved.
    """
    text = PARAGRAPH_BREAK_PATTERN.sub("<PARA>", text)
    text = NEWLINE_PATTERN.sub(" ", text)
    text = text.replace("<PARA>", "\n")
    return text.strip()


def summarize_comment(comment: str, model: str = MODEL_NAME) -> str:
    """Summarize a comment using an LLM.
    
    Returns a summary up to 200 words.
    
    Args:
        comment: The text of the comment to summarize.
        model: The name of the OpenAI model to use.
        
    Returns:
        The summary string (up to 200 words).
    """
    prompt = (
        "You will receive a public comment on a policy proposal. "
        "Summarize the main points and arguments of the comment in up to 200 words. "
        "Be concise, accurate, and neutral."
        "\n\n[COMMENT]\n" + comment
    )
    
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=CommentSummary,
        seed=SEED,
    )
    return resp.summary


def process_file(file: Path) -> dict:
    """Read a comment file, extract file id, cleaned content, and LLM summary.
    
    Args:
        file: Path to the comment file.
        
    Returns:
        Dictionary with file_id, comment, and summary.
    """
    content = file.read_text(encoding=ENCODING)
    cleaned_content = clean_comment_text(content)
    file_id = file.stem
    summary = summarize_comment(cleaned_content)
    return {"file_id": file_id, "comment": cleaned_content, "summary": summary}


def main() -> None:
    """Extract file id, cleaned comment content, and LLM summary from all txt files."""
    files = []
    for d in COMMENT_DIRS:
        files.extend(sorted(d.glob("**/*.txt")))

    data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(files), desc="Extracting and summarizing comments") as pbar:
            futures = [executor.submit(process_file, file) for file in files]
            for future in as_completed(futures):
                data.append(future.result())
                pbar.update(1)

    comment_summary_df = pd.DataFrame(data)
    comment_summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
