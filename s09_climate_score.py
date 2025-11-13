#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import openai
import pandas as pd
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
import instructor


import os

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

client = instructor.patch(openai.OpenAI(api_key=openai.api_key))

COMMENT_DIRS = [
    Path("data/comments/occ/txt"),
    Path("data/comments/fdic/txt"),
    Path("data/comments/fed/txt"),
]
DOWNLOAD_LINKS_FILE = Path("data/metadata/download_links.csv")
MODEL_NAME = "gpt-4o-2024-08-06"
OUTPUT_FILE = "outputs/comment_climate_engagement_scores.xlsx"
MAX_WORKERS = 1


class CommentClassification(BaseModel):
    """Data model for the classified comment."""

    score: int | None = Field(
        ...,
        description="The climate policy engagement score from 1 to 5. Null if not applicable.",
    )
    explanation: str = Field(
        ...,
        description="A brief explanation for the score, using quotes from the comment.",
    )
    author_person_name: str | None = Field(
        ...,
        description=(
            "The full name of the individual who signed the comment. "
            "Null if not applicable or if the comment is from an organization without a specific signatory."
        ),
    )
    author_organization_name: str | None = Field(
        ...,
        description=(
            "The name of the organization the author represents. "
            "Null if the author is a private individual not representing an organization."
        ),
    )
    author_type: Literal["private individual", "professional on behalf of an organization", "academic"] | None = (
        Field(..., description="The classification of the author.")
    )
    author_organization_type: (
        Literal["climate action group", "banking association", "government agency", "bank", "other"] | None
    ) = Field(
        ...,
        description="The classification of the organization. Null if no organization.",
    )
    author_full_address: str | None = Field(..., description="Full mailing address if provided.")
    author_state: str | None = Field(
        ...,
        description="Two-letter U.S. state abbreviation. Null if outside U.S. or not provided.",
    )
    author_country: str | None = Field(
        ...,
        description="Country of the author. Default to 'USA' if a state is provided. Null if not provided.",
    )


class OrganizationTypeClassification(BaseModel):
    """Data model for organization type classification."""

    author_organization_type_metadata: (
        Literal["climate action group", "banking association", "government agency", "bank", "other"] | None
    ) = Field(
        ...,
        description="Classification of the organization type based on the organization name or government agency field.",
    )


def classify_comment(comment: str, model: str = MODEL_NAME) -> dict:
    """Classify a comment using an LLM.
    
    Extracts a score, explanation, and author metadata.
    
    Args:
        comment: The text of the comment to classify.
        model: The name of the OpenAI model to use.
        
    Returns:
        A dictionary containing the classification results.
    """
    prompt = (
        """You will receive a comment on a policy proposal addressed to a U.S. regulator.
Your task is to analyze the comment and return a structured JSON object with two parts: a climate policy engagement score and author metadata.

First, assign a climate policy engagement score from 1 to 5 according to the definitions below, and provide a brief explanation that uses quotes from the comment.
Score definitions:
1 = Strong opposition to climate action by the regulator. Explicitly resists climate measures. May deny climate or climate risks.
2 = Skeptical or hesitant. Questions the need for special treatment or warns about costs and unintended consequences.
3 = Neutral. Takes no strong position for or against climate action.
4 = Supportive. Backs climate actions of the regulator. May support other climate measures. May advocate for more incremental steps.
5 = Strong advocate. Fully supports ambitious, binding climate targets and broad reforms. May seek to strengthen proposed initiatives.

Second, extract the following metadata about the author:
- author_person_name: The full name of the individual who signed the comment. Null if not applicable or if the comment is from an organization without a specific signatory.
- author_organization_name: The name of the organization the author represents. Null if the author is a private individual not representing an organization.
- author_type: Classify the author as one of: "private individual", "professional on behalf of an organization", "academic".
- author_organization_type: If an organization is named, classify it as one of: "climate action group", "banking association", "government agency", "bank", "other". Null if no organization.
- author_full_address: The full mailing address, if provided.
- author_state: The two-letter U.S. state abbreviation (e.g., "NY", "CA"). Null if the address is outside the U.S. or not provided.
- author_country: The author's country. Default to "USA" if a U.S. state is present. Null if no address is provided.

[COMMENT]\n"""
        + comment
    )
    
    for attempt in range(3):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_model=CommentClassification,
        )

        result = resp.model_dump()

        if result.get("score") is None:
            result["score"] = "na"
        return result
    
    raise RuntimeError("Failed to classify comment after 3 attempts")


def classify_organization_type(org_name: str, gov_agency: str, model: str = MODEL_NAME) -> dict[str, str | None]:
    """Classify organization type based on organization name or government agency field.
    
    Args:
        org_name: The organization name from the CSV.
        gov_agency: The government agency field from the CSV.
        model: The name of the OpenAI model to use.
        
    Returns:
        A dictionary containing the organization type classification.
    """
    org_text = org_name if pd.notna(org_name) and org_name.strip() else gov_agency

    if pd.isna(org_text) or not org_text.strip():
        return {"author_organization_type_metadata": None}

    prompt = f"""Classify the following organization into one of these categories:
- "climate action group": Environmental organizations, climate advocacy groups, sustainability organizations
- "banking association": Trade associations representing banks, financial industry groups
- "government agency": Federal, state, or local government agencies, regulatory bodies
- "bank": Individual banks, financial institutions
- "other": Any other type of organization not fitting the above categories

Organization: {org_text}

Return only the classification category."""

    for attempt in range(3):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_model=OrganizationTypeClassification,
        )
        return resp.model_dump()
    
    raise RuntimeError("Failed to classify organization type after 3 attempts")


def process_file(file: Path, metadata_df: pd.DataFrame) -> dict:
    """Read a comment file, classify it, and return a record with metadata.
    
    Args:
        file: Path to the comment file.
        metadata_df: DataFrame containing metadata from CSV.
        
    Returns:
        Dictionary containing classification results and metadata.
    """
    text = file.read_text(encoding="utf-8")

    tracking_number = file.stem

    metadata_row = metadata_df[metadata_df["Tracking Number"] == tracking_number]

    author_organization_type_metadata = None
    author_organization_name_metadata = None
    author_state_metadata = None
    author_country_metadata = None
    first_name_metadata = None
    last_name_metadata = None

    if not metadata_row.empty:
        row = metadata_row.iloc[0]
        org_name = row.get("Organization Name")
        gov_agency = row.get("Government Agency")

        org_classification = classify_organization_type(org_name, gov_agency)
        author_organization_type_metadata = org_classification.get("author_organization_type_metadata")

        author_organization_name_metadata = org_name
        author_state_metadata = row.get("State/Province")
        author_country_metadata = row.get("Country")
        first_name_metadata = row.get("First Name")
        last_name_metadata = row.get("Last Name")

    out = classify_comment(text)

    record = {
        "comment_path": str(file),
        **out,
        "author_organization_type_metadata": author_organization_type_metadata,
        "author_organization_name_metadata": author_organization_name_metadata,
        "author_state_metadata": author_state_metadata,
        "author_country_metadata": author_country_metadata,
        "first_name_metadata": first_name_metadata,
        "last_name_metadata": last_name_metadata,
    }

    return record


def main() -> None:
    """Main function to process comments, classify them, save results, and generate a plot."""
    print("Loading metadata from download links CSV...")
    comment_metadata_df = pd.read_csv(DOWNLOAD_LINKS_FILE)
    print(f"Loaded {len(comment_metadata_df)} metadata records")

    files: list[Path] = []
    for d in COMMENT_DIRS:
        files.extend(sorted(d.glob("**/*.txt")))

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(files), desc="Scoring comments") as pbar:
            futures = [executor.submit(process_file, file, comment_metadata_df) for file in files]
            for future in as_completed(futures):
                records.append(future.result())
                pbar.update(1)

    comment_score_df = pd.DataFrame(records)
    comment_score_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Generated {OUTPUT_FILE}")

    def detect_source(path_value: object) -> str:
        text = str(path_value).lower()
        if "comments/occ" in text:
            return "OCC"
        if "comments/fdic" in text:
            return "FDIC"
        if "comments/fed" in text:
            return "FED"
        return "Unknown"

    comment_score_df["source"] = comment_score_df["comment_path"].apply(detect_source)

    comment_score_valid_df = comment_score_df[
        comment_score_df["score"].apply(lambda x: isinstance(x, int))
    ].copy()

    comment_score_valid_df["score"] = pd.to_numeric(comment_score_valid_df["score"])

    plt.figure()

    bins = list(range(1, 7))

    agency_color_pairs = [("OCC", "#1f77b4"), ("FDIC", "#ff7f0e"), ("FED", "#2ca02c")]
    
    for label, color in agency_color_pairs:
        agency_df = comment_score_valid_df[comment_score_valid_df["source"] == label]
        if not agency_df.empty:
            plt.hist(
                agency_df["score"],
                bins=bins,
                alpha=0.6,
                label=label,
                color=color,
            )
    plt.xticks(range(1, 6))
    plt.xlabel("Climate Engagement Score")
    plt.ylabel("Count")
    plt.title("Distribution of Scores: OCC vs. FDIC vs. FED")
    plt.legend()
    plt.tight_layout()
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/score_distribution.png")
    print("Saved plots/score_distribution.png")


if __name__ == "__main__":
    main()
