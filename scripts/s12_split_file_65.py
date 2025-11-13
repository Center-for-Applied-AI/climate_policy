#!/usr/bin/env python3
from pathlib import Path


SRC_PATH = Path(
    "data/comments/fdic/txt/"
    "2022-statement-principles-climate-related-financial-risk-management-3064-za32-c-065.txt"
)
DELIMITER = "Mr. James P. Sheesley, Assistant Executive Secretary"


def main() -> None:
    """Split a specific FDIC comment file by delimiter."""
    text = SRC_PATH.read_text(encoding="utf-8")

    parts = text.split(DELIMITER)

    if parts and not parts[0].strip():
        parts = parts[1:]

    parts = [f"{DELIMITER}{part}" for part in parts]

    stem = SRC_PATH.stem
    suffix = SRC_PATH.suffix
    parentdir = SRC_PATH.parent

    for i, chunk in enumerate(parts):
        outfile = parentdir / f"{stem}_{i}{suffix}"
        outfile.write_text(chunk.lstrip("\n"), encoding="utf-8")

    print(f"Split into {len(parts)} files in {parentdir}")


if __name__ == "__main__":
    main()
