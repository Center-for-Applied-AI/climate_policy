#!/usr/bin/env python3
from pathlib import Path
import argparse


def count_symbols_in_file(file_path):
    """Count total number of characters in a text file.
    
    Args:
        file_path: Path to the text file.
        
    Returns:
        Number of characters in the file.
    """
    return len(file_path.read_text(encoding="utf-8"))


def main():
    """Main entry point for symbol counting."""
    parser = argparse.ArgumentParser(
        description="Count symbols in .txt files in a directory"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="data/comments/fdic/txt",
        help="Directory containing .txt files"
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    total_symbols = 0
    
    for txt_file in directory.rglob("*.txt"):
        count = count_symbols_in_file(txt_file)
        print(f"{txt_file}: {count}")
        total_symbols += count

    print(f"Total symbols across all .txt files: {total_symbols}")


if __name__ == "__main__":
    main()
