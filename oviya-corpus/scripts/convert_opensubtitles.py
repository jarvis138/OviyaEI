#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Create and parse command-line arguments for converting OpenSubtitles to a unified parquet schema.
    
    The parser requires two options: `--input` (path to the OpenSubtitles corpus root) and `--output` (destination parquet path).
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes `input` (str) and `output` (str).
    """
    parser = argparse.ArgumentParser(description="Convert OpenSubtitles to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to OpenSubtitles corpus root")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    Run the OpenSubtitles-to-parquet conversion using command-line arguments.
    
    Parses CLI arguments, converts the provided input and output paths to Path objects, and performs the conversion that writes a unified parquet file to the output location.
    
    Raises:
        NotImplementedError: Conversion logic is not implemented.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement OpenSubtitles -> parquet conversion")


if __name__ == "__main__":
    main()