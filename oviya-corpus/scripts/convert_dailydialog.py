#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Create and parse command-line arguments for converting DailyDialog to a unified parquet schema.
    
    Adds two required options:
    - `--input`: path to the DailyDialog archive or an extracted directory.
    - `--output`: path where the output parquet file will be written.
    
    Returns:
        argparse.Namespace: Namespace with `input` and `output` attributes containing the provided path strings.
    """
    parser = argparse.ArgumentParser(description="Convert DailyDialog to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to DailyDialog archive or extracted directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the CLI that parses arguments and runs a DailyDialog-to-Parquet conversion.
    
    Parses command-line options, resolves input and output paths, and performs the conversion producing a Parquet file that matches the repository's root schema.
    
    Raises:
        NotImplementedError: Conversion logic is not yet implemented.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement DailyDialog -> parquet conversion")


if __name__ == "__main__":
    main()