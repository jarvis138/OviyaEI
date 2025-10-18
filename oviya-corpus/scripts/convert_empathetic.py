#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for converting EmpatheticDialogues to the unified parquet schema.
    
    Recognizes two required options:
    - --input: path to EmpatheticDialogues repository or archive
    - --output: output parquet path
    
    Returns:
        argparse.Namespace: Namespace with `input` and `output` attributes set to the provided paths.
    """
    parser = argparse.ArgumentParser(description="Convert EmpatheticDialogues to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to EmpatheticDialogues repo or archive")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    Run the EmpatheticDialogues -> parquet conversion using command-line arguments.
    
    Parses command-line arguments for input and output paths, prepares filesystem paths, and performs the conversion from the EmpatheticDialogues dataset into the project's unified parquet schema.
    
    Raises:
        NotImplementedError: Conversion logic is not yet implemented; raised with message
            "Implement EmpatheticDialogues -> parquet conversion".
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement EmpatheticDialogues -> parquet conversion")


if __name__ == "__main__":
    main()