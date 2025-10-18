#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for converting RAVDESS archives or directories to a unified parquet schema.
    
    Parameters:
        None
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - input (str): Required path to the RAVDESS archive file or extracted directory.
            - output (str): Required path where the output parquet file should be written.
    """
    parser = argparse.ArgumentParser(description="Convert RAVDESS to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to RAVDESS archive or extracted directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    Run the RAVDESS-to-parquet conversion using command-line arguments.
    
    Parses CLI arguments, resolves the input and output paths, and performs the conversion from the RAVDESS dataset to the project's unified parquet schema.
    
    Raises:
        NotImplementedError: Conversion is not yet implemented.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement RAVDESS -> parquet conversion")


if __name__ == "__main__":
    main()