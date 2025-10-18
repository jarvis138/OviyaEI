#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for converting MELD data to the unified parquet schema.
    
    This reads and validates the required CLI options for the conversion tool.
    
    Parameters:
        None
    
    Returns:
        argparse.Namespace: Namespace containing:
            - input (str): Path to MELD.zip or an extracted MELD directory.
            - output (str): Output parquet file or directory path.
    """
    parser = argparse.ArgumentParser(description="Convert MELD to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to MELD.zip or extracted directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    Run the CLI that prepares input and output paths for converting MELD to a unified parquet schema.
    
    Parses command-line arguments, converts the provided input and output values to Path objects, and hands them to the MELD-to-parquet conversion routine (the conversion is currently unimplemented).
    
    Raises:
        NotImplementedError: always raised until the MELD-to-parquet conversion is implemented.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement MELD -> parquet conversion")


if __name__ == "__main__":
    main()