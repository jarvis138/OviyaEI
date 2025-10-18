#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the IEMOCAP -> parquet conversion tool.
    
    The parser requires two arguments:
    - --input: path to the IEMOCAP archive or an extracted directory
    - --output: destination path for the generated parquet file
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes `input` and `output` as strings.
    """
    parser = argparse.ArgumentParser(description="Convert IEMOCAP to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to IEMOCAP archive or extracted directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point that parses command-line arguments and runs the IEMOCAP -> parquet conversion pipeline.
    
    Parses the --input and --output arguments, converts them to Path objects, and invokes the conversion logic that produces a parquet file matching the repository's root data/schema. Currently the conversion implementation is not provided.
    
    Raises:
        NotImplementedError: Always raised until the IEMOCAP to parquet conversion is implemented.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement IEMOCAP -> parquet conversion")


if __name__ == "__main__":
    main()