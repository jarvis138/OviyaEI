#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for converting XED data to a unified parquet schema.
    
    Specifies two required options:
    - --input: path to the XED repository or archive
    - --output: destination path for the generated parquet file
    
    Returns:
        argparse.Namespace: Namespace with `input` and `output` attributes as string paths.
    """
    parser = argparse.ArgumentParser(description="Convert XED to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to XED repo or archive")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    Run the conversion from an XED repository or archive to a unified Parquet dataset at the specified output path.
    
    Parses command-line arguments (`--input`, `--output`), resolves them to filesystem paths, and performs the conversion producing a Parquet matching the root data/schema.
    
    Raises:
        NotImplementedError: Conversion logic is not yet implemented.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement XED -> parquet conversion")


if __name__ == "__main__":
    main()