#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the GoEmotions conversion CLI.
    
    Parameters:
        None
    
    Returns:
        args (argparse.Namespace): Parsed arguments with attributes:
            - input (str): Path to the GoEmotions dataset (Hugging Face download).
            - output (str): Destination path for the output Parquet file.
    """
    parser = argparse.ArgumentParser(description="Convert GoEmotions to unified parquet schema")
    parser.add_argument("--input", required=True, help="Path to GoEmotions dataset (Hugging Face download)")
    parser.add_argument("--output", required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point that parses command-line arguments and orchestrates conversion of the GoEmotions dataset to a unified Parquet schema.
    
    Parses `--input` and `--output` arguments, converts them to Path objects, and runs the conversion producing a Parquet file matching the repository's root data/schema. Currently raises NotImplementedError until the conversion logic is implemented.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Implement conversion logic producing a parquet matching root data/schema
    raise NotImplementedError("Implement GoEmotions -> parquet conversion")


if __name__ == "__main__":
    main()