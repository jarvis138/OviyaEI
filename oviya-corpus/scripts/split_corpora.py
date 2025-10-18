#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# Splits normalized/ into normalized_commercial/ based on license allowlist.


def parse_args():
    """
    Create and parse CLI arguments for splitting corpora.
    
    Defines `--src` (source directory of parquet files, default "normalized"), `--dst` (destination directory, default "normalized_commercial"), and `--allowlist` (path to a JSON file containing an `allowed` list of parquet basenames, default "data/commercial_allowlist.json").
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes `src`, `dst`, and `allowlist`.
    """
    p = argparse.ArgumentParser(description="Split normalized corpora into commercial-safe subset")
    p.add_argument("--src", default="normalized", help="Source dir of parquet files")
    p.add_argument("--dst", default="normalized_commercial", help="Destination dir")
    p.add_argument(
        "--allowlist",
        default="data/commercial_allowlist.json",
        help="JSON file listing allowed parquet basenames under key 'allowed'",
    )
    return p.parse_args()


def load_allowlist(path: Path) -> set:
    """
    Load allowed filenames from a JSON allowlist file.
    
    Reads the JSON at `path` and extracts the top-level "allowed" list, returning its items as a set. If `path` does not exist, or the "allowed" key is missing, an empty set is returned.
    
    Parameters:
        path (Path): Path to the JSON allowlist file which should contain an "allowed" list of filenames.
    
    Returns:
        set: A set of allowed filename strings (empty if the file is missing or contains no "allowed" entries).
    """
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    allowed = data.get("allowed", [])
    return set(allowed)


def main():
    """
    Copy Parquet files from a source directory to a destination directory, keeping only files whose basenames appear in the allowlist.
    
    Reads command-line arguments (source dir, destination dir, allowlist path), creates the destination directory if needed, loads the allowlist, and copies each '*.parquet' file from the source to the destination when its filename is present in the allowlist.
    """
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    allowlist_path = Path(args.allowlist)

    dst.mkdir(parents=True, exist_ok=True)
    allowed = load_allowlist(allowlist_path)

    for p in src.glob("*.parquet"):
        if p.name in allowed:
            (dst / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()