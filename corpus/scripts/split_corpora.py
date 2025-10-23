#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# Splits normalized/ into normalized_commercial/ based on license allowlist.


def parse_args():
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
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    allowed = data.get("allowed", [])
    return set(allowed)


def main():
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
