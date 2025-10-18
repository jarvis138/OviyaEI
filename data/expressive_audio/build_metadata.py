import json
from pathlib import Path
import argparse
import torchaudio


def infer_emotion_from_path(fpath: Path) -> str:
    # naive: parent directory name or 'neutral'
    """
    Infer the emotion label from an audio file path.
    
    Parameters:
        fpath (Path): Path to the audio file whose parent directory name encodes the emotion.
    
    Returns:
        emotion (str): The parent directory name, or "neutral" if the parent is "norm_24k" or "raw".
    """
    return fpath.parent.name if fpath.parent.name not in ("norm_24k", "raw") else "neutral"


def main():
    """
    Scan a directory of WAV files and write a JSON file containing per-file metadata.
    
    Recursively walks the directory provided via the --input argument, collects metadata for each `.wav` file, and writes a JSON array to the path provided via --out. Each item in the array is an object with the following keys:
    - "path": string path to the WAV file
    - "emotion": inferred emotion (parent directory name, with "norm_24k" and "raw" mapped to "neutral")
    - "culture": "en_us"
    - "duration": duration in seconds (0.0 if audio info could not be read)
    - "speaker": parent directory name
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="normalized wav root (24k mono)")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    root = Path(args.input)
    items = []
    for f in root.rglob("*.wav"):
        try:
            info = torchaudio.info(str(f))
            duration = info.num_frames / float(info.sample_rate)
        except Exception:
            duration = 0.0
        items.append({
            "path": str(f),
            "emotion": infer_emotion_from_path(f),
            "culture": "en_us",
            "duration": duration,
            "speaker": f.parent.name
        })

    with open(args.out, "w") as fp:
        json.dump(items, fp)
    print(f"[✓] Wrote metadata: {args.out} ({len(items)} items)")


if __name__ == "__main__":
    main()

