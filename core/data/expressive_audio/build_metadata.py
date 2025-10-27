import json
from pathlib import Path
import argparse
import torchaudio


def infer_emotion_from_path(fpath: Path) -> str:
    # naive: parent directory name or 'neutral'
    return fpath.parent.name if fpath.parent.name not in ("norm_24k", "raw") else "neutral"


def main():
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
        except FileNotFoundError as e:
            duration = 0.0
        except PermissionError as e:
            duration = 0.0
        except OSError as e:
            duration = 0.0
        except RuntimeError as e:
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



