import argparse
from pathlib import Path
import torch
import torchaudio


def main():
    """
    Tokenize all WAV files under an input directory into Mimi tokens and write them to an output directory.
    
    Parses CLI arguments --input (required, path to normalized WAV root), --out (required, output tokens root), and --batch-size (optional). For each .wav file under --input, loads audio, resamples to 24000 Hz if needed, mixes channels to mono, encodes the waveform with a pretrained Mimi model ("kyutai/mimi"), and saves the resulting tokens as a .pt file in the output directory using the input file stem.
    
    Raises:
        RuntimeError: if the Mimi library cannot be imported or if a produced token object is not a dict containing the 'acoustic' key.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="normalized wav root (24k mono)")
    ap.add_argument("--out", required=True, help="output tokens root")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    try:
        from mimi import Mimi
    except Exception as e:
        raise RuntimeError("Mimi not installed or unavailable") from e

    model = Mimi.from_pretrained("kyutai/mimi")
    in_root = Path(args.input)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    files = list(in_root.rglob("*.wav"))
    for f in files:
        wav, sr = torchaudio.load(str(f))
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.mean(dim=0)
        tokens = model.encode(wav)
        out_path = out_root / (f.stem + ".pt")
        torch.save(tokens, out_path)
        # simple smoke: ensure it has acoustic codebooks
        if not isinstance(tokens, dict) or 'acoustic' not in tokens:
            raise RuntimeError("Unexpected Mimi token format")
    print(f"[✓] Tokenized {len(files)} files → {out_root}")


if __name__ == "__main__":
    main()

