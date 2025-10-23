#!/usr/bin/env bash
set -euo pipefail

# Minimal sample downloader for expressive datasets (demo-sized)
# Downloads small subsets where possible and normalizes to 24 kHz mono WAV.

ROOT_DIR=${1:-"$(pwd)/data/expressive_audio"}
OUT_RAW="$ROOT_DIR/raw"
OUT_NORM="$ROOT_DIR/norm_24k"

mkdir -p "$OUT_RAW" "$OUT_NORM"

echo "[+] Downloading sample ESD (if available via HF public link)"
echo "    Please ensure you have rights to download these datasets for your use case."

# This script uses placeholder samples; replace with your dataset links/commands.

SAMPLE_URL="https://files.freemusicarchive.org/storage-freemusicarchive-org/tracks/SC8d9a0G1T2bQ6QmGJQj3tX9J4p2cYdKQmN2cEwR.mp3"
curl -L "$SAMPLE_URL" -o "$OUT_RAW/sample1.mp3" || true

echo "[+] Normalizing to 24k mono"
if command -v ffmpeg >/dev/null 2>&1; then
  for f in "$OUT_RAW"/*; do
    base=$(basename "$f")
    ffmpeg -y -i "$f" -ac 1 -ar 24000 "$OUT_NORM/${base%.*}.wav" >/dev/null 2>&1 || true
  done
else
  echo "[!] ffmpeg not found; please install it for normalization"
fi

echo "[✓] Done. Normalized files at: $OUT_NORM"




