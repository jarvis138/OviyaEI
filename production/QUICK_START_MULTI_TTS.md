# Quick Start Guide - Multi-TTS Emotion References
==================================================

## Run Complete Setup

```bash
cd production
python setup_complete_multi_tts.py
```

## What It Does

1. ✅ Downloads OpenVoiceV2 from Hugging Face
2. ✅ Downloads Coqui TTS (XTTS-v2) via pip
3. ✅ Downloads Bark via pip
4. ✅ Downloads emotion datasets (RAVDESS, MELD, etc.)
5. ✅ Generates emotion references from all TTS models
6. ✅ Extracts references from datasets
7. ✅ Merges everything for CSM-1B

## Output

- `data/emotion_references/emotion_map.json` - Emotion mapping
- `data/emotion_references/*.wav` - Reference audio files

## CSM-1B Integration

References are automatically used by CSM-1B according to:
- https://huggingface.co/sesame/csm-1b
- Format: `{"type": "audio", "audio": audio_array}` ✅

## Verification

```bash
# Check references
ls -lh data/emotion_references/*.wav

# Check mapping
cat data/emotion_references/emotion_map.json | python -m json.tool
```

## Status

✅ **COMPLETE** - All TTS models integrated and ready!

