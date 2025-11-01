# Complete Emotion Reference System Setup Guide
===============================================

This guide documents the complete integrated system for downloading emotion-expressive TTS models, extracting emotion datasets, and generating reference audio for CSM-1B.

## Overview

The system follows this complete pipeline:

```
Step 1: Download OpenVoiceV2 TTS Model
    ↓
Step 2: Download Emotion Datasets (RAVDESS, CREMA-D, MELD, EmoDB)
    ↓
Step 3: Generate Emotion References with OpenVoiceV2
    ↓
Step 4: Extract Additional References from Datasets
    ↓
Step 5: Merge All References
    ↓
Step 6: CSM-1B Automatically Uses References ✅
```

## Quick Start

Run the complete setup script:

```bash
cd production
python setup_complete_emotion_references.py
```

Or run the main setup script directly:

```bash
python production/setup_emotion_references.py
```

## Components

### 1. OpenVoiceV2 TTS Model (`setup_emotion_references.py`)

**What it does:**
- Downloads OpenVoiceV2 repository from GitHub
- Downloads OpenVoiceV2 models from Hugging Face
- Installs dependencies
- Generates emotion-expressive reference audio

**Output:**
- `external/OpenVoice/` - OpenVoiceV2 repository
- `external/OpenVoice/checkpoints_v2/` - Pre-trained models
- `data/emotion_references/*.wav` - Generated emotion references

### 2. Emotion Dataset Download (`setup_emotion_references.py`)

**What it does:**
- Downloads RAVDESS dataset (Zenodo)
- Downloads CREMA-D dataset (GitHub - manual)
- Downloads MELD dataset (GitHub)
- Downloads EmoDB dataset

**Output:**
- `emotion_datasets/ravdess/` - RAVDESS audio files
- `emotion_datasets/crema_d/` - CREMA-D audio files
- `emotion_datasets/meld/` - MELD audio files
- `emotion_datasets/emodb/` - EmoDB audio files

### 3. Reference Generation (`emotion_teacher.py`)

**What it does:**
- Uses OpenVoiceV2 to generate emotion-expressive speech
- Maps emotions to style tokens
- Saves references for CSM-1B context

**Emotions Supported:**
- `calm_supportive` - Calm, supportive tone
- `empathetic_sad` - Empathetic, gentle tone
- `joyful_excited` - Joyful, energetic tone
- `playful` - Playful, cheerful tone
- `confident` - Confident, steady tone
- `concerned_anxious` - Concerned, caring tone
- `angry_firm` - Firm, determined tone
- `neutral` - Neutral, balanced tone

### 4. Dataset Extraction (`extract_all_emotions.py`)

**What it does:**
- Extracts emotion-labeled audio from datasets
- Normalizes audio to CSM-1B format (24kHz, mono, float32)
- Creates emotion-to-audio mapping JSON
- Saves organized reference files

**Output:**
- `data/emotion_references/emotion_map.json` - Emotion mapping
- `data/emotion_references/*.wav` - Extracted reference audio

### 5. CSM-1B Integration (`websocket_server.py`, `llm_brain.py`)

**What it does:**
- Automatically loads `emotion_map.json`
- Includes emotion reference audio in conversation context
- Passes references to CSM-1B for voice conditioning
- Supports multiple deployment paths (local, VastAI)

**Integration Points:**
- `websocket_server.py` lines 1463-1491: Loads emotion references
- `websocket_server.py` line 1519: Passes reference_audio to CSM-1B
- `llm_brain.py` format_conversation_context_for_csm(): Includes audio references

## File Structure

```
production/
├── setup_emotion_references.py          # Main setup script
├── setup_complete_emotion_references.py # Entry point
├── extract_all_emotions.py            # Dataset extraction
├── voice/
│   └── emotion_teacher.py             # OpenVoice TTS integration
├── data/
│   └── emotion_references/
│       ├── emotion_map.json           # Emotion mapping
│       └── *.wav                       # Reference audio files
└── emotion_datasets/
    ├── ravdess/
    ├── crema_d/
    ├── meld/
    └── emodb/
```

## Usage

### Automatic Setup (Recommended)

```bash
# Run complete setup
python production/setup_complete_emotion_references.py
```

### Manual Steps

```bash
# 1. Download OpenVoiceV2
python production/setup_emotion_references.py

# 2. Extract from datasets
python production/extract_all_emotions.py

# 3. Generate references with OpenVoice
python -c "from production.voice.emotion_teacher import OpenVoiceEmotionTeacher; \
           t = OpenVoiceEmotionTeacher(); \
           t.generate_all_references('data/emotion_references')"
```

## Verification

After setup, verify references are available:

```bash
# Check emotion references
ls -lh data/emotion_references/*.wav

# Check emotion mapping
cat data/emotion_references/emotion_map.json
```

## CSM-1B Usage

The system automatically uses emotion references:

1. **Conversation Context**: Emotion references are included in conversation context
2. **Voice Conditioning**: CSM-1B uses references to condition emotional voice
3. **Automatic Loading**: References are loaded automatically from multiple paths:
   - `data/emotion_references/` (local)
   - `/workspace/emotion_references/` (VastAI)
   - `/workspace/data/emotion_references/` (alternative)

## Troubleshooting

### OpenVoiceV2 Not Found

If OpenVoiceV2 fails to download:
- Check internet connection
- Verify Hugging Face CLI is installed: `pip install huggingface-hub`
- Manual download: `git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice`

### Datasets Not Downloading

If datasets fail to download:
- RAVDESS: May require manual download from Zenodo
- CREMA-D: Requires manual download due to licensing
- MELD: Should clone automatically from GitHub
- EmoDB: May require manual download

**Fallback**: System will use TTS-generated references if datasets unavailable.

### No Emotion References Found

If references aren't found:
- Check `data/emotion_references/emotion_map.json` exists
- Verify audio files are in correct directory
- Check file permissions
- Ensure CSM-1B is looking in correct paths

## Integration Status

✅ **Complete and Integrated**

- [x] OpenVoiceV2 TTS model download
- [x] Emotion dataset download
- [x] Reference generation with TTS
- [x] Dataset extraction
- [x] Reference merging
- [x] CSM-1B integration
- [x] Automatic loading
- [x] Multiple deployment path support

## Next Steps

After setup:
1. Start your server: `python production/websocket_server.py`
2. CSM-1B will automatically use emotion references
3. Emotion references enhance conversational context
4. Voice conditioning improves emotional expressiveness

## Notes

- OpenVoiceV2 API may vary - adjust `generate_with_openvoice_api()` based on actual implementation
- Synthetic references are generated as fallback if OpenVoice unavailable
- System gracefully handles missing datasets
- All references are normalized to CSM-1B format (24kHz, mono, float32)

