# 🎨 Expanded Emotion Library - Implementation Guide

## Overview

The Oviya EI system now supports **28 emotions** across 3 tiers, providing nuanced emotional expression through a hybrid multi-teacher approach.

---

## 📊 Emotion Library Structure

### Tier 1: Core Emotions (70% usage)
**10 emotions for everyday interactions:**
- `calm_supportive` - Gentle, reassuring tone
- `empathetic_sad` - Understanding when user is down
- `joyful_excited` - Happy, energetic responses
- `confident` - Self-assured, strong
- `neutral` - Baseline, informative
- `comforting` - Soothing, caring
- `encouraging` - Motivating, supportive
- `thoughtful` - Contemplative, reflective
- `affectionate` - Warm, loving
- `reassuring` - Calming, protective

### Tier 2: Contextual Emotions (25% usage)
**9 emotions for specific situations:**
- `playful` - Fun, lighthearted
- `concerned_anxious` - Worried, cautious
- `melancholy` - Deeply sad, somber
- `wistful` - Nostalgic, longing
- `tired` - Fatigued, low energy
- `curious` - Inquisitive, interested
- `dreamy` - Ethereal, peaceful
- `relieved` - Grateful, unburdened
- `proud` - Accomplished, pleased

### Tier 3: Expressive Emotions (5% usage)
**9 emotions for dramatic moments:**
- `angry_firm` - Strong, assertive
- `sarcastic` - Playfully mocking
- `mischievous` - Playful with intent
- `tender` - Delicate, gentle
- `amused` - Entertained, laughing
- `sympathetic` - Deeply understanding
- `reflective` - Looking back, analyzing
- `grateful` - Thankful, appreciative
- `apologetic` - Sorry, regretful

---

## 🔧 Implementation Components

### 1. Emotion Blender (`voice/emotion_blender.py`)
Creates new emotions by interpolating existing ones:

```python
from voice.emotion_blender import EmotionBlender

blender = EmotionBlender()
blender.load_base_embeddings()      # Load 8 base emotions
blender.blend_embeddings()           # Create 20 new emotions
blender.save_blended_embeddings()    # Save to disk
```

**Blend Recipes:**
- `comforting` = 60% calm_supportive + 40% empathetic_sad
- `encouraging` = 50% joyful + 30% confident + 20% calm
- `melancholy` = 80% sad + 20% calm
- *(See `emotion_blender.py` for full recipes)*

### 2. Emotion Library Manager (`voice/emotion_library.py`)
Handles emotion selection, aliases, and validation:

```python
from voice.emotion_library import get_emotion_library

library = get_emotion_library()

# Resolve aliases
emotion = library.get_emotion("happy")  # Returns "joyful_excited"

# Sample weighted emotions
random_emotion = library.sample_emotion()  # Tier-weighted sampling

# Get similar emotions
similar = library.get_similar_emotions("calm_supportive")
```

**Emotion Aliases:**
- `happy` → `joyful_excited`
- `sad` → `empathetic_sad`
- `worried` → `concerned_anxious`
- *(See `emotion_library.py` for complete list)*

### 3. CSM Server (`scripts/vastai_csm_server_expanded.py`)
Updated server supports all 28 emotions:

```python
# Endpoints:
# GET  /health    - Server status + emotion count
# GET  /emotions  - List all available emotions
# POST /generate  - Generate speech with emotion reference

# Example request:
{
  "text": "I'm here for you",
  "speaker": 0,
  "reference_emotion": "comforting",  # Can use any tier emotion
  "max_audio_length_ms": 10000
}
```

### 4. Pipeline Integration (`pipeline.py`)
Main pipeline now uses emotion library:

```python
# Automatically resolves emotions
pipeline = OviyaPipeline()
result = pipeline.process("I'm feeling stressed")
# Brain outputs "worried" → Library resolves to "concerned_anxious"
# CSM uses concerned_anxious reference for generation
```

---

## 🚀 Vast.ai Setup Instructions

### Step 1: Upload Files to Vast.ai

```bash
# From local machine
scp scripts/generate_expanded_emotions_vastai.py root@<VAST_IP>:/workspace/scripts/
scp scripts/vastai_csm_server_expanded.py root@<VAST_IP>:/workspace/scripts/
scp scripts/setup_expanded_emotions_vastai.sh root@<VAST_IP>:/workspace/
```

### Step 2: Run Setup Script

```bash
# SSH into Vast.ai
ssh root@<VAST_IP>

# Run setup (generates 28 emotions + starts server)
cd /workspace
chmod +x setup_expanded_emotions_vastai.sh
./setup_expanded_emotions_vastai.sh
```

**What it does:**
1. Creates `/workspace/emotion_references/` directory
2. Generates 28 emotion WAV files using CSM (~15-20 min)
3. Creates `emotion_library.json` config
4. Starts CSM server on port 6006

### Step 3: Verify Installation

```bash
# Check generated files
ls -lh /workspace/emotion_references/*.wav | wc -l
# Should show 28

# Test server
curl http://localhost:6006/health
# Should show: "emotion_library": 28, "available_emotions": 28

# List all emotions
curl http://localhost:6006/emotions
```

### Step 4: Test Generation

```bash
# Test with expanded emotion
curl -X POST http://localhost:6006/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "It will all be okay, I promise.",
    "reference_emotion": "comforting"
  }'
```

---

## 🧪 Testing the Expanded Library

### Test Script 1: Verify All Emotions

```python
#!/usr/bin/env python3
"""Test all 28 emotions"""

from voice.emotion_library import get_emotion_library
import requests

library = get_emotion_library()
csm_url = "https://your-ngrok-url.ngrok-free.dev/generate"

# Test texts for each tier
test_texts = {
    "tier1_core": "I'm here for you, everything will be fine.",
    "tier2_contextual": "That's really interesting, tell me more.",
    "tier3_expressive": "Oh wow, I didn't expect that at all!"
}

# Test each emotion
for tier, emotions in library.tiers.items():
    print(f"\n{'='*60}")
    print(f"Testing {tier}: {len(emotions)} emotions")
    print('='*60)
    
    for emotion in emotions:
        try:
            response = requests.post(csm_url, json={
                "text": test_texts[tier],
                "reference_emotion": emotion
            }, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ {emotion}")
            else:
                print(f"❌ {emotion}: {response.status_code}")
        except Exception as e:
            print(f"❌ {emotion}: {e}")

print("\n✅ All tests complete!")
```

### Test Script 2: Emotion Resolution

```python
#!/usr/bin/env python3
"""Test emotion aliases"""

from voice.emotion_library import get_emotion_library

library = get_emotion_library()

# Test aliases
test_cases = [
    ("happy", "joyful_excited"),
    ("sad", "empathetic_sad"),
    ("worried", "concerned_anxious"),
    ("calm", "calm_supportive"),
    ("excited", "joyful_excited"),
    ("caring", "affectionate"),
    ("unknown_emotion", "neutral")  # Fallback
]

print("🧪 Testing Emotion Resolution\n")
for input_emotion, expected in test_cases:
    resolved = library.get_emotion(input_emotion)
    status = "✅" if resolved == expected else "❌"
    print(f"{status} '{input_emotion}' → '{resolved}' (expected: '{expected}')")
```

---

## 📈 Performance Expectations

| Metric | Value |
|--------|-------|
| **Total emotions** | 28 |
| **Generation time** | ~1-2s per utterance |
| **Reference loading** | ~50ms per emotion |
| **Memory overhead** | ~200MB (all refs loaded) |
| **Disk space** | ~50MB (28 WAV files) |

---

## 🔄 Future Enhancements (Phase 2)

### EmotiVoice Integration

**Goal:** Add 15-20 more emotions from EmotiVoice model

```bash
# On Vast.ai
cd /workspace
git clone https://github.com/netease-youdao/EmotiVoice.git
cd EmotiVoice
pip install -r requirements.txt

# Generate EmotiVoice references
python3 scripts/extract_emotivoice_refs.py
# This will add: flirty, lonely, awestruck, etc.
```

### IEMOCAP/RAVDESS Dataset Integration

**Goal:** Add real human prosody references

```bash
# Download RAVDESS
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
unzip Audio_Speech_Actors_01-24.zip -d /workspace/ravdess/

# Extract embeddings
python3 scripts/extract_ravdess_refs.py
```

**Target:** 50-60 total emotions

---

## 🐛 Troubleshooting

### Issue: "Reference not found" error

```bash
# Check if emotion exists
ls /workspace/emotion_references/<emotion>.wav

# Regenerate if missing
cd /workspace/csm/csm
python3 << EOF
from generator import load_csm_1b
import torchaudio
gen = load_csm_1b(device="cuda")
audio = gen.generate(text="Test", speaker=0, context=[])
torchaudio.save("/workspace/emotion_references/test.wav", audio.unsqueeze(0).cpu(), gen.sample_rate)
EOF
```

### Issue: "Emotion library config not found"

```bash
# Check config exists
cat /workspace/emotion_references/emotion_library.json

# Copy from local if missing
scp config/emotion_library.json root@<VAST_IP>:/workspace/emotion_references/
```

### Issue: Server not responding

```bash
# Check server process
ps aux | grep vastai_csm_server

# Check port availability
netstat -tulpn | grep 6006

# Restart server
cd /workspace/csm/csm
python3 /workspace/scripts/vastai_csm_server_expanded.py
```

---

## 📚 API Reference

### Emotion Library API

```python
from voice.emotion_library import get_emotion_library

library = get_emotion_library()

# Get emotion (with alias resolution)
emotion = library.get_emotion("happy")  # Returns "joyful_excited"

# Validate emotion
is_valid, resolved = library.validate_emotion("worried")
# (True, "concerned_anxious")

# Get tier
tier = library.get_tier("comforting")  # "tier1_core"

# Sample random emotion (tier-weighted)
random_emotion = library.sample_emotion()

# Get similar emotions
similar = library.get_similar_emotions("joyful_excited", count=3)
# ["encouraging", "proud", "grateful"]

# Statistics
stats = library.get_emotion_stats()
# {"total_emotions": 28, "tier1_count": 10, ...}
```

### CSM Server API

```bash
# Health check
curl http://localhost:6006/health

# List emotions
curl http://localhost:6006/emotions

# Generate speech
curl -X POST http://localhost:6006/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello there",
    "speaker": 0,
    "reference_emotion": "playful",
    "max_audio_length_ms": 10000
  }'
```

---

## ✅ Implementation Checklist

- [x] Create emotion blending system
- [x] Build emotion library manager
- [x] Create expanded emotion config (28 emotions)
- [x] Write Vast.ai generation script
- [x] Update CSM server for expanded library
- [x] Integrate with main pipeline
- [x] Create setup automation script
- [ ] Generate all 28 emotion references on Vast.ai
- [ ] Test all emotions through pipeline
- [ ] Verify emotion resolution/aliases
- [ ] Document performance metrics
- [ ] (Phase 2) Add EmotiVoice integration
- [ ] (Phase 2) Add RAVDESS/IEMOCAP refs

---

## 📞 Quick Reference

| Component | Location |
|-----------|----------|
| Emotion blender | `voice/emotion_blender.py` |
| Library manager | `voice/emotion_library.py` |
| Config file | `config/emotion_library.json` |
| CSM server | `scripts/vastai_csm_server_expanded.py` |
| Generation script | `scripts/generate_expanded_emotions_vastai.py` |
| Setup script | `scripts/setup_expanded_emotions_vastai.sh` |
| Reference dir | `/workspace/emotion_references/` (Vast.ai) |

---

**Last Updated:** 2025-10-10
**Version:** 1.0 (Phase 1 - Base 28 emotions)

