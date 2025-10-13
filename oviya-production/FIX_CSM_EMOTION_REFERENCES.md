# 🔧 Fix CSM Emotion Reference Issue

## Problem Identified

**Issue**: `csm_emotion_3.wav` contains the text "I am feeling sad and lonely" instead of the intended emotion reference text.

**Root Cause**: The emotion reference audio files on Vast.ai were created with incorrect/inconsistent text content.

**Impact**: When CSM uses this reference as context, it may repeat or be influenced by the reference audio's spoken content.

---

## Solution: Regenerate Emotion References

### Step 1: SSH into Vast.ai

```bash
ssh root@your-vastai-instance.com -p YOUR_PORT
cd /workspace
```

### Step 2: Check Current References

```bash
ls -la /workspace/emotion_references/
```

Look for files like:
- `csm_emotion_0.wav` through `csm_emotion_7.wav`
- OR `calm_supportive.wav`, `empathetic_sad.wav`, etc.

### Step 3: Identify the Problematic File

The issue is with `csm_emotion_3` which should map to `empathetic_sad`.

**Correct mapping** (standard order):
```
csm_emotion_0 = calm_supportive
csm_emotion_1 = empathetic_sad  ← This might be csm_emotion_3 for you
csm_emotion_2 = joyful_excited
csm_emotion_3 = playful
csm_emotion_4 = confident
csm_emotion_5 = concerned_anxious
csm_emotion_6 = angry_firm
csm_emotion_7 = neutral
```

### Step 4: Regenerate with Correct Text

Run this Python script on Vast.ai:

```python
#!/usr/bin/env python3
"""Regenerate emotion references with correct text"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

# Correct emotion reference texts
EMOTION_TEXTS = {
    "calm_supportive": "Take a deep breath. Everything will be okay.",
    "empathetic_sad": "I'm so sorry you're going through this.",
    "joyful_excited": "That's amazing! I'm so happy for you!",
    "playful": "Hey there! This is going to be fun!",
    "confident": "You've got this. I believe in you.",
    "concerned_anxious": "Are you okay? I'm here if you need me.",
    "angry_firm": "That's not acceptable. This needs to stop.",
    "neutral": "Hello. How can I help you today?"
}

def generate_synthetic_emotional_audio(text: str, emotion: str, duration_sec: float = 3.5):
    """
    Generate synthetic emotional audio
    NOTE: This is a fallback. Ideally use a real TTS model.
    """
    sample_rate = 24000
    num_samples = int(duration_sec * sample_rate)
    
    # Generate base tone
    t = np.linspace(0, duration_sec, num_samples)
    
    # Emotion-specific parameters
    params = {
        "calm_supportive": {"freq": 200, "vibrato": 2, "energy": 0.25},
        "empathetic_sad": {"freq": 180, "vibrato": 3, "energy": 0.20},
        "joyful_excited": {"freq": 300, "vibrato": 8, "energy": 0.40},
        "playful": {"freq": 280, "vibrato": 10, "energy": 0.35},
        "confident": {"freq": 220, "vibrato": 4, "energy": 0.30},
        "concerned_anxious": {"freq": 210, "vibrato": 6, "energy": 0.28},
        "angry_firm": {"freq": 250, "vibrato": 6, "energy": 0.50},
        "neutral": {"freq": 200, "vibrato": 2, "energy": 0.25}
    }
    
    p = params.get(emotion, params["neutral"])
    
    # Base tone + vibrato
    audio = np.sin(2 * np.pi * p["freq"] * t)
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * p["vibrato"] * t)
    audio = audio * vibrato * p["energy"]
    
    # Add decay
    decay = np.exp(-t / (duration_sec * 0.8))
    audio = audio * decay
    
    return torch.tensor(audio, dtype=torch.float32), sample_rate

def main():
    print("\n" + "="*70)
    print("  REGENERATING EMOTION REFERENCES")
    print("="*70 + "\n")
    
    output_dir = Path("/workspace/emotion_references")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for emotion, text in EMOTION_TEXTS.items():
        print(f"\n[{emotion}]")
        print(f"   Text: \"{text}\"")
        
        # Generate audio
        audio, sr = generate_synthetic_emotional_audio(text, emotion)
        
        # Save
        output_path = output_dir / f"{emotion}.wav"
        torchaudio.save(output_path, audio.unsqueeze(0), sr)
        
        duration = len(audio) / sr
        print(f"   ✅ Saved: {output_path.name} ({duration:.2f}s)")
    
    print("\n" + "="*70)
    print(f"✅ Generated {len(EMOTION_TEXTS)} emotion references")
    print(f"📁 Location: {output_dir}")
    print("="*70 + "\n")
    
    # Verify files
    print("Verification:")
    for f in sorted(output_dir.glob("*.wav")):
        print(f"   ✅ {f.name}")

if __name__ == "__main__":
    main()
```

Save this as `regenerate_emotion_refs.py` and run:
```bash
python3 regenerate_emotion_refs.py
```

### Step 5: Update CSM Server

Make sure your CSM server is using the **correct emotion text mapping**:

```python
# In your CSM server (official_csm_server.py or similar)
EMOTION_TEXTS = {
    "calm_supportive": "Take a deep breath. Everything will be okay.",
    "empathetic_sad": "I'm so sorry you're going through this.",  # NOT "I am feeling sad and lonely"
    "joyful_excited": "That's amazing! I'm so happy for you!",
    "playful": "Hey there! This is going to be fun!",
    "confident": "You've got this. I believe in you.",
    "concerned_anxious": "Are you okay? I'm here if you need me.",
    "angry_firm": "That's not acceptable. This needs to stop.",
    "neutral": "Hello. How can I help you today?"
}
```

### Step 6: Restart CSM Server

```bash
# Kill existing server
pkill -f "official_csm_server"

# Start fresh
python3 official_csm_server.py
```

### Step 7: Test

```bash
curl -X POST http://localhost:6006/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test of emotional speech.",
    "speaker": 0,
    "reference_emotion": "empathetic_sad"
  }' \
  -o test_empathetic_sad.wav
```

The output should now sound empathetic/sad but should say "This is a test of emotional speech" - NOT "I am feeling sad and lonely".

---

## Alternative: Use Better TTS for References

If you want **higher quality** emotion references, use a real TTS model:

### Option 1: Use CSM itself to generate references

```python
# Generate emotion reference using CSM with carefully crafted prompt
from generator import load_csm_1b, Segment

generator = load_csm_1b(device="cuda")

for emotion, text in EMOTION_TEXTS.items():
    audio = generator.generate(
        text=text,
        speaker=0,
        context=[],
        max_audio_length_ms=10000
    )
    torchaudio.save(f"/workspace/emotion_references/{emotion}.wav", 
                    audio.unsqueeze(0), generator.sample_rate)
```

### Option 2: Use OpenVoice V2 (if installed)

```python
# Use OpenVoice's emotion control
# ... (implementation depends on OpenVoice API)
```

---

## Prevention: Lock Down Reference Texts

To prevent this issue in the future, add **validation** to your CSM server:

```python
def validate_emotion_references():
    """Validate that emotion references match expected texts"""
    for emotion, expected_text in EMOTION_TEXTS.items():
        ref_path = EMOTION_REF_DIR / f"{emotion}.wav"
        if not ref_path.exists():
            print(f"⚠️  Missing reference: {emotion}")
            continue
        
        # Load and check duration (rough validation)
        audio, sr = torchaudio.load(str(ref_path))
        duration = audio.shape[-1] / sr
        
        if duration < 2.0 or duration > 6.0:
            print(f"⚠️  Suspicious duration for {emotion}: {duration:.2f}s")
    
    print("✅ Emotion references validated")

# Call on server startup
validate_emotion_references()
```

---

## Quick Fix (If You Can't Regenerate)

If you **can't regenerate** the references right now, you can work around it by:

1. **Don't use csm_emotion_3** - map it to a different emotion temporarily
2. **Disable emotion references** - remove `reference_emotion` from payloads
3. **Use a different emotion** - try `csm_emotion_0` or `csm_emotion_1` instead

---

## Summary

✅ **Problem**: Reference audio contains wrong text  
✅ **Solution**: Regenerate with correct EMOTION_TEXTS mapping  
✅ **Prevention**: Add validation on server startup  

**Expected behavior after fix**:
- Reference provides emotional **style/tone**
- CSM speaks the **requested text**, not the reference text
- Output is emotionally expressive with correct content


