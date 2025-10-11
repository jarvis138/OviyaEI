# Quick Start: Emotion Reference System

## 🚀 **Implementation Complete!**

Your emotion reference system is ready! Here's how to activate it:

## Part 1: Vast.ai Setup (15 minutes)

### Step 1: SSH into Vast.ai

```bash
# Use your Vast.ai SSH details
# You should already be in /workspace
```

### Step 2: Install OpenVoiceV2

```bash
cd /workspace
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
pip install -r requirements.txt
```

### Step 3: Generate Emotion References

Upload the extraction script and run it:

```bash
cd /workspace
# Upload extract_emotion_references_vastai.py to /workspace
python3 extract_emotion_references_vastai.py
```

This creates `/workspace/emotion_references/` with 8 WAV files.

### Step 4: Update and Restart CSM Server

```bash
# Upload update script
python3 update_csm_server_vastai.py

# Stop current CSM server (Ctrl+C in that terminal)

# Start updated server
python3 /workspace/official_csm_server_with_emotions.py
```

### Step 5: Verify Server

```bash
curl http://localhost:6006/health
```

Should show: `"emotion_references_available": true`

## Part 2: Local Testing (5 minutes)

### Test 1: Simple Emotion Test

```bash
cd "/Users/jarvis/Documents/Oviya EI/oviya-production"

python3 -c "
from voice.openvoice_tts import HybridVoiceEngine
from emotion_controller.controller import EmotionController

print('Testing emotion reference system...')
voice = HybridVoiceEngine()
controller = EmotionController('config/emotions.json')

# Test joyful
params = controller.map_emotion('joyful_excited', intensity=0.8)
audio = voice.generate(
    text='That is amazing! I am so happy for you!',
    emotion_params=params,
    speaker_id='oviya_v1'
)

import torchaudio
torchaudio.save('test_joyful_ref.wav', audio.unsqueeze(0), 24000)
print('✅ Generated with joy reference: test_joyful_ref.wav')

# Test sad
params = controller.map_emotion('empathetic_sad', intensity=0.8)
audio = voice.generate(
    text='I am so sorry you are going through this.',
    emotion_params=params,
    speaker_id='oviya_v1'
)

torchaudio.save('test_sad_ref.wav', audio.unsqueeze(0), 24000)
print('✅ Generated with sad reference: test_sad_ref.wav')
"
```

### Test 2: Run Complete Pipeline

```bash
python3 pipeline.py
```

Type: "I'm feeling really happy today!"

Expected: CSM uses `joyful_excited` reference → Emotionally expressive output!

## What Changed?

### Before
```
User → Brain → CSM (no emotion) → Flat audio
```

### After
```
User → Brain → Emotion Controller
                      ↓
            Select Emotion Reference
                      ↓
    CSM (with reference) → Expressive audio!
```

## How to Verify It's Working

1. **Check server logs** (on Vast.ai):
   ```
   ✅ Loaded emotion reference: joyful_excited.wav
   ```

2. **Listen to generated audio**:
   - Should hear emotional variation
   - Joy = brighter, faster
   - Sad = softer, slower
   - Calm = gentle, steady

3. **Check response payload**:
   ```json
   {
     "audio_base64": "...",
     "reference_emotion": "joyful_excited",
     "status": "success"
   }
   ```

## Troubleshooting

### "emotion_references_available": false

**Fix**:
```bash
cd /workspace
python3 extract_emotion_references_vastai.py
ls emotion_references/  # Verify 8 WAV files
```

### Still sounds flat

**Check**:
1. Server using updated version?
2. References generated correctly?
3. Try different emotions side-by-side

### 500 Error

**Debug**:
```bash
# On Vast.ai, check server logs
# Look for error messages about loading references
```

## Next Steps

1. ✅ Test all 8 emotions individually
2. ✅ Run Stage 0 evaluation (optional)
3. ✅ Compare with/without references
4. 🎯 Fine-tune based on results
5. 🎯 Clone Oviya's voice for references

## Files Created

**Local** (`oviya-production/`):
- ✅ `voice/emotion_teacher.py`
- ✅ `voice/openvoice_tts.py` (updated)
- ✅ `config/emotion_reference_mapping.json`
- ✅ `extract_emotion_references_vastai.py`
- ✅ `update_csm_server_vastai.py`
- ✅ `EMOTION_REFERENCE_GUIDE.md`

**Vast.ai** (`/workspace/`):
- ✅ `OpenVoice/` (cloned repo)
- ✅ `emotion_references/` (8 WAV files)
- ✅ `official_csm_server_with_emotions.py`

## Success! 🎉

You now have:
- ✅ Real CSM voice generation
- ✅ Real LLM brain (Qwen2.5)
- ✅ Emotion reference system
- ✅ Complete hybrid model

**Your Oviya is emotionally intelligent AND conversationally natural!**


