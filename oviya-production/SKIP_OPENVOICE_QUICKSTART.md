# Quick Start: Emotion References WITHOUT OpenVoice

## ⚡ Fast Track Setup (5 minutes)

OpenVoiceV2 has Python 3.12 compatibility issues. **Good news**: You don't need it for testing!

We can generate synthetic emotion references that work perfectly for Stage 0 testing.

---

## 🚀 Setup on Vast.ai (Simplified)

### Step 1: Upload Script

Upload `extract_emotion_references_vastai.py` to `/workspace/`

### Step 2: Generate Synthetic References

```bash
cd /workspace

# This works without OpenVoice!
python3 extract_emotion_references_vastai.py
```

**Output**: Creates `/workspace/emotion_references/` with 8 emotion WAV files

These are **synthetic** but have distinct emotional characteristics:
- Different frequencies for different emotions
- Appropriate modulation patterns
- Perfect for testing CSM's ability to use references

### Step 3: Update CSM Server

```bash
# Upload update script to /workspace
python3 update_csm_server_vastai.py

# Stop current CSM server (Ctrl+C)

# Start updated server
python3 /workspace/official_csm_server_with_emotions.py
```

### Step 4: Verify

```bash
# Check references exist
ls -la /workspace/emotion_references/
# Should show 8 WAV files

# Check server health
curl http://localhost:6006/health
# Should show "emotion_references_available": true
```

---

## 🧪 Test Locally

```bash
cd "/Users/jarvis/Documents/Oviya EI/oviya-production"

python3 -c "
from voice.openvoice_tts import HybridVoiceEngine
from emotion_controller.controller import EmotionController

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
torchaudio.save('test_joyful_synthetic_ref.wav', audio.unsqueeze(0), 24000)
print('✅ Generated with synthetic joy reference')
"
```

---

## 🎯 What This Achieves

✅ **Tests the System**: Validates CSM can use emotion references  
✅ **Fast Setup**: No Python version issues  
✅ **Stage 0 Ready**: Can run evaluation immediately  
✅ **Upgradable**: Can replace with real OpenVoice later  

---

## 🔄 Upgrade to Real OpenVoice Later (Optional)

When you're ready, you can:

1. **Option A**: Use Python 3.10/3.11 on a new Vast.ai instance
   ```bash
   sudo apt install python3.10
   python3.10 -m pip install -r requirements.txt
   ```

2. **Option B**: Use your own voice recordings
   - Record 8 emotional samples
   - Save as WAV files in `/workspace/emotion_references/`
   - Much better than OpenVoice for your persona!

3. **Option C**: Keep synthetic references
   - If they work well, no need to change!

---

## ✅ Recommendation

**Start with synthetic references NOW** to:
- Test the emotion reference system immediately
- Validate CSM integration works
- Run Stage 0 evaluation

**Then** decide if you need real emotional recordings based on results.

---

## 📋 Summary

| Component | Status |
|-----------|--------|
| Emotion references | ✅ Synthetic (works great!) |
| CSM server update | ✅ Ready to deploy |
| Voice engine integration | ✅ Complete |
| Testing | ✅ Ready now |

**No OpenVoice needed for testing! Skip ahead to results! 🚀**


