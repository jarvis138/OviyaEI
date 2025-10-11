# 🚀 Quick Start - Expanded Emotions

## 30-Second Overview

Your Oviya system now has **28 emotions** instead of 8. Just deploy on Vast.ai and use!

---

## 📋 What You Need

1. Vast.ai instance running CSM
2. 3 scripts uploaded
3. 20 minutes to generate emotions
4. Ngrok for tunnel

---

## ⚡ 5-Minute Deploy

### Step 1: Upload (2 min)
```bash
cd "/Users/jarvis/Documents/Oviya EI/oviya-production"
ssh root@<VAST_IP> "mkdir -p /workspace/scripts"
scp scripts/generate_expanded_emotions_vastai.py root@<VAST_IP>:/workspace/scripts/
scp scripts/vastai_csm_server_expanded.py root@<VAST_IP>:/workspace/scripts/
scp config/emotion_library.json root@<VAST_IP>:/workspace/
```

### Step 2: Generate (20 min - automatic)
```bash
ssh root@<VAST_IP>
cd /workspace/csm/csm
python3 /workspace/scripts/generate_expanded_emotions_vastai.py
```

### Step 3: Start Server (1 min)
```bash
# Create config
cp /workspace/emotion_library.json /workspace/emotion_references/

# Start server
cd /workspace/csm/csm
python3 /workspace/scripts/vastai_csm_server_expanded.py &
```

### Step 4: Tunnel (2 min)
```bash
cd ~
./ngrok http 6006
# Copy the https://xxx.ngrok-free.dev URL
```

### Step 5: Update Local (30 sec)
```python
# In pipeline.py line ~63
csm_url="https://your-ngrok-url.ngrok-free.dev/generate"
```

### Step 6: Test (30 sec)
```bash
python3 test_pipeline.py
```

---

## 🧪 Quick Test

```bash
# Test emotion resolution
python3 test_emotion_library.py

# Should output: ✅ ALL TESTS PASSED!
```

---

## 🎨 28 Emotions Available

### Tier 1 (Use Most Often) - 10 emotions
```
calm_supportive, empathetic_sad, joyful_excited, confident, neutral,
comforting, encouraging, thoughtful, affectionate, reassuring
```

### Tier 2 (Situational) - 9 emotions
```
playful, concerned_anxious, melancholy, wistful, tired,
curious, dreamy, relieved, proud
```

### Tier 3 (Rare/Dramatic) - 9 emotions
```
angry_firm, sarcastic, mischievous, tender, amused,
sympathetic, reflective, grateful, apologetic
```

---

## 🔗 Emotion Aliases

Your LLM can output simple words, we'll resolve them:
```
happy      → joyful_excited
sad        → empathetic_sad
worried    → concerned_anxious
calm       → calm_supportive
excited    → joyful_excited
supportive → comforting
caring     → affectionate
stressed   → concerned_anxious
frustrated → angry_firm
```

---

## 💻 Usage Examples

### In Pipeline
```python
pipeline = OviyaPipeline()  # Auto-loads 28 emotions

# Alias resolution
pipeline.process("I'm stressed")
# Brain outputs "worried" → Resolves to "concerned_anxious"

# Direct emotion
pipeline.process("Tell me more!", user_emotion="curious")
```

### Direct CSM Call
```python
import requests

response = requests.post("https://your-ngrok-url.ngrok-free.dev/generate", json={
    "text": "I'm here for you",
    "reference_emotion": "comforting",  # Any of the 28
    "max_audio_length_ms": 10000
})

audio_b64 = response.json()["audio_base64"]
```

---

## 🐛 Troubleshooting One-Liners

```bash
# Check emotions generated
ls /workspace/emotion_references/*.wav | wc -l  # Should be 28

# Test server
curl http://localhost:6006/health | jq '.emotion_library'  # Should be 28

# Restart server
pkill -f vastai_csm_server && cd /workspace/csm/csm && python3 /workspace/scripts/vastai_csm_server_expanded.py &

# Check ngrok
ps aux | grep ngrok

# Full status
echo "Emotions: $(ls /workspace/emotion_references/*.wav | wc -l), Server: $(ps aux | grep vastai_csm_server | grep -v grep | wc -l), Ngrok: $(ps aux | grep ngrok | grep -v grep | wc -l)"
```

---

## 📚 Full Documentation

- **EXPANDED_EMOTIONS_GUIDE.md** - Complete guide
- **VASTAI_DEPLOYMENT.md** - Step-by-step deployment
- **EXPANDED_EMOTIONS_SUMMARY.md** - Technical overview

---

## ✅ Checklist

Quick deployment checklist:
- [ ] Scripts uploaded to Vast.ai
- [ ] 28 emotions generated (`ls /workspace/emotion_references/*.wav | wc -l`)
- [ ] Server running (`ps aux | grep vastai_csm_server`)
- [ ] Ngrok tunnel active (`./ngrok http 6006`)
- [ ] Local pipeline updated with ngrok URL
- [ ] Test passed (`python3 test_emotion_library.py`)

---

## 🎯 Expected Results

**Before (8 emotions):**
- Basic emotional range
- Limited nuance
- Generic responses

**After (28 emotions):**
- Rich emotional palette
- Nuanced expression
- Contextually appropriate responses

**Example:**
```
User: "I'm feeling stressed about work"

OLD: Uses "worried" (one of 8 base emotions)
NEW: Uses "concerned_anxious" (dedicated stress emotion)
     OR "tired" (if fatigue-related)
     OR "reassuring" (Oviya's response emotion)
```

---

## 🚀 Ready to Deploy?

1. Open terminal
2. Copy commands from "5-Minute Deploy" above
3. Run them in order
4. Test with `python3 test_pipeline.py`
5. Done! 🎉

---

**Last Updated:** 2025-10-10
**Time to Deploy:** ~25 minutes (20 min generation + 5 min setup)
**Difficulty:** Easy (copy-paste commands)

