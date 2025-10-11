# 🚀 Vast.ai Deployment Guide - Expanded Emotions

## Quick Start

This guide will help you deploy the expanded 28-emotion CSM system on Vast.ai.

---

## 📋 Prerequisites

- Vast.ai account with credits
- Local machine with SSH access
- Files ready to upload:
  - `scripts/generate_expanded_emotions_vastai.py`
  - `scripts/vastai_csm_server_expanded.py`
  - `scripts/setup_expanded_emotions_vastai.sh`

---

## Step 1: Get Vast.ai IP Address

From your Vast.ai instance dashboard, note:
- **IP Address**: e.g., `45.78.17.160`
- **SSH Port**: Usually `22` or check port forwarding
- **Direct SSH Port**: Check "Connect" button for exact command

---

## Step 2: Upload Scripts to Vast.ai

### Option A: Using SCP (Recommended)

```bash
# From your local machine (in oviya-production directory)
cd "/Users/jarvis/Documents/Oviya EI/oviya-production"

# Create scripts directory on Vast.ai
ssh root@45.78.17.160 "mkdir -p /workspace/scripts"

# Upload scripts
scp scripts/generate_expanded_emotions_vastai.py root@45.78.17.160:/workspace/scripts/
scp scripts/vastai_csm_server_expanded.py root@45.78.17.160:/workspace/scripts/
scp scripts/setup_expanded_emotions_vastai.sh root@45.78.17.160:/workspace/
scp config/emotion_library.json root@45.78.17.160:/workspace/
```

### Option B: Manual Copy-Paste (If SSH port blocked)

1. SSH into Vast.ai via Jupyter terminal or web interface
2. Create files manually:

```bash
# On Vast.ai terminal
mkdir -p /workspace/scripts

# Create generation script
cat > /workspace/scripts/generate_expanded_emotions_vastai.py << 'EOFGEN'
# (Copy entire contents of generate_expanded_emotions_vastai.py here)
EOFGEN

# Create server script
cat > /workspace/scripts/vastai_csm_server_expanded.py << 'EOFSERV'
# (Copy entire contents of vastai_csm_server_expanded.py here)
EOFSERV

# Create setup script
cat > /workspace/setup_expanded_emotions_vastai.sh << 'EOFSETUP'
# (Copy entire contents of setup_expanded_emotions_vastai.sh here)
EOFSETUP

chmod +x /workspace/setup_expanded_emotions_vastai.sh
```

---

## Step 3: Generate 28 Emotion References

```bash
# SSH into Vast.ai
ssh root@45.78.17.160

# Navigate to CSM directory
cd /workspace/csm/csm

# Run generation script
python3 /workspace/scripts/generate_expanded_emotions_vastai.py
```

**Expected output:**
```
======================================================================
🎨 EXPANDED EMOTION REFERENCE GENERATOR
======================================================================

Generating 28 emotion references...
This will take ~15-20 minutes

🔄 Loading CSM model...
✅ CSM loaded successfully!

[1/28] Generating 'calm_supportive'...
   Text: "Take a deep breath. Everything will be okay..."
   ✅ Saved calm_supportive.wav (3.45s)

[2/28] Generating 'empathetic_sad'...
...
```

**Time:** ~15-20 minutes

**Verification:**
```bash
# Check generated files
ls -lh /workspace/emotion_references/*.wav | wc -l
# Should output: 28

# Check sizes (each should be ~100-500KB)
du -sh /workspace/emotion_references/
```

---

## Step 4: Create Emotion Library Config

```bash
# Copy config from uploaded file or create manually
cp /workspace/emotion_library.json /workspace/emotion_references/emotion_library.json

# OR create manually
cat > /workspace/emotion_references/emotion_library.json << 'EOF'
{
  "version": "1.0",
  "total_emotions": 28,
  "tiers": {
    "tier1_core": [
      "calm_supportive", "empathetic_sad", "joyful_excited",
      "confident", "neutral", "comforting", "encouraging",
      "thoughtful", "affectionate", "reassuring"
    ],
    "tier2_contextual": [
      "playful", "concerned_anxious", "melancholy", "wistful",
      "tired", "curious", "dreamy", "relieved", "proud"
    ],
    "tier3_expressive": [
      "angry_firm", "sarcastic", "mischievous", "tender",
      "amused", "sympathetic", "reflective", "grateful", "apologetic"
    ]
  },
  "emotion_texts": {
    "calm_supportive": "Take a deep breath. Everything will be okay.",
    "empathetic_sad": "I'm so sorry you're going through this.",
    "joyful_excited": "Wow! That is wonderful! I am so excited!",
    "playful": "Hey there! This is going to be fun!",
    "confident": "You've got this. I believe in you.",
    "concerned_anxious": "Are you okay? I'm here if you need me.",
    "angry_firm": "That's not acceptable. This needs to stop.",
    "neutral": "Hello. How can I help you today.",
    "comforting": "It's okay. I'm here for you, everything will be alright.",
    "encouraging": "You can do this! I believe in you completely.",
    "thoughtful": "Let me think about that for a moment. That's interesting.",
    "affectionate": "I care about you so much. You mean a lot to me.",
    "reassuring": "Don't worry. You're safe, and everything is going to be fine.",
    "melancholy": "Sometimes things are hard, and that's okay to feel.",
    "wistful": "I remember those days. It feels like a distant dream now.",
    "tired": "It's been a long day. I'm feeling a bit worn out.",
    "curious": "Really? Tell me more! I want to know everything about it.",
    "dreamy": "Imagine a peaceful place where everything is calm and beautiful.",
    "relieved": "Oh thank goodness! I'm so glad that worked out.",
    "proud": "Look at what you've accomplished! That's truly impressive.",
    "sarcastic": "Oh yeah, that's exactly what I meant. Totally.",
    "mischievous": "I have an idea, and you're going to love this.",
    "tender": "You're so precious to me. I want you to know that.",
    "amused": "Ha! That's actually pretty funny when you think about it.",
    "sympathetic": "I understand how you feel. That must be really difficult.",
    "reflective": "Looking back, I can see how all of this connects.",
    "grateful": "Thank you so much. I really appreciate everything you've done.",
    "apologetic": "I'm truly sorry. I didn't mean for things to turn out this way."
  }
}
EOF
```

---

## Step 5: Start CSM Server with Expanded Emotions

```bash
# Navigate to CSM directory
cd /workspace/csm/csm

# Start server (runs in foreground)
python3 /workspace/scripts/vastai_csm_server_expanded.py
```

**Expected output:**
```
======================================================================
🎨 CSM SERVER - EXPANDED EMOTION LIBRARY
======================================================================
📚 Loaded emotion library: 28 emotions
📂 Found 28 emotion references in /workspace/emotion_references

🔄 Loading CSM model...
✅ CSM model loaded successfully!

🚀 Starting server on port 6006...
======================================================================
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:6006
 * Running on http://10.x.x.x:6006
```

### To run in background:

```bash
# Run with nohup
nohup python3 /workspace/scripts/vastai_csm_server_expanded.py > /workspace/csm_server.log 2>&1 &

# Check process
ps aux | grep vastai_csm_server

# View logs
tail -f /workspace/csm_server.log
```

---

## Step 6: Setup Ngrok Tunnel

Since Vast.ai port 6006 might be blocked, use ngrok:

```bash
# On Vast.ai terminal
cd ~
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xvzf ngrok-v3-stable-linux-amd64.tgz
./ngrok authtoken YOUR_NGROK_TOKEN

# Start tunnel (in new terminal or background)
./ngrok http 6006
```

**Copy the ngrok URL** (e.g., `https://abc123.ngrok-free.dev`)

---

## Step 7: Test the Server

### Test 1: Health Check

```bash
# On Vast.ai
curl http://localhost:6006/health

# From local machine (via ngrok)
curl https://abc123.ngrok-free.dev/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "emotion_library": 28,
  "reference_dir": "/workspace/emotion_references",
  "available_emotions": 28
}
```

### Test 2: List Emotions

```bash
curl https://abc123.ngrok-free.dev/emotions
```

**Expected response:**
```json
{
  "total": 28,
  "emotions": [
    "affectionate", "amused", "angry_firm", "apologetic",
    "calm_supportive", "comforting", ...
  ],
  "tiers": {
    "tier1_core": [...],
    "tier2_contextual": [...],
    "tier3_expressive": [...]
  }
}
```

### Test 3: Generate Speech

```bash
curl -X POST https://abc123.ngrok-free.dev/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am here for you, everything will be fine.",
    "speaker": 0,
    "reference_emotion": "comforting",
    "max_audio_length_ms": 10000
  }' | jq '.duration'
```

---

## Step 8: Update Local Pipeline

Update your local `pipeline.py` with the ngrok URL:

```python
# In pipeline.py, line ~63
self.tts = HybridVoiceEngine(
    csm_url="https://abc123.ngrok-free.dev/generate",  # Your ngrok URL
    default_engine="auto"
)
```

---

## Step 9: Test Full Pipeline

```bash
# On local machine
cd "/Users/jarvis/Documents/Oviya EI/oviya-production"
python3 test_pipeline.py
```

Test with different emotions:
```python
# Test Tier 1 emotion
pipeline.process("I'm feeling stressed out", user_emotion="stressed")
# Should use "concerned_anxious" via alias

# Test Tier 2 emotion  
pipeline.process("Tell me something interesting")
# Brain might select "curious"

# Test Tier 3 emotion
pipeline.process("Wow that's amazing!")
# Brain might select "proud" or "joyful_excited"
```

---

## 🐛 Troubleshooting

### Issue: "Reference not found" error

```bash
# Check if emotion file exists
ls /workspace/emotion_references/comforting.wav

# Regenerate specific emotion
cd /workspace/csm/csm
python3 << EOF
import sys, os
sys.path.insert(0, '/workspace/csm/csm')
os.environ["NO_TORCH_COMPILE"] = "1"
import torch, torchaudio
from generator import load_csm_1b

gen = load_csm_1b(device="cuda")
audio = gen.generate(
    text="It's okay. I'm here for you, everything will be alright.",
    speaker=0,
    context=[],
    max_audio_length_ms=15000
)
torchaudio.save(
    "/workspace/emotion_references/comforting.wav",
    audio.unsqueeze(0).cpu(),
    gen.sample_rate
)
print("✅ Generated comforting.wav")
EOF
```

### Issue: Server not responding

```bash
# Check if server is running
ps aux | grep vastai_csm_server

# Check port
netstat -tulpn | grep 6006

# Restart server
pkill -f vastai_csm_server
cd /workspace/csm/csm
python3 /workspace/scripts/vastai_csm_server_expanded.py
```

### Issue: Ngrok tunnel closed

```bash
# Restart ngrok
pkill ngrok
./ngrok http 6006

# Update pipeline.py with new ngrok URL
```

### Issue: Out of memory

```bash
# Check GPU memory
nvidia-smi

# Check disk space
df -h

# Clear cache if needed
rm -rf /workspace/.cache/huggingface/hub/*
```

---

## 📊 Monitoring

### Check Server Logs

```bash
# If running in background
tail -f /workspace/csm_server.log

# Search for errors
grep "ERROR\|Failed" /workspace/csm_server.log
```

### Check Emotion Usage

```bash
# Count emotion requests in logs
grep "reference_emotion" /workspace/csm_server.log | \
  awk -F'"reference_emotion":' '{print $2}' | \
  awk -F',' '{print $1}' | \
  sort | uniq -c | sort -rn
```

### Check Generation Performance

```bash
# Average generation time
grep "Generated.*audio" /workspace/csm_server.log | \
  awk '{print $NF}' | \
  awk -F's' '{sum+=$1; count++} END {print sum/count "s average"}'
```

---

## ✅ Deployment Checklist

- [ ] Vast.ai instance running
- [ ] Scripts uploaded to `/workspace/scripts/`
- [ ] 28 emotion WAV files generated in `/workspace/emotion_references/`
- [ ] `emotion_library.json` config created
- [ ] CSM server running on port 6006
- [ ] Ngrok tunnel active
- [ ] Health check passes (`/health` endpoint)
- [ ] Emotions list available (`/emotions` endpoint)
- [ ] Test generation works (any emotion)
- [ ] Local pipeline updated with ngrok URL
- [ ] Full pipeline test successful

---

## 🔄 Quick Restart Commands

```bash
# SSH into Vast.ai
ssh root@45.78.17.160

# Start CSM server in background
cd /workspace/csm/csm
nohup python3 /workspace/scripts/vastai_csm_server_expanded.py > /workspace/csm_server.log 2>&1 &

# Start ngrok
cd ~
./ngrok http 6006 &

# Check status
curl http://localhost:6006/health | jq
```

---

## 📞 Support Commands

```bash
# Full system status
echo "=== CSM Server ===" && ps aux | grep vastai_csm_server && \
echo "=== Ngrok ===" && ps aux | grep ngrok && \
echo "=== Disk Space ===" && df -h /workspace && \
echo "=== GPU ===" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv && \
echo "=== Emotions ===" && ls /workspace/emotion_references/*.wav | wc -l
```

---

**Last Updated:** 2025-10-10
**Deployment Version:** 1.0 (28 emotions)

