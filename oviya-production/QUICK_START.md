# 🚀 Oviya Multi-Source Emotion System - Quick Start

## ⚡ TL;DR

**On Vast.ai** (one-time setup):
```bash
cd /workspace
./setup_complete_emotion_system.sh
python3 csm_server_expanded_emotions.py
```

**Result**: CSM server with 49+ emotions running on port 19517

---

## 📁 Files You Need

Copy to `/workspace` on Vast.ai:
1. `download_openvoice_models.sh`
2. `download_emotion_datasets.sh`
3. `extract_all_emotions.py`
4. `csm_server_expanded_emotions.py`
5. `setup_complete_emotion_system.sh`

---

## 🎯 What You Get

### **49+ Emotions**:
- **8 Base**: OpenVoiceV2 emotions
- **19 Blended**: Interpolated nuances
- **20+ Dataset**: Real human emotions (optional)

### **3 Tiers**:
- **Tier 1** (70%): Everyday emotions
- **Tier 2** (25%): Contextual emotions
- **Tier 3** (5%): Dramatic emotions

---

## 🧪 Quick Test

```bash
# Test server health
curl http://localhost:19517/health

# List emotions
curl http://localhost:19517/emotions

# Generate with emotion
curl -X POST http://localhost:19517/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy!", "reference_emotion": "joyful_excited"}'
```

---

## 📊 Emotion Quick Reference

### **Common Emotions** (Use These Most):
```
calm_supportive  - Peaceful, reassuring
joyful_excited   - Happy, energetic
empathetic_sad   - Compassionate
confident        - Strong, assured
neutral          - Balanced
comforting       - Soothing, warm
encouraging      - Motivating
```

### **Nuanced Emotions**:
```
melancholic      - Sad, reflective
wistful          - Nostalgic
tender           - Gentle, caring
thoughtful       - Considerate
grateful         - Thankful
curious          - Interested
relieved         - Glad, peaceful
```

### **Dramatic Emotions** (Use Sparingly):
```
sarcastic        - Ironic
mischievous      - Playful, teasing
frustrated       - Annoyed
determined       - Focused
hopeful          - Optimistic
```

---

## 🐛 Troubleshooting

### Server won't start?
```bash
# Check CSM is installed
ls /workspace/csm/generator.py

# Check emotions exist
ls /workspace/emotion_references/
```

### No emotions generated?
```bash
# Re-run extraction
python3 extract_all_emotions.py
```

### Out of memory?
```bash
# Reduce audio length
{"max_audio_length_ms": 5000}
```

---

## 📖 Full Documentation

- **Setup Guide**: `COMPLETE_EMOTION_SYSTEM_GUIDE.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Vast.ai Instructions**: Already included in setup scripts

---

## ✅ Success Indicators

You're ready when:
1. ✅ Server starts on port 19517
2. ✅ `/health` shows 27+ emotions
3. ✅ Generated audio has emotional variation
4. ✅ Volume is loud and clear

---

## 🎉 You're Done!

Your Oviya system now has **49+ emotions** with:
- OpenVoiceV2 references
- Mathematical blending
- Real human datasets (optional)
- Production-ready server

**Enjoy building emotionally intelligent AI!** 🚀


