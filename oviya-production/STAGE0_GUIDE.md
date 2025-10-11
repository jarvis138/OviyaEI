# 🧪 Stage 0: Emotion Reference Evaluation Guide

## What is Stage 0?

**Stage 0** is a baseline evaluation that tests if **CSM can reproduce emotions** from **OpenVoice V2's emotional references** WITHOUT any fine-tuning.

This is a critical first step before investing time in training - it validates that CSM has the right "hooks" for emotional conditioning.

## The Strategy

```
OpenVoice V2 (Teacher)
      ↓
  Has built-in emotional reference library
  (calm, sad, happy, angry, etc.)
      ↓
Extract emotional reference audio
      ↓
Feed to CSM (Student) as conditioning
      ↓
Test: Does CSM reproduce the emotion?
      ↓
IF YES → Proceed to Stage 1 (Fine-tuning)
IF NO → Need architectural changes
```

## Quick Start

### 1. Prerequisites

```bash
# Ensure CSM is running on Vast.ai (port 6006)
# Check connection:
python3 test_csm_connection.py
```

### 2. (Optional) Install OpenVoice V2

```bash
# Clone OpenVoice V2 for real emotional references
git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice
cd external/OpenVoice
pip install -e .
cd ../..
```

**Note:** If OpenVoice V2 is not installed, the system will use mock emotional references for testing.

### 3. Run Stage 0 Evaluation

```bash
python3 stage0_emotion_test.py
```

This will:
1. ✅ Load OpenVoice V2's emotional references
2. ✅ Feed each emotion reference to CSM
3. ✅ Generate CSM audio for each emotion
4. ✅ Compare teacher vs student outputs
5. ✅ Save results and audio files

### 4. Review Results

```bash
# Check audio outputs
ls -la output/emotion_transfer/

# Listen to comparisons:
# teacher_calm_supportive.wav = OpenVoice V2 reference
# csm_calm_supportive.wav = CSM's reproduction

# Check JSON results
cat output/emotion_transfer/evaluation_results.json
```

## Understanding Results

### Similarity Scores

| Score Range | Interpretation | Next Step |
|-------------|----------------|-----------|
| **> 0.6** | Strong emotion transfer | ✅ Proceed to Stage 1 (Fine-tuning) |
| **0.4-0.6** | Moderate transfer | ⚠️ Test with longer references |
| **< 0.4** | Weak transfer | ❌ Consider architectural changes |

### What to Listen For

When comparing `teacher_*.wav` vs `csm_*.wav`:

✅ **Good Signs:**
- Similar pacing/rhythm
- Similar pitch patterns
- Similar energy/intensity
- Emotion feels similar

❌ **Bad Signs:**
- Monotone CSM output
- Completely different pacing
- No emotional variation
- Robotic delivery

## What This Tells Us

### ✅ If CSM responds to emotions:
- CSM's architecture supports emotional conditioning
- We can proceed to fine-tune with Oviya's voice
- The teacher-student approach will work
- Stage 1: Add emotion conditioning during training
- Stage 2: Add Oviya voice LoRA

### ❌ If CSM doesn't respond:
- CSM may need explicit emotion embeddings
- Consider adding emotion adapter layers
- May need to use different base model
- Alternative: Use OpenVoice V2 directly with LoRA

## Files Created

```
oviya-production/
├── stage0_emotion_test.py              # Main test script
├── voice/
│   └── emotion_teacher.py              # OpenVoice V2 wrapper
├── emotion_reference/
│   ├── __init__.py
│   └── emotion_evaluator.py            # Evaluation logic
├── config/
│   └── emotion_reference.json          # Test configuration
└── output/
    └── emotion_transfer/                # Test results
        ├── teacher_*.wav                # OpenVoice references
        ├── csm_*.wav                   # CSM outputs
        ├── references/                  # Reference library
        └── evaluation_results.json      # Metrics
```

## Advanced Testing

### Test Individual Emotions

```python
from voice.emotion_teacher import OpenVoiceEmotionTeacher
from voice.openvoice_tts import HybridVoiceEngine
from emotion_reference.emotion_evaluator import EmotionTransferEvaluator

teacher = OpenVoiceEmotionTeacher()
student = HybridVoiceEngine(csm_url="http://localhost:6006/generate")
evaluator = EmotionTransferEvaluator(teacher, student)

# Test specific emotion
result = evaluator.test_emotion_transfer(
    emotion="joyful_excited",
    text="I'm so proud of you, that's amazing!"
)
```

### Extract Emotion Embeddings

```python
from voice.emotion_teacher import OpenVoiceEmotionTeacher

teacher = OpenVoiceEmotionTeacher()

# Extract all emotion embeddings
teacher.save_embeddings("data/emotion_embeddings")

# Embeddings saved as .npy files for later training
```

## Troubleshooting

### CSM Not Responding
```bash
# Check CSM service
curl http://localhost:6006/health

# Restart if needed on Vast.ai
```

### OpenVoice V2 Not Found
- System will use mock references
- Mock references have basic emotion variations
- Install OpenVoice V2 for real emotional references

### Low Similarity Scores
- Try longer text (more words for CSM to process)
- Check if CSM service is actually using context
- Verify reference audio is audible/clear

## Next Steps

### If Stage 0 Succeeds:
1. **Stage 1:** Fine-tune CSM with emotion conditioning
2. **Stage 2:** Add Oviya voice LoRA
3. **Stage 3:** Production deployment

### If Stage 0 Fails:
1. Test with explicit emotion tokens
2. Add emotion adapter to CSM
3. Consider hybrid approach (OpenVoice + CSM)

## Research Context

This approach is based on standard **teacher-student distillation** for expressive TTS:

1. **Teacher** (OpenVoice V2) already knows emotions
2. **Student** (CSM) learns to reproduce them
3. **Distillation** transfers emotional knowledge
4. **Fine-tuning** adds persona (Oviya's voice)

This is exactly how research labs build production TTS systems!

## Questions?

Check the evaluation results and audio files first.
The similarity scores and audio comparisons will tell you
everything you need to know about CSM's emotional responsiveness.

**🎯 Goal: Validate CSM's emotional capability before training!**



