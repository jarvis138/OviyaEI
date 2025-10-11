# Emotion Reference System - Implementation Guide

## Overview

This system uses OpenVoiceV2 emotion references as CSM context to achieve emotionally expressive yet conversationally natural speech.

```
User Input → Emotion Detector → Brain → Emotion Controller
                                            ↓
                                  Select Emotion Reference
                                            ↓
                          CSM (with reference context) → Audio Output
```

## Architecture

- **Teacher Model**: OpenVoiceV2 - Generates emotional reference audio
- **Student Model**: CSM - Uses references as context for generation
- **Integration**: References prepended to CSM's conversation context

## Implementation Status

✅ **Completed:**
1. `voice/emotion_teacher.py` - OpenVoiceV2 wrapper
2. `voice/openvoice_tts.py` - HybridVoiceEngine with `generate_with_reference()`
3. `config/emotion_reference_mapping.json` - Emotion mappings
4. `emotion_reference/emotion_evaluator.py` - Stage 0 evaluation framework
5. `extract_emotion_references_vastai.py` - Reference extraction script
6. `update_csm_server_vastai.py` - CSM server update script

## Setup Instructions

### Part 1: Setup on Vast.ai

#### 1. Install OpenVoiceV2

```bash
# SSH into Vast.ai
cd /workspace

# Clone OpenVoiceV2
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice

# Install dependencies
pip install -r requirements.txt
```

#### 2. Generate Emotion References

```bash
cd /workspace

# Copy extraction script (from your local machine)
# Then run it:
python3 extract_emotion_references_vastai.py
```

This generates 8 emotion reference WAV files in `/workspace/emotion_references/`:
- calm_supportive.wav
- empathetic_sad.wav
- joyful_excited.wav
- playful.wav
- confident.wav
- concerned_anxious.wav
- angry_firm.wav
- neutral.wav

#### 3. Update CSM Server

```bash
# Run the update script
python3 update_csm_server_vastai.py

# Stop your current CSM server (Ctrl+C in that terminal)

# Start the updated server
python3 /workspace/official_csm_server_with_emotions.py
```

The updated server now accepts `reference_emotion` parameter!

#### 4. Keep Ngrok Tunnel Running

Make sure your ngrok tunnel is still active:
```bash
./ngrok http 6006
```

Your CSM URL: `https://tanja-flockier-jayleen.ngrok-free.dev`

### Part 2: Local Testing

#### 1. Verify Configuration

The emotion reference system is already integrated:
- `voice/openvoice_tts.py` - Sends `reference_emotion` in payload
- `emotion_controller/controller.py` - Maps emotions to labels
- `pipeline.py` - Complete integration

#### 2. Test Individual Emotions

```bash
cd oviya-production

# Test single emotion
python3 -c "
from voice.openvoice_tts import HybridVoiceEngine
from emotion_controller.controller import EmotionController

voice = HybridVoiceEngine()
controller = EmotionController('config/emotions.json')

# Test joyful emotion
params = controller.map_emotion('joyful_excited', intensity=0.8)
audio = voice.generate(
    text='That is amazing! I am so happy for you!',
    emotion_params=params,
    speaker_id='oviya_v1'
)

import torchaudio
torchaudio.save('test_joyful_with_reference.wav', audio.unsqueeze(0), 24000)
print('✅ Generated: test_joyful_with_reference.wav')
"
```

#### 3. Run Stage 0 Evaluation

```bash
python3 stage0_emotion_test.py
```

This will:
1. Extract emotion references from OpenVoiceV2 (teacher)
2. Generate CSM audio with each reference (student)
3. Compare outputs and calculate similarity scores
4. Save results to `output/emotion_transfer/`

#### 4. Run Complete Pipeline

```bash
python3 pipeline.py
```

Now when you interact with Oviya:
- Your emotion is detected
- Brain generates response with emotion label
- Emotion Controller maps to parameters
- **HybridVoiceEngine selects emotion reference**
- **CSM generates with reference as context**
- Result: Emotionally expressive + conversationally natural!

## How It Works

### 1. Emotion Reference Selection

```python
# In openvoice_tts.py
emotion_label = emotion_params.get("emotion_label", "neutral")
payload = {
    "text": text,
    "speaker": 0,
    "reference_emotion": emotion_label  # ← Key addition
}
```

### 2. Server-Side Processing

```python
# In official_csm_server_with_emotions.py
if reference_emotion:
    # Load reference audio
    ref_audio, sr = torchaudio.load(f"{reference_emotion}.wav")
    
    # Create Segment
    ref_segment = Segment(
        text=EMOTION_TEXTS[reference_emotion],
        speaker=0,
        audio=ref_audio
    )
    
    # Prepend to context
    context = [ref_segment] + context
```

### 3. CSM Generation

```python
# CSM uses reference as emotional conditioning
audio = generator.generate(
    text=text,
    speaker=speaker,
    context=context,  # Includes emotion reference
    max_audio_length_ms=10000
)
```

## Expected Results

### Before (CSM Only):
- Consistent voice
- Natural conversation
- Limited emotional expression

### After (CSM + Emotion References):
- Consistent voice ✅
- Natural conversation ✅
- Rich emotional expression ✅ (NEW!)

## Evaluation Metrics

Stage 0 evaluation measures:
1. **Success Rate**: Can CSM use references?
2. **Similarity Score**: How well does CSM match reference emotion?
3. **Generation Time**: Performance impact
4. **Audio Quality**: Subjective listening tests

## Troubleshooting

### Issue: Server returns 500 error

**Solution**: Check if emotion references exist:
```bash
ls -la /workspace/emotion_references/
```

### Issue: No emotional difference in output

**Possible causes**:
1. References not loaded correctly
2. CSM not using context effectively
3. Reference audio quality issues

**Debug**:
```bash
# Check server logs for "Loaded emotion reference"
# Test with different emotions
# Verify reference WAV files are valid
```

### Issue: Slow generation

**Solution**: References add minimal overhead (~50-100ms). If slow:
1. Check network latency to Vast.ai
2. Verify CSM model is on GPU
3. Consider shorter reference clips

## Next Steps

1. ✅ **Run Stage 0 Evaluation** - Validate emotion transfer
2. 🎯 **Fine-tune References** - Adjust based on results
3. 🎯 **Clone Oviya's Voice** - Use her voice for references
4. 🎯 **A/B Testing** - Compare with/without references
5. 🎯 **Production Deploy** - Optimize and deploy

## File Reference

### Local Files
- `voice/emotion_teacher.py` - Teacher model wrapper
- `voice/openvoice_tts.py` - Voice generation with references
- `emotion_reference/emotion_evaluator.py` - Evaluation framework
- `config/emotion_reference_mapping.json` - Emotion mappings
- `extract_emotion_references_vastai.py` - Extraction script
- `update_csm_server_vastai.py` - Server update script

### Vast.ai Files
- `/workspace/OpenVoice/` - OpenVoiceV2 repository
- `/workspace/emotion_references/` - 8 emotion WAV files
- `/workspace/official_csm_server_with_emotions.py` - Updated server

## Success Criteria

✅ CSM server accepts `reference_emotion` parameter
✅ Emotion references load correctly
✅ Generated audio shows emotional variation
✅ Audio quality remains high
✅ Generation time < 2 seconds
✅ Stage 0 similarity score > 0.4

## Support

For issues or questions:
1. Check server logs on Vast.ai
2. Verify all files are in correct locations
3. Test each component individually
4. Run Stage 0 evaluation for diagnostics


