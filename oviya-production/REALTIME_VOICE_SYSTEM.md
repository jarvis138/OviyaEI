# Oviya Real-Time Voice Conversation System

## Overview
ChatGPT-style voice mode for Oviya with real-time audio processing, word-level timestamps, and emotional voice responses.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                              │
│  👤 User speaks → 🎤 Audio capture → 📝 Real-time transcription │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 1: REAL-TIME VOICE INPUT                      │
│  • WhisperX (large-v2) for transcription                        │
│  • Word-level timestamp alignment                                │
│  • Voice Activity Detection (VAD)                                │
│  • Conversation context tracking                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 2: BRAIN (LLM + EMOTIONAL INTELLIGENCE)       │
│  • Ollama LLM (qwen2.5:7b)                                      │
│  • Emotional memory tracking                                     │
│  • Prosodic markup generation                                    │
│  • Backchannel system                                            │
│  • Epistemic prosody analysis                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 3: EMOTION CONTROLLER                         │
│  • 49-emotion library (3 tiers)                                 │
│  • Intensity mapping                                             │
│  • Contextual modifiers                                          │
│  • Acoustic parameter generation                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 4: VOICE OUTPUT (CSM)                         │
│  • Sesame CSM TTS model                                         │
│  • Emotion-conditioned synthesis                                 │
│  • Advanced respiratory system                                   │
│  • Audio post-processing (Maya-level)                           │
│  • Prosody processing                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    🔊 Emotional voice response
```

## Components

### 1. RealTimeVoiceInput (`voice/realtime_input.py`)
**Purpose**: Capture and transcribe user speech in real-time with word-level timestamps

**Key Features**:
- WhisperX integration with `large-v2` model
- Word-level timestamp alignment
- Streaming audio processing
- Conversation context tracking
- Voice Activity Detection (VAD)

**Methods**:
```python
# Initialize models
voice_input = RealTimeVoiceInput()
voice_input.initialize_models()

# Start recording with callback
def on_transcription(result):
    print(f"User said: {result['text']}")

voice_input.start_recording(callback=on_transcription)

# Add audio chunks (for web streaming)
voice_input.add_audio_chunk(audio_array)

# Stop and get final result
result = voice_input.stop_recording()

# Get conversation context
context = voice_input.get_conversation_context()
```

**Output Format**:
```python
{
    "text": "Full transcribed text",
    "duration": 3.5,  # seconds
    "word_timestamps": [
        {
            "word": "Hello",
            "start": 0.0,
            "end": 0.3,
            "confidence": 0.95,
            "speaker": "user"
        },
        # ... more words
    ],
    "segments": [...],  # WhisperX segments
    "language": "en",
    "timestamp": 1234567890.0
}
```

### 2. RealTimeConversation (`realtime_conversation.py`)
**Purpose**: Orchestrate the complete conversation pipeline

**Key Features**:
- Integrates all 4 layers
- Real-time emotion detection from speech timing
- Automatic voice response generation
- Turn-based conversation management

**Usage**:
```python
# Initialize system
conversation = RealTimeConversation(
    ollama_url="https://your-ollama-url/api/generate",
    csm_url="https://your-csm-url/generate"
)

# Start real-time conversation
conversation.start_conversation()  # Press Ctrl+C to stop

# Or simulate conversation for testing
test_messages = [
    "Hey Oviya, how are you?",
    "I'm feeling anxious about my exam.",
    "Can you help me feel better?"
]
conversation.simulate_conversation(test_messages)
```

## Word-Level Timestamp Context

### Why Word Timestamps Matter
Word-level timestamps enable:
1. **Emotion Detection**: Speech rate analysis (fast = anxious/excited, slow = sad/thoughtful)
2. **Context Understanding**: Timing patterns reveal emphasis and emotion
3. **Prosody Matching**: Oviya can mirror user's speech rhythm
4. **Memory Enhancement**: Track when specific topics were discussed

### Speech Rate Analysis
```python
def _analyze_user_emotion(text, word_timestamps):
    duration = word_timestamps[-1]["end"] - word_timestamps[0]["start"]
    words_per_second = len(word_timestamps) / duration
    
    if words_per_second > 3.5:
        return "excited" or "anxious"  # Fast speech
    elif words_per_second < 2.0:
        return "sad" or "thoughtful"  # Slow speech
    else:
        return "neutral"  # Normal pace
```

## Voice Activity Detection (VAD)

### Integration
VAD is automatically integrated into the WhisperX transcription pipeline:
- Filters out silence and background noise
- Detects speech segments automatically
- Improves transcription accuracy
- Reduces processing overhead

### How It Works
1. Audio chunks are buffered in real-time
2. WhisperX applies VAD during transcription
3. Only speech segments are processed
4. Silence is automatically skipped

## Conversation Memory

### Context Tracking
The system maintains:
- **Conversation History**: All transcribed turns
- **Word Timestamps**: Complete word-level timing data
- **Turn Count**: Number of conversation exchanges
- **Total Duration**: Cumulative speaking time

### Memory Structure
```python
context = {
    "history": [
        {
            "text": "...",
            "duration": 2.5,
            "word_timestamps": [...],
            "timestamp": 1234567890.0
        },
        # ... more turns
    ],
    "word_timestamps": [...],  # All words from all turns
    "turn_count": 5,
    "total_duration": 12.5
}
```

### Memory Management
```python
# Get current context
context = voice_input.get_conversation_context()

# Reset conversation
voice_input.reset_conversation()

# Clear buffer
voice_input.clear_buffer()
```

## Testing

### Run Complete Test Suite
```bash
cd /Users/jarvis/Documents/Oviya\ EI/oviya-production
python3 test_realtime_system.py
```

### Test Components
1. **Complete Pipeline**: End-to-end conversation flow
2. **Word Timestamps**: Timestamp extraction and tracking
3. **VAD Integration**: Voice activity detection
4. **Conversation Memory**: Context tracking and reset

### Expected Output
```
✓ Real-time transcription: Working
✓ Brain processing: Working
✓ Emotion mapping: Working
✓ Voice generation: Working
✓ Word-level timestamps: Working
✓ Prosodic markup: Working
✓ Emotional memory: Working
```

## Integration with Existing System

### No Changes Required To:
- ✅ CSM server (no restart needed)
- ✅ Brain/LLM logic
- ✅ Emotion controller
- ✅ Audio post-processor
- ✅ Prosodic markup system

### New Additions:
- ✅ `voice/realtime_input.py` - WhisperX integration
- ✅ `realtime_conversation.py` - Pipeline orchestration
- ✅ `test_realtime_system.py` - Comprehensive tests
- ✅ Word-level timestamp tracking
- ✅ Conversation memory system

## Dependencies

### Installed
```
whisperx              # Real-time transcription
silero-vad            # Voice Activity Detection
sounddevice>=0.4.6    # Audio recording
torch>=2.0.0          # PyTorch backend
torchaudio>=2.0.0     # Audio processing
```

### Installation
```bash
pip3 install whisperx
pip3 install git+https://github.com/snakers4/silero-vad.git
pip3 install sounddevice
```

## Performance

### Latency Targets
- Transcription: < 500ms per chunk
- Brain processing: < 1.5s
- Voice generation: < 2s
- **Total turn latency**: < 4s

### Optimization
- Streaming audio chunks (300-500ms)
- Parallel processing where possible
- GPU acceleration for WhisperX and CSM
- Efficient buffer management

## Production Deployment

### Web Interface Integration
```javascript
// Frontend: Capture audio and stream to backend
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    // Stream audio chunks to backend via WebSocket
    audioContext.createMediaStreamSource(stream);
  });

// Backend: Process audio chunks
voice_input.add_audio_chunk(audio_chunk)
```

### Mobile Integration
- Use native audio recording APIs
- Stream audio to backend via WebSocket/HTTP
- Receive voice responses and play locally

## Future Enhancements

### Planned Features
1. **Speaker Diarization**: Distinguish multiple speakers
2. **Emotion Detection from Audio**: Analyze voice tone, not just text
3. **Adaptive Pauses**: Learn user's preferred response timing
4. **Interrupt Handling**: Allow user to interrupt Oviya mid-response
5. **Multi-language Support**: Extend beyond English

### Advanced Features
1. **Prosody Mirroring**: Match user's speech patterns
2. **Contextual Energy Decay**: Adjust energy based on conversation flow
3. **Paralinguistic Sounds**: Add sighs, laughter, "mm-hmm"
4. **Adaptive Breath Timing**: Learn from user's breathing patterns

## Troubleshooting

### Common Issues

**1. WhisperX model not loading**
```bash
# Ensure CUDA is available (if using GPU)
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall WhisperX
pip3 uninstall whisperx
pip3 install whisperx
```

**2. Audio recording not working**
```bash
# Install portaudio (macOS)
brew install portaudio

# Install portaudio (Linux)
sudo apt-get install portaudio19-dev
```

**3. Transcription too slow**
```python
# Use smaller model for faster inference
voice_input = RealTimeVoiceInput()
# Modify initialize_models() to use "base" or "small" model
```

**4. Memory issues**
```python
# Reduce max buffer size
voice_input.max_buffer_seconds = 15  # Default: 30

# Clear buffer more frequently
voice_input.clear_buffer()
```

## API Reference

### RealTimeVoiceInput

#### `__init__(device: str = "cuda")`
Initialize voice input system

#### `initialize_models()`
Load WhisperX and alignment models

#### `start_recording(callback: Optional[Callable] = None)`
Start real-time recording with optional callback

#### `stop_recording() -> Optional[Dict]`
Stop recording and return final transcription

#### `add_audio_chunk(audio_chunk: np.ndarray)`
Add audio chunk for manual streaming

#### `get_transcription(timeout: float = 0.1) -> Optional[Dict]`
Get latest transcription from queue

#### `get_conversation_context() -> Dict`
Get full conversation history and context

#### `clear_buffer()`
Clear audio buffer and queues

#### `reset_conversation()`
Reset conversation history and context

### RealTimeConversation

#### `__init__(ollama_url: str, csm_url: str)`
Initialize conversation system with service URLs

#### `start_conversation()`
Start real-time conversation (blocking)

#### `stop_conversation()`
Stop conversation and show summary

#### `simulate_conversation(test_messages: list)`
Simulate conversation with pre-defined messages

## Status

✅ **COMPLETE - NO GAPS**

All components implemented:
- ✅ Real-time voice input with WhisperX
- ✅ Word-level timestamp extraction
- ✅ Voice Activity Detection (VAD)
- ✅ Conversation memory and context tracking
- ✅ Integration with existing Oviya pipeline
- ✅ Comprehensive test suite
- ✅ Documentation

Ready for production deployment!


