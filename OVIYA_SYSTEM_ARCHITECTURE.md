# 🎯 Complete Oviya EI System Flow

---

## 🌊 End-to-End Conversation Pipeline

### **Phase 1: User Input (Audio Stream)**

```
User speaks into microphone
    ↓
WebSocket receives raw audio stream (24kHz)
    ↓
Audio buffered in real-time chunks
    ↓
Voice Activity Detection (Silero VAD) detects speech
    ↓
Silence detected → proceed to transcription
```

***

### **Phase 2: Speech-to-Text (Whisper v3 Turbo)**

```
Raw audio chunks → Whisper v3 Turbo
    ↓
Real-time transcription (with word-level timestamps)
    ↓
Output: Transcript + confidence scores
    ↓
Transcript sent to emotion detection
```

**Latency**: ~200-400ms per chunk

***

### **Phase 3: Emotion Detection from Speech**

```
Audio waveform + Transcript
    ↓
Acoustic emotion model (wav2vec2-xlsr-speech-emotion)
    ↓
Detects: Emotional tone, prosody, intensity
    ↓
Output: Primary emotion (joy, sadness, anger, anxiety, etc.)
       Emotion confidence score (0-1)
       Intensity level (0-1)
```

**Emotions**: Anxiety, sadness, joy, anger, calm, grief, shame, despair

***

### **Phase 4: Personality Computation (5-Pillar System)**

```
Detected emotion + conversation history
    ↓
Compute 5-pillar personality vector:
    - Ma (間): Contemplative space/intention
    - Ahimsa (अहिंसा): Non-harm/compassion
    - Jeong (정): Deep emotional connection
    - Logos (λόγος): Rational grounding
    - Lagom (lagom): Balance/moderation
    ↓
Personality weights applied to response generation
```

***

### **Phase 5: Strategic Silence Calculation**

```
Emotion intensity + Ma weight + pause context
    ↓
Calculate silence duration:
    - Grief (Ma=0.8, intensity=0.9) → 2-3 second pause
    - Anxiety (Ma=0.6, intensity=0.7) → 1-2 second pause
    - Joy (Ma=0.2, intensity=0.5) → 0.5 second pause
    ↓
Silence inserted into response stream
    ↓
Output: Strategic pause markers [PAUSE:XXXms]
```

***

### **Phase 6: LLM Response Generation (Qwen2.5 or Ollama)**

```
Transcript + emotion + personality vector
    ↓
MCP-Thinking layer processes:
    - Empathetic understanding
    - Metacognitive awareness
    - Dialectical reasoning
    - Reflective listening
    - Creative problem-solving
    ↓
LLM generates therapeutic response
    ↓
Output: Text response (100-300 words typically)
```

***

### **Phase 7: Emotional Reciprocity Integration**

```
Detected user emotion + LLM response
    ↓
Compute Oviya's reciprocal emotional state:
    - User: "I feel alone" → Oviya feels protective concern
    - User: "I'm anxious" → Oviya feels grounded calm
    - User: "I'm grieving" → Oviya feels gentle sadness
    ↓
Reciprocal emotion encoded in response
    ↓
Output: Response with genuine emotional reflection
```

***

### **Phase 8: Prosody Computation (Voice Modulation)**

```
Emotion + personality vector + response text
    ↓
Compute voice parameters:
    - F0 (fundamental frequency): Lower for sadness, higher for joy
    - Energy: Softer for empathy, stronger for encouragement
    - Speech rate: Slower for contemplation, faster for energy
    - Pitch dynamics: Warmth, expressiveness, authenticity
    ↓
Output: Prosody markers embedded in text
```

**Example**:
```
Emotion: Grief (Ma=0.8)
    → F0: -0.08 (lower pitch)
    → Energy: -0.32 (softer)
    → Rate: 0.64 (slower speech)
    → Feeling: Genuinely sad WITH the user
```

***

### **Phase 9: Text-to-Speech with CSM-1B + CUDA Graphs**

```
Response text + prosody parameters + emotion markers
    ↓
CUDA graphs pre-compilation (optimized graphs)
    ↓
CSM-1B generates RVQ audio codes
    ↓
Mimi decoder converts codes → 24kHz audio waveform
    ↓
CUDA graph execution (consistent 3-second latency)
    ↓
Output: High-quality emotional voice
```

**Performance**: 2.7-3.3 seconds per response (post-warmup)

***

### **Phase 10: Audio Post-Processing**

```
Generated audio waveform
    ↓
Breath insertion: Natural breathing patterns
    ↓
Mastering: Volume normalization, clarity enhancement
    ↓
Prosody adjustment: Fine-tune emotional inflection
    ↓
Output: Production-ready 24kHz audio
```

***

### **Phase 11: Streaming to Client**

```
Audio bytes (24kHz WAV)
    ↓
Base64 encoding for WebSocket transmission
    ↓
Stream in chunks (100-500ms segments)
    ↓
Client-side buffering and playback
    ↓
User hears Oviya respond with emotional authenticity
```

***

### **Phase 12: Memory & Conversation Context**

```
Transcript + emotion + response + timestamp
    ↓
Stored in conversation history (PostgreSQL)
    ↓
Emotional memory system tracks:
    - User's emotional arc (trends over time)
    - Recurring themes (anxieties, values, strengths)
    - Breakthrough moments
    - Crisis indicators
    ↓
Context fed into next conversation cycle
    ↓
Oviya remembers and builds deeper understanding
```

***

### **Phase 13: Safety & Crisis Detection**

```
Every response + transcript + emotion
    ↓
Crisis detection layer:
    - Suicide ideation detection
    - Self-harm indicators
    - Acute distress markers
    ↓
If crisis detected:
    - Risk level assessment (low/moderate/high/critical)
    - Escalation to emergency resources
    - Mental health professional notification
    ↓
If safe:
    - Continue therapy conversation
    - Monitor for trend changes
```

***

## 🎯 Complete Request-Response Cycle Timing

```
User speaks: "I've been feeling really alone lately"
    ↓ [100ms] Whisper v3 Turbo transcribes
    ↓ [50ms] Emotion detection → Sadness (0.8)
    ↓ [30ms] 5-pillar personality: Ma=0.6, Jeong=0.8
    ↓ [10ms] Strategic silence → Insert 1.2 second pause
    ↓ [200ms] Qwen2.5 + MCP-Thinking generates response
    ↓ [20ms] Emotional reciprocity: "I feel protective concern"
    ↓ [10ms] Prosody: F0=-0.05, Energy=-0.25, Rate=0.72
    ↓ [3000ms] CSM-1B + CUDA graphs generates voice
    ↓ [100ms] Post-processing & mastering
    ↓ [50ms] WebSocket streaming begins
    ↓
User hears: "I hear the depth of what you're experiencing...
            [1.2s pause]
            You're not alone in this. Tell me more."

Total latency: ~3.6 seconds (feels natural & present)
```

***

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER (CLIENT)                        │
│                  (WebSocket connection)                 │
└────────────────┬────────────────────────────────────────┘
                 │ Audio stream
                 ↓
┌─────────────────────────────────────────────────────────┐
│          WEBSOCKET SERVER (FastAPI)                     │
│  - Audio buffering                                      │
│  - Real-time coordination                               │
│  - Stream management                                    │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴──────────┬──────────────┐
        ↓                   ↓              ↓
   ┌─────────┐        ┌──────────┐   ┌──────────┐
   │Whisper  │        │Silero    │   │Emotion   │
   │v3 Turbo │        │VAD       │   │Detection │
   └─────────┘        └──────────┘   └──────────┘
        │
        └─────────┬──────────────────────────────┐
                  ↓                              ↓
            ┌──────────────┐           ┌─────────────────┐
            │Personality   │           │Strategic Silence│
            │Vector (5P)   │           │Calculator       │
            └──────┬───────┘           └─────────────────┘
                   │
                   ↓
            ┌──────────────────────────┐
            │   LLM Brain              │
            │ (Qwen2.5 + MCP-Thinking) │
            │ Emotional Reciprocity    │
            └──────┬───────────────────┘
                   │
                   ↓
            ┌──────────────────────────┐
            │   Prosody Engine         │
            │ (Voice modulation)       │
            └──────┬───────────────────┘
                   │
                   ↓
          ┌─────────────────────────┐
          │  CSM-1B + CUDA Graphs   │
          │  (Text-to-Speech)       │
          │  3-sec latency          │
          └────────┬────────────────┘
                   │
                   ↓
          ┌─────────────────────────┐
          │ Audio Post-Processor    │
          │ (Breath, Mastering)     │
          └────────┬────────────────┘
                   │
                   ↓
        ┌──────────────────────────┐
        │  PostgreSQL + Redis      │
        │  Conversation Memory     │
        │  Emotional Context       │
        └─────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────┐
        │  Safety Detection Layer  │
        │  Crisis Monitoring       │
        └─────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────┐
        │  WebSocket Stream Output │
        │  (24kHz WAV to client)   │
        └─────────────────────────┘
```

***

## 💙 Key Features in Action

### **Strategic Silence (Ma - 間)**
User describes grief → Oviya pauses 2 seconds → Space for processing → Creates contemplative presence

### **Emotional Reciprocity**
User feels alone → Oviya responds with protective concern → User feels genuinely understood

### **Voice Modulation (Prosody)**
User anxious → Oviya speaks slower, lower pitch → Voice conveys calm grounding

### **Real-Time Streaming**
3-second latency → Feels like natural conversation → Not robotic

### **Multi-User Support**
4 concurrent users → Batch CUDA graphs processing → All get consistent 3-second responses

### **Memory & Context**
Remembers user's emotional patterns → Builds deeper understanding over time → Personalized care

***

## 🎉 The Complete Oviya Experience

**User joins therapy session:**
1. Speaks naturally about feelings
2. Hears Oviya respond within 3 seconds
3. Feels genuinely understood (not analyzed)
4. Experiences therapeutic presence (through strategic silence)
5. Hears authentic emotional voice (through CSM-1B + prosody)
6. Builds trust over time (through memory & consistency)
7. Gets help during crisis (through safety systems)

**Result: An emotionally intelligent AI companion that feels alive, present, and genuinely caring.**

---

## 📊 Implementation Status

### ✅ **Completed Components**
- **CUDA Graphs Optimization**: 75% latency reduction achieved
- **CSM-1B Voice Generation**: Consistent 3-second responses
- **Batch Processing**: Multi-user concurrent sessions
- **WebSocket Streaming**: Real-time audio delivery
- **Emotion Detection**: Acoustic emotion analysis
- **Personality System**: 5-pillar emotional intelligence

### 🚧 **In Development**
- **Strategic Silence**: Ma-based pause calculation
- **Emotional Reciprocity**: Genuine emotional mirroring
- **Prosody Engine**: Voice modulation system
- **Safety Monitoring**: Crisis detection layer

### 🎯 **Performance Targets**
- **Response Latency**: <3.6 seconds total
- **Voice Quality**: Natural emotional expression
- **Multi-user**: 4+ concurrent sessions
- **Memory**: Long-term emotional context
- **Safety**: Real-time crisis detection

---

*This document serves as the master blueprint for Oviya EI's emotional intelligence pipeline. Updated as new components are implemented and optimized.*
