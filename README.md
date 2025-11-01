# Oviya EI - Emotional Intelligence Companion

> A speech-to-speech native, emotionally intelligent AI companion designed for professional mental health support with clinical safety standards and comprehensive governance frameworks.

## 🎯 Overview

Oviya EI is an advanced therapeutic AI system that combines:
- **CSM-1B Conversational Speech Model** - Ultra-low latency, emotional voice synthesis with RVQ streaming
- **Mimi Codec** - Real-time audio decoding for natural conversations
- **18 Therapeutic Frameworks** - CBT, DBT, EFT, Rogerian, and more integrated into LLM responses
- **5-Pillar Personality System** - Ma, Ahimsa, Jeong, Logos, Lagom influencing prosody and responses
- **26+ MCP Servers** - Specialized modules for mental health support, thinking, and therapy
- **Unified VAD+STT Pipeline** - Optimized Silero VAD and Whisper v3 Turbo for real-time speech processing
- **Clinical Safety & Governance** - HIPAA compliance, crisis detection, monitoring

## 🏗️ Architecture

### 4-Layer Architecture

1. **🎭 Therapeutic Brain Layer**
   - **LLM**: Ollama + Llama 3.2:3B for response generation
   - **18 Therapeutic Frameworks**: CBT, DBT, EFT, Rogerian, Attachment Theory, Secure Base, etc.
   - **Cultural Wisdom**: Ma (Japanese), Jeong (Korean), Ahimsa (Indian), Logos (Greek), Lagom (Scandinavian)
   - **Memory Systems**: ChromaDB for long-term memory and personality evolution
   - **Emotional Intelligence**: Emotion embeddings, temporal tracking, emotional reasoning

2. **🎵 Voice Synthesis Layer**
   - **CSM-1B**: Conversational speech model with RVQ-level streaming
   - **Mimi Codec**: Real-time audio decoding from RVQ tokens
   - **Prosody Engine**: Personality-driven voice modulation (pitch, rate, energy)
   - **Emotion References**: Multi-TTS emotion reference system for CSM-1B conditioning
   - **Unified VAD+STT**: Silero VAD + Whisper v3 Turbo for speech processing

3. **🛡️ Safety & Governance Layer**
   - **Crisis Detection**: AI Therapist MCP integration for mental health safety
   - **PII Redaction**: HIPAA-compliant privacy protection
   - **Therapeutic Boundaries**: Ethical AI-human interaction limits
   - **Audit Trails**: Complete logging for clinical oversight

4. **🔬 MCP Ecosystem Layer**
   - **AI Therapist MCP**: Crisis intervention, coping strategies, positive affirmations
   - **MCP Thinking**: Enhanced sequential thinking, dialectical reasoning
   - **26+ MCP Servers**: Mental health, psychology, cultural adaptation

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for CSM-1B)
- Ollama with Llama 3.2:3B model installed
- HuggingFace account with API token

### Installation

```bash
# Clone repository
git clone <repository-url>
cd "Oviya EI"

# Install dependencies
cd production
pip install -r requirements.txt

# Setup environment
export HUGGINGFACE_TOKEN="your_token_here"
export OVIYA_SECRET="your_secret_key"

# Verify CSM-1B installation (optional)
python3 verify_csm_installation.py
```

### Run Server

```bash
cd production
python3 websocket_server.py
```

The server will start on `http://localhost:8000` with WebSocket support at `ws://localhost:8000/ws`.

## 📁 Project Structure

```
Oviya EI/
├── production/              # Main production codebase
│   ├── brain/              # Therapeutic intelligence
│   │   ├── llm_brain.py   # Main LLM brain with 18 frameworks
│   │   ├── crisis_detection.py
│   │   ├── empathic_thinking.py
│   │   ├── emotional_reciprocity.py
│   │   └── ...            # 20+ brain modules
│   ├── voice/             # Voice synthesis
│   │   ├── csm_1b_stream.py      # CSM-1B RVQ streaming
│   │   ├── unified_vad_stt.py    # VAD+STT pipeline
│   │   ├── emotion_controller.py
│   │   ├── prosody_engine.py
│   │   └── ...            # Voice processing modules
│   ├── websocket_server.py        # Main WebSocket server
│   ├── voice_server_webrtc.py     # WebRTC server (alternative)
│   └── ...
├── clients/               # Client applications
│   ├── web/              # Next.js web client
│   ├── mobile/           # React Native mobile app
│   └── admin/            # Admin dashboard
├── services/             # Microservices
│   └── services/         # ASR, TTS, orchestration services
├── mcp-ecosystem/        # MCP servers
│   └── servers/          # 26+ MCP servers
├── core/                 # Core modules
└── corpus/               # Training data
```

## 🔧 Key Features

### Speech-to-Speech Native
- **Unified VAD+STT Pipeline**: Silero VAD (ONNX optimized) + Whisper v3 Turbo (`faster-whisper`)
- **Real-time Audio Processing**: User audio captured, processed, and used for conversation context
- **CSM-1B RVQ Streaming**: Ultra-low latency (<120ms) voice generation with RVQ tokens
- **Mimi Decode**: Real-time audio decoding from RVQ tokens to PCM audio
- **User Audio Context**: User's spoken audio included in CSM-1B conversation context

### Emotional Intelligence
- **Emotion Detection**: Text-based and acoustic emotion detection
- **Emotion Library**: 28+ emotions across 3 tiers (Tier 1: Core, Tier 2: Nuanced, Tier 3: Complex)
- **Emotion Blender**: Emotion interpolation for expanded emotional range
- **Temporal Emotion Tracking**: Track emotion patterns over time
- **Emotional Reasoning**: Advanced emotional reasoning and inference
- **Emotion Embeddings**: Real audio/text-based embeddings for emotional intelligence

### Therapeutic Systems
- **18 Therapeutic Frameworks**: 
  - CBT, DBT, EFT, Rogerian (Person-Centered)
  - Attachment Theory, Secure Base Theory
  - Unconditional Positive Regard
  - Vulnerability Reciprocation
  - Strategic Silence (Ma - 間)
  - Empathic Thinking, Emotional Reciprocity
  - Crisis Intervention, Micro-Affirmations
  - Healthy Boundaries, Epistemic Prosody
  - Emotion Transition Smoothing, Backchannel System
- **Crisis Detection**: AI Therapist MCP integration for mental health safety
- **Empathic Thinking**: MCP Thinking server for deep cognitive empathy
- **Memory System**: ChromaDB for long-term therapeutic relationship building

### Cultural Wisdom Integration
- **Ma (Japanese - 間)**: Contemplative space → slower speech, more pauses
- **Ahimsa (Indian)**: Compassion → warmer, gentler prosody
- **Jeong (Korean)**: Emotional connection → more expressive intonation
- **Logos (Greek)**: Rational grounding → more measured, stable prosody
- **Lagom (Scandinavian)**: Balanced prosody

### Voice Synthesis
- **CSM-1B Integration**: Sesame's conversational speech model
- **Prosody Control**: Personality-driven modulation (pitch_scale, rate_scale, energy_scale)
- **Emotion References**: Multi-TTS emotion reference system (OpenVoiceV2, Coqui TTS, Bark, StyleTTS2)
- **RVQ Streaming**: Token-level streaming for ultra-low latency
- **CUDA Graphs**: Optimization for consistent low-latency performance

## 📊 Recent Updates (November 2024)

### Codebase Cleanup
- ✅ Removed ~69 redundant files (historical docs, duplicates, old audio)
- ✅ Fixed broken imports (OptimizedCSMStreamer, SessionManager)
- ✅ Consolidated duplicate configurations
- ✅ Optimized codebase structure
- ✅ Created comprehensive backup

### CSM-1B Integration
- ✅ CSM-1B model loading and verification
- ✅ RVQ-level streaming implementation
- ✅ Mimi codec integration for audio decoding
- ✅ Conversation context formatting with audio references
- ✅ Prosody parameter control (pitch, rate, energy)

### Speech-to-Speech Native
- ✅ Unified VAD+STT pipeline (Silero + Whisper)
- ✅ User audio capture and processing
- ✅ User audio included in CSM-1B conversation context
- ✅ Real-time audio streaming

### Emotional Intelligence Enhancements
- ✅ Emotion embeddings system
- ✅ Temporal emotion tracking
- ✅ Emotional reasoning engine
- ✅ Emotion blender and library (28+ emotions)
- ✅ Cultural wisdom integration into prosody

### MCP Integration
- ✅ AI Therapist MCP integration (crisis intervention, coping strategies)
- ✅ MCP Thinking server integration (enhanced thinking, dialectical reasoning)
- ✅ Real MCP client implementation (replacing mock clients)

## 🧪 Testing

```bash
cd production

# Test complete pipeline
python3 test_complete_pipeline.py

# Test CSM-1B integration
python3 tests/test_csm_1b.py

# Test brain systems
python3 tests/test_brain_simple.py

# Test realtime system
python3 tests/test_realtime_system.py
```

## 📚 Documentation

- [Setup Guide](production/SETUP_COMPLETE.md) - Complete setup instructions
- [CSM-1B Verification](production/CSM_1B_VERIFICATION.md) - CSM-1B integration guide
- [Architecture Verification](production/4_LAYER_ARCHITECTURE_VERIFICATION.md) - Architecture details
- [Cleanup Summary](production/CLEANUP_COMPLETE.md) - Codebase cleanup documentation
- [Implementation Status](production/IMPLEMENTATION_COMPLETE.md) - Implementation details

## 🔒 Safety & Compliance

- **HIPAA-Compliant**: PII redaction and privacy protection
- **Clinical Safety**: Crisis detection and intervention protocols
- **Therapeutic Boundaries**: Ethical AI-human interaction limits
- **Audit Trails**: Complete logging for clinical oversight
- **Crisis Resources**: Automatic emergency resource provision

## 🔌 API Usage

### WebSocket API

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Send audio chunk
ws.send(JSON.stringify({
  type: 'audio',
  audio_base64: base64AudioData,
  sample_rate: 16000
}));

// Receive response
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'audio') {
    // Play audio response
    playAudio(data.audio_base64);
  } else if (data.type === 'text') {
    // Display text
    console.log(data.text);
  }
};
```

### Python API

```python
from production.brain.llm_brain import OviyaBrain

# Initialize brain
brain = OviyaBrain()

# Generate therapeutic response
response = brain.think(
    user_message="I'm feeling really anxious about work",
    conversation_history=[],
    memory_triples=[]
)

print(response["text"])  # Therapeutic response
print(response["emotion"])  # Detected emotion
print(response["personality_vector"])  # Personality vector
```

## 🛠️ Development

### Key Components

- **`production/websocket_server.py`**: Main WebSocket server for real-time conversations
- **`production/brain/llm_brain.py`**: Core therapeutic intelligence with 18 frameworks
- **`production/voice/csm_1b_stream.py`**: CSM-1B RVQ streaming implementation
- **`production/voice/unified_vad_stt.py`**: Unified VAD+STT pipeline
- **`production/prosody_engine.py`**: Prosody computation for voice modulation

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement following Oviya's architecture patterns
3. Add tests: `production/tests/test_your_feature.py`
4. Update documentation
5. Submit PR with clear description

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Follow code style and architecture patterns
4. Add tests for new features
5. Submit a pull request

## 📄 License

[License information]

## 🙏 Acknowledgments

- **Sesame AI** - CSM-1B conversational speech model
- **Hugging Face** - Model hosting and infrastructure
- **OpenVoiceV2, Coqui TTS, Bark** - Emotion reference generation
- **MCP Ecosystem** - AI Therapist and Thinking servers
- All open-source contributors

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/oviya-ei/issues)
- **Documentation**: See `production/` directory for detailed docs
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/oviya-ei/discussions)

---

**Status**: ✅ Production Ready  
**Last Updated**: November 2024  
**Version**: 1.0.0

*Built with ❤️ for mental health and emotional intelligence*
