# Oviya EI - Emotional Intelligence Companion

> A speech-to-speech native, emotionally intelligent AI companion designed for professional mental health support with clinical safety standards and comprehensive governance frameworks.

## 🎯 Overview

Oviya is an advanced therapeutic AI system that combines:
- **CSM-1B Conversational Speech Model** - Ultra-low latency, emotional voice synthesis
- **Mimi Codec** - Real-time RVQ-level streaming for natural conversations
- **18 Therapeutic Frameworks** - CBT, DBT, EFT, Rogerian, and more
- **5-Pillar Personality System** - Ma, Ahimsa, Jeong, Logos, Lagom
- **26+ MCP Servers** - Specialized modules for mental health support
- **Clinical Safety & Governance** - HIPAA compliance, crisis detection, monitoring

## 🏗️ Architecture

### 4-Layer Architecture

1. **🎭 Therapeutic Brain Layer**
   - LLM-based response generation (Ollama + Llama 3.2:3B)
   - 18 therapeutic frameworks integration
   - Cultural wisdom adaptation (Ma, Jeong, Ahimsa, Logos, Lagom)
   - Memory and personality systems

2. **🎵 Voice Synthesis Layer**
   - CSM-1B conversational speech model
   - Real-time audio processing with RVQ streaming
   - Emotion-driven voice modulation
   - Professional audio mastering

3. **🛡️ Safety & Governance Layer**
   - Clinical safety protocols
   - Privacy protection systems
   - Experimental governance
   - Continuous monitoring

4. **🔬 MCP Ecosystem Layer**
   - 26+ specialized MCP servers
   - Mental health content generation
   - Cultural context adaptation
   - Research integration

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for CSM-1B)
- Ollama with Llama 3.2:3B model
- HuggingFace account with API token

### Installation

```bash
# Clone repository
git clone <repository-url>
cd "Oviya EI"

# Install dependencies
cd production
pip install -r requirements.txt

# Setup CSM-1B and emotion references
export HUGGINGFACE_TOKEN="your_token_here"
python3 complete_setup.py
```

### Run Server

```bash
cd production
python3 websocket_server.py
```

## 📁 Project Structure

```
Oviya EI/
├── production/          # Main production codebase
│   ├── brain/          # Therapeutic intelligence systems
│   ├── voice/          # CSM-1B, Mimi, prosody engines
│   ├── clients/        # Web, mobile, admin clients
│   ├── services/       # Microservices
│   └── mcp-ecosystem/  # 26+ MCP servers
├── core/               # Core modules
├── corpus/             # Training data
└── clients/            # Client applications
```

## 🔧 Key Features

### Speech-to-Speech Native
- Real-time audio processing with Unified VAD+STT pipeline
- CSM-1B RVQ-level streaming for ultra-low latency
- User audio captured and used for conversation context
- Emotion-expressive voice synthesis

### Emotional Intelligence
- 28+ emotion library (Tier 1-3)
- Emotion blending and interpolation
- Temporal emotion tracking
- Acoustic emotion detection
- Emotional reasoning engine

### Therapeutic Systems
- 18 therapeutic frameworks integrated
- Crisis detection and intervention
- Empathic thinking engine
- Secure base system
- Vulnerability reciprocation
- Strategic silence (Ma - 間)

### Cultural Wisdom
- **Ma (Japanese)** - Contemplative space, slower speech, pauses
- **Ahimsa (Indian)** - Compassion, warmer, gentler prosody
- **Jeong (Korean)** - Emotional connection, expressive intonation
- **Logos (Greek)** - Rational grounding, measured prosody
- **Lagom (Scandinavian)** - Balanced prosody

## 📊 Recent Updates

### Codebase Cleanup (Latest - November 2024)
- ✅ Removed ~69 redundant files
- ✅ Fixed all broken imports
- ✅ Consolidated duplicate configurations
- ✅ Optimized codebase structure
- ✅ Created comprehensive backup

### CSM-1B Integration
- ✅ CSM-1B model loading and verification
- ✅ RVQ-level streaming implementation
- ✅ Emotion reference audio system
- ✅ Multi-TTS emotion extraction
- ✅ Conversation context formatting

### Emotional Intelligence Enhancements
- ✅ Emotion embeddings system
- ✅ Temporal emotion tracking
- ✅ Emotional reasoning engine
- ✅ Emotion blender and library
- ✅ Cultural wisdom integration

## 🧪 Testing

```bash
cd production
python3 test_complete_pipeline.py
```

## 📚 Documentation

- [Setup Guide](production/SETUP_COMPLETE.md)
- [CSM-1B Verification](production/CSM_1B_VERIFICATION.md)
- [4-Layer Architecture](production/4_LAYER_ARCHITECTURE_VERIFICATION.md)
- [Cleanup Summary](production/CLEANUP_COMPLETE.md)
- [Implementation Status](production/IMPLEMENTATION_COMPLETE.md)

## 🔒 Safety & Compliance

- HIPAA-compliant architecture
- Clinical safety protocols
- Crisis detection and intervention
- Privacy protection systems
- Audit trails and monitoring

## 🤝 Contributing

Please read our contributing guidelines before submitting PRs.

## 📄 License

[License information]

## 🙏 Acknowledgments

- Sesame AI for CSM-1B model
- Hugging Face for model hosting
- All open-source contributors

---

**Status**: ✅ Production Ready  
**Last Updated**: November 2024  
**Version**: 1.0.0
