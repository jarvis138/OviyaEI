# Oviya - AI Emotional Companion

> ChatGPT-style real-time voice mode with empathetic responses

## 🎯 Mission

Build the world's most empathetic AI companion with <2s latency and natural voice conversations.

## 🏗️ Architecture

- **CSM TTS**: Emotional voice generation with audio context
- **Silero VAD**: <50ms interrupt detection
- **Whisper ASR**: Real-time transcription
- **Gemini 2.0 Flash**: Fast, cost-effective LLM
- **Real-time Streaming**: WebSocket-based duplex audio

## 📊 Current Status

**Sprint 0: Technical Validation** (Week 1-2)
- [ ] CSM model validation
- [ ] Audio context emotion testing
- [ ] Streaming implementation
- [ ] Silero VAD integration
- [ ] GO/NO-GO decision

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU or RunPod account
- Gemini API key
- Hugging Face account (for CSM access)

### Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-base.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run validation tests
cd validation/csm-benchmark
python basic_test.py
```

## 📁 Project Structure

```
oviya-ai/
├── validation/          # Sprint 0 - Technical validation
├── services/           # Backend microservices
├── apps/              # Frontend applications
├── infrastructure/    # Deployment configs
└── docs/             # Documentation
```

## 🔑 Environment Variables

See `.env.example` for required configuration.

## 📚 Documentation

- [Architecture Plan](../plan-v4.md)
- [Execution Framework](../execution-framework.md)
- [Web Frontend Plan](../web-frontend-plan.md)
- [Gap Analysis](../gap-analysis.md)

## 📈 Success Metrics

- End-to-end latency: <2100ms (p99)
- Interrupt response: <150ms
- Emotion quality: >80% accuracy
- User satisfaction: >4/5 stars

## 🤝 Contributing

We follow a sprint-based development process. See [execution-framework.md](../execution-framework.md) for details.

## 📄 License

Proprietary - All rights reserved

---

**Current Sprint**: Sprint 0 - Technical Validation
**Timeline**: 20 weeks to beta launch
**Team**: Solo developer + AI assistance
