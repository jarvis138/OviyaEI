# Oviya - Emotional Intelligence AI Companion

> *"Hey, I'm Ovia, your emotional intelligence companion. That means I'm here to help you navigate those wild and wonderful emotions we all feel."*

Oviya is a real-time voice AI companion that provides empathetic, emotionally-aware conversations. Built with a 4-layer architecture combining advanced AI models for natural human-like interaction.

## 🏗️ Project Structure

```
oviya/
├── core/                          # Shared core components
│   ├── brain/                    # LLM brain components (personality, safety, etc.)
│   ├── voice/                    # Voice processing (TTS, prosody, etc.)
│   ├── data/                     # Data processing & bias filtering
│   ├── monitoring/               # Analytics & metrics
│   ├── validation/               # Emotion validation
│   ├── serving/                  # API endpoints
│   └── rlhf/                     # Reinforcement learning
│
├── production/                   # Main production system (monolithic)
│   ├── brain/                    # Production brain (uses core + extensions)
│   ├── voice/                    # Production voice (uses core + extensions)
│   ├── emotion_detector/         # Emotion detection
│   ├── emotion_controller/       # Emotion mapping & control
│   ├── websocket_server.py      # WebSocket API server
│   ├── realtime_conversation.py  # Real-time pipeline
│   └── docker-compose.yml        # Production deployment
│
├── services/                     # Alternative microservices architecture
│   ├── asr/                      # Speech recognition service
│   ├── vad/                      # Voice activity detection
│   ├── csm/                      # Voice synthesis service
│   ├── orchestrator/             # Request orchestration
│   └── docker-compose.yml        # Services deployment
│
├── clients/                      # Client applications
│   ├── mobile/                   # React Native mobile app
│   ├── web/                      # Web client
│   ├── website/                  # Public website
│   └── admin/                    # Admin interface
│
├── corpus/                       # Data processing pipeline
│   ├── raw/                      # Raw emotional datasets
│   ├── processed/                # Processed training data
│   └── scripts/                  # Processing scripts
│
├── mcp-ecosystem/               # Model Context Protocol ecosystem
│   ├── monitoring/               # System monitoring & analytics
│   ├── servers/                  # MCP server implementations
│   │   ├── tier1/               # Core thinking & reasoning
│   │   ├── tier2/               # Data & persistence services
│   │   └── tier3/               # External integrations
│   └── config/                   # MCP configuration files
│
├── shared/                       # Infrastructure & deployment
│   ├── docker/                   # Docker configurations
│   ├── k8s/                      # Kubernetes manifests
│   └── ci/                       # CI/CD pipelines
│
├── tests/                        # Integration tests
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Development orchestration
├── .coderabbit.yml              # CodeRabbit AI review configuration
└── README.md                     # This file
```

## 🚀 Quick Start

### Option 1: Production System (Recommended)

```bash
# 1. Install Ollama locally
brew install ollama
ollama serve &
ollama pull qwen2.5:7b

# 2. Start production system
cd production
docker-compose up -d

# 3. Access at http://localhost:8000
```

### Option 2: Development Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp production/.env production/.env.local
# Edit .env.local with your service URLs

# 3. Run locally
cd production
python websocket_server.py
```

### Option 3: Microservices Architecture

```bash
cd services
docker-compose up -d
```

## 🎯 Architecture Overview

### Production System (Monolithic)
- **WebSocket Server**: Real-time bidirectional communication
- **4-Layer Pipeline**:
  1. **Emotion Detector**: Analyzes user emotional state
  2. **Brain (LLM)**: Generates empathetic responses
  3. **Emotion Controller**: Maps emotions to acoustic parameters
  4. **Voice Engine**: Synthesizes emotional speech

### Microservices Alternative
- **ASR Service**: Speech-to-text processing
- **VAD Service**: Voice activity detection
- **CSM Service**: Voice synthesis
- **Orchestrator**: Request coordination

## 🛠️ Key Components

### Core Shared Libraries
- **Bias Filtering**: Cultural sensitivity and safety
- **Personality System**: Long-term relationship memory
- **Prosody Engine**: Natural speech patterns
- **Monitoring**: Performance analytics

### Voice & Audio
- **CSM-1B**: Conversational speech synthesis
- **OpenVoice V2**: Voice cloning and fine control
- **Real-time Processing**: <500ms transcription latency

### Brain & Intelligence
- **Qwen2.5:7B**: Primary LLM for responses
- **49 Emotion Taxonomy**: Comprehensive emotional range
- **Context Awareness**: Conversation history and memory

## 📊 Performance

- **Transcription**: <500ms latency
- **Brain Processing**: <1.5s response time
- **Voice Synthesis**: <2s total
- **End-to-end**: <4s conversational turn

## 🔧 Configuration

Environment variables control service endpoints:

```bash
# LLM Service
OLLAMA_URL=http://localhost:11434/api/generate

# Voice Services
CSM_URL=http://localhost:19517/generate
CSM_STREAM_URL=http://localhost:19517/generate/stream

# Speech Recognition
WHISPERX_URL=http://localhost:1111
```

## 🤖 CodeRabbit AI Code Reviews

Oviya uses CodeRabbit for AI-powered code reviews that validate implementation against our therapeutic AI vision. Every pull request is automatically reviewed for:

### Vision Verification Rules
- **Unconditional Positive Regard**: Ensures non-judgmental, accepting language patterns
- **Secure Base Principles**: Validates attachment theory-based emotional responses
- **Vulnerability Reciprocity**: Checks appropriate reciprocal self-disclosure
- **Bid for Connection**: Verifies micro-affirmation responses to emotional bids
- **Global Wisdom Integration**: Ensures balance between Western psychology and global traditions
- **Safety & Ethics**: Validates locked fallback responses and audit trails

### Technical Excellence
- **Real-time Performance**: <500ms transcription, <4s end-to-end latency
- **GPU Memory Management**: Proper CUDA resource handling
- **Voice Data Privacy**: Encrypted processing with consent validation
- **Emotional Continuity**: Context preservation across conversation turns

### Setup
```bash
# Add to GitHub repository secrets
OPENAI_API_KEY=your_openai_api_key_here
```

**Documentation**: [CodeRabbit Setup Guide](.github/CODE_RABBIT_SETUP.md)

## 🔄 MCP Ecosystem

Oviya includes a comprehensive Model Context Protocol (MCP) ecosystem for enhanced AI capabilities and system integration.

### Architecture Tiers

#### **Tier 1: Core Thinking & Reasoning**
- **Thinking Server**: Advanced cognitive empathy modes
- **Personality Engine**: Dynamic personality adaptation
- **Emotional Intelligence**: Deep emotional pattern recognition

#### **Tier 2: Data & Persistence**
- **PostgreSQL Server**: Structured data and conversation history
- **Redis Server**: High-performance caching and session state
- **Vector Storage**: Semantic search and memory retrieval

#### **Tier 3: External Integrations**
- **WhatsApp Server**: Social media integration
- **Stripe Server**: Payment processing and monetization
- **Analytics Server**: Usage metrics and performance monitoring

### Key Features
- **Modular Architecture**: Independent services with clear interfaces
- **Real-time Communication**: WebSocket-based inter-service messaging
- **Monitoring & Observability**: Comprehensive logging and metrics collection
- **Scalable Deployment**: Docker Compose and Kubernetes support

### Quick Start MCP
```bash
# Start the complete MCP ecosystem
cd mcp-ecosystem
docker-compose up -d

# Or start individual services
cd mcp-ecosystem/servers/tier2/postgres
docker-compose up -d
```

**Documentation**: [MCP Ecosystem Guide](mcp-ecosystem/README.md)

## 🧪 Testing

```bash
# Run integration tests
python -m pytest tests/

# Test production system
cd production
python production_sanity_tests.py

# Test specific scenarios
python tests/test_5_scenarios.py
```

## 📦 Deployment Options

### Docker (Recommended)
```bash
# Production deployment
cd production
docker-compose up -d

# Development with local Ollama
docker-compose --profile with-ollama up -d
```

### Kubernetes
```yaml
# Use manifests in shared/k8s/
kubectl apply -f shared/k8s/
```

## 🤝 Contributing

1. **Core Components**: Changes to `core/` affect all systems
2. **Production**: Main development in `production/`
3. **Services**: Microservices development in `services/`
4. **Clients**: UI/UX development in `clients/`

## 📄 License

See individual component licenses.

## 🔗 Links

- [Architecture Documentation](production/README.md)
- [API Reference](production/websocket_server.py)
- [Client Integration](clients/README.md)
- [CodeRabbit Setup Guide](.github/CODE_RABBIT_SETUP.md)
- [MCP Ecosystem Guide](mcp-ecosystem/README.md)
- [MCP Configuration](mcp-ecosystem/config/oviya-mcp-config.json)




