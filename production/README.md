# Oviya Production System

The main production deployment of Oviya - a complete, monolithic emotional AI system.

## Quick Start

### Prerequisites

1. **Ollama** (for LLM):
   ```bash
   brew install ollama
   ollama serve &
   ollama pull qwen2.5:7b
   ```

2. **Voice Services** (CSM TTS):
   - Either run locally or use cloud endpoints
   - Default: `http://localhost:19517/generate`

3. **Speech Recognition** (WhisperX):
   - Either run locally or use cloud endpoints
   - Default: `http://localhost:1111`

### Environment Setup

```bash
# Copy environment template
cp .env .env.local

# Edit with your service URLs
OLLAMA_URL=http://localhost:11434/api/generate
CSM_URL=http://localhost:19517/generate
WHISPERX_URL=http://localhost:1111
```

### Run Production System

```bash
# Using Docker (recommended)
docker-compose up -d

# Or run locally
python websocket_server.py
```

## Architecture

### 4-Layer Pipeline

```
User Input → [1] Emotion Detector → [2] Brain (LLM) → [3] Emotion Controller → [4] Voice Output
```

#### Layer 1: Emotion Detector
- **Input**: User text/audio
- **Output**: Emotion labels + confidence scores
- **Models**: Rule-based + acoustic features

#### Layer 2: Brain (LLM)
- **Input**: User message + emotion context
- **Output**: Empathetic response text + emotion label
- **Model**: Qwen2.5:7B with personality conditioning

#### Layer 3: Emotion Controller
- **Input**: Emotion labels + intensity
- **Output**: Acoustic parameters (pitch, rate, energy)
- **Library**: 49 emotion taxonomy → CSM parameters

#### Layer 4: Voice Engine
- **Input**: Text + acoustic parameters
- **Output**: Emotional speech audio
- **Models**: CSM-1B + OpenVoice V2 hybrid

### WebSocket API

Real-time bidirectional communication:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/conversation?user_id=user123');

// Send audio data
ws.send(JSON.stringify({
  type: 'audio',
  data: base64AudioData
}));

// Receive responses
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  // Handle emotional response with audio
};
```

## Configuration

### Core Configuration Files

- **`config/oviya_persona.json`**: Personality traits and response guidelines
- **`config/emotions_49.json`**: 49 emotion taxonomy
- **`config/service_urls.py`**: Service endpoint configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | LLM service endpoint |
| `CSM_URL` | `http://localhost:19517/generate` | Voice synthesis endpoint |
| `CSM_STREAM_URL` | `http://localhost:19517/generate/stream` | Streaming voice endpoint |
| `WHISPERX_URL` | `http://localhost:1111` | Speech recognition endpoint |

## Development

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Integration tests
python production_sanity_tests.py

# Specific scenarios
python tests/test_5_scenarios.py
```

### Adding New Emotions

1. Update `config/emotions_49.json`
2. Add mapping in `emotion_controller/controller.py`
3. Test with `emotion_controller/tests/`

### Voice Cloning

```bash
# Clone a voice for Oviya
python -c "
from voice.openvoice_tts import HybridVoiceEngine
engine = HybridVoiceEngine()
engine.clone_voice('path/to/reference_audio.wav', 'oviya_v1')
"
```

## Monitoring

### Built-in Metrics

- **Latency Tracking**: Response times per component
- **Emotion Distribution**: Monitor emotional range usage
- **User Satisfaction**: Optional feedback collection
- **Error Rates**: System reliability monitoring

### Logs

```bash
# View logs
docker-compose logs -f oviya-backend

# Or locally
tail -f logs/*.log
```

## Deployment

### Docker Production

```bash
# Build and deploy
docker-compose up -d --build

# Scale services
docker-compose up -d --scale oviya-backend=3

# Update
docker-compose pull && docker-compose up -d
```

### Health Checks

```bash
# Check all services
curl http://localhost:8000/health

# Individual components
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:19517/health    # CSM
curl http://localhost:1111/health     # WhisperX
```

## Troubleshooting

### Common Issues

**"Connection refused" errors:**
- Check service URLs in `.env`
- Ensure external services are running
- Verify network connectivity

**High latency:**
- Check GPU availability for voice synthesis
- Monitor Ollama model loading
- Review network bandwidth

**Emotion detection issues:**
- Verify `config/emotions_49.json` format
- Check acoustic feature extraction
- Review confidence thresholds

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH=/app
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from websocket_server import app
# Run with debug flags
"
```

## 📈 Performance Tuning

### Optimization Areas

1. **Model Caching**: Pre-load models to reduce startup time
2. **Connection Pooling**: Reuse connections to external services
3. **Async Processing**: Parallelize emotion detection and synthesis
4. **Response Caching**: Cache similar responses

### Benchmarking

```bash
# Run performance tests
python tools/ttfb_interrupt_test.py

# Monitor resource usage
python monitoring.py
```

## 🔗 Related Systems

- **`core/`**: Shared components used by this system
- **`services/`**: Alternative microservices architecture
- **`clients/`**: Frontend applications that connect here
- **`corpus/`**: Data processing that feeds this system