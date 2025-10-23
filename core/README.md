# Oviya Core Components

Shared libraries and utilities used across all Oviya systems (production, services, clients).

## 📁 Directory Structure

```
core/
├── brain/                    # LLM brain components
│   ├── attachment_style.py   # Personality attachment patterns
│   ├── auto_decider.py       # Situation/emotion detection
│   ├── backchannels.py       # Micro-affirmations
│   ├── bids.py              # Bid-for-connection responses
│   ├── consistent_persona.py # Personality consistency
│   ├── emotion_transitions.py # Smooth emotion changes
│   ├── epistemic_prosody.py  # Uncertainty/prosody
│   ├── global_soul.py       # Cultural wisdom integration
│   ├── healthy_boundaries.py # Usage boundary management
│   ├── llm_brain.py         # Main brain logic (imported)
│   ├── personality_store.py # Long-term memory
│   ├── personality_vector.py # Personality embeddings
│   ├── relationship_memory.py # Relationship tracking
│   ├── safety_router.py     # Safety response routing
│   ├── secure_base.py       # Secure base behaviors
│   ├── unconditional_regard.py # Non-judgmental responses
│   └── vulnerability.py     # Vulnerability reciprocity
│
├── voice/                    # Voice processing components
│   ├── acoustic_emotion_detector.py # Audio emotion analysis
│   ├── audio_postprocessor.py # Audio mastering/prosody
│   ├── csm_1b_client.py     # CSM TTS client
│   ├── csm_1b_stream.py     # Streaming CSM client
│   ├── csm_streaming_pipeline.py # Streaming pipeline
│   ├── csm_style_adapter.py # Style adaptation
│   ├── emotion_blender.py   # Emotion mixing
│   ├── emotion_library.py   # Emotion taxonomy
│   ├── emotion_teacher.py   # Voice cloning
│   ├── humanlike_prosody.py # Natural speech patterns
│   ├── micro_affirmations.py # Backchannel responses
│   ├── opens2s_tts.py       # OpenVoice V2 integration
│   ├── openvoice_tts.py     # Hybrid voice engine
│   ├── prosody_controller.py # Prosody control
│   ├── realtime_input.py    # Real-time audio input
│   ├── session_state.py     # Session management
│   ├── silero_vad_adapter.py # Voice activity detection
│   └── whisper_client.py    # Whisper integration
│
├── data/                     # Data processing
│   ├── bias_filter.py       # Cultural bias filtering
│   ├── emotion_taxonomy.json # Emotion definitions
│   └── expressive_audio/    # Audio dataset processing
│       ├── build_metadata.py
│       └── download_datasets.sh
│
├── monitoring/               # Analytics & observability
│   ├── analytics_pipeline.py # Conversation analytics
│   └── psych_metrics.py     # Psychological metrics
│
├── validation/               # Testing & validation
│   └── emotion_validator.py # Emotion validation framework
│
├── serving/                  # API endpoints
│   └── conditioning_api.py  # Personality conditioning API
│
└── rlhf/                     # Reinforcement learning
    └── feedback_store.py    # User feedback storage
```

## 🎯 Purpose

Core components provide:

- **Shared Business Logic**: Common algorithms used across systems
- **Reusability**: Avoid code duplication between production/services
- **Consistency**: Unified implementations of key features
- **Maintainability**: Single source of truth for critical components

## 🔧 Key Components

### Brain Components

#### Personality System
```python
from core.brain.personality_store import PersonalityStore
from core.brain.personality_vector import PersonalityEMA

# Long-term memory and personality adaptation
store = PersonalityStore()
ema = PersonalityEMA()  # Exponential moving average for personality vectors
```

#### Safety & Ethics
```python
from core.brain.safety_router import SafetyRouter
from core.data.bias_filter import CulturalBiasFilter

# Safety response routing and bias filtering
router = SafetyRouter(persona_config)
filter = CulturalBiasFilter()
```

#### Emotional Intelligence
```python
from core.brain.auto_decider import AutoDecider
from core.brain.epistemic_prosody import EpistemicProsodyAnalyzer

# Situation analysis and prosody adaptation
decider = AutoDecider(persona_config)
prosody_analyzer = EpistemicProsodyAnalyzer()
```

### Voice Components

#### Speech Synthesis
```python
from core.voice.openvoice_tts import HybridVoiceEngine
from core.voice.csm_1b_client import CSM1BClient

# Hybrid voice synthesis (CSM + OpenVoice)
engine = HybridVoiceEngine()
csm_client = CSM1BClient()
```

#### Audio Processing
```python
from core.voice.audio_postprocessor import AudioPostProcessor
from core.voice.acoustic_emotion_detector import AcousticEmotionDetector

# Audio mastering and emotion detection
processor = AudioPostProcessor()
emotion_detector = AcousticEmotionDetector()
```

### Data Components

#### Bias & Safety
```python
from core.data.bias_filter import CulturalBiasFilter

# Cultural sensitivity filtering
filter = CulturalBiasFilter()
is_safe, details = filter.filter_sample(sample)
```

### Monitoring Components

#### Analytics
```python
from core.monitoring.analytics_pipeline import AnalyticsPipeline

# Conversation analytics and metrics
pipeline = AnalyticsPipeline()
pipeline.track_conversation(conversation_data)
```

## 🔄 Usage in Systems

### Production System (`production/`)
```python
# production/brain/llm_brain.py
from core.brain.safety_router import SafetyRouter
from core.data.bias_filter import CulturalBiasFilter
from core.monitoring.psych_metrics import BIAS_FILTER_DROP

# Extends core components with production-specific logic
```

### Services System (`services/`)
```python
# services/orchestrator/
from core.brain.auto_decider import AutoDecider
from core.voice.micro_affirmations import MicroAffirmationGenerator

# Uses core components in microservices architecture
```

### Client Applications (`clients/`)
```python
# clients/web/
from core.serving.conditioning_api import router

# Uses core APIs for client-server communication
```

## 🧪 Testing Core Components

```bash
# Test all core components
cd core
python -m pytest tests/ -v

# Test specific component
python -c "from core.brain.auto_decider import AutoDecider; print('Import successful')"
```

## 📦 Dependencies

Core components have minimal external dependencies:

- **torch**: Machine learning framework
- **numpy**: Numerical computing
- **requests**: HTTP client
- **fastapi**: API framework (serving components)
- **pytest**: Testing framework

## 🔧 Development Guidelines

### Adding New Components

1. **Place in appropriate subdirectory** (`brain/`, `voice/`, `data/`, etc.)
2. **Follow naming conventions** (snake_case for files, CamelCase for classes)
3. **Add comprehensive docstrings** and type hints
4. **Include unit tests** in `tests/` subdirectory
5. **Update this README** with new component descriptions

### Component Design Principles

- **Single Responsibility**: Each component has one clear purpose
- **Dependency Injection**: Accept configuration rather than hardcoding
- **Error Resilience**: Graceful failure handling
- **Performance Conscious**: Efficient algorithms for real-time use
- **Observable**: Proper logging and metrics

### Import Patterns

```python
# Correct: Import specific components
from core.brain.auto_decider import AutoDecider
from core.voice.emotion_library import get_emotion_library

# Avoid: Wildcard imports
# from core.brain import *

# Avoid: Deep nested imports in user code
# from core.brain.auto_decider.situational_analysis import SituationAnalyzer
```

## 🚀 Deployment

Core components are deployed as part of larger systems:

- **Production**: Embedded in monolithic deployment
- **Services**: Used by individual microservices
- **Clients**: Used by client applications for local processing

No standalone deployment - always used as dependencies.


