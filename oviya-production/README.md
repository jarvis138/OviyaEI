# Oviya Production - Hybrid Architecture with CSM + OpenVoiceV2

Production-ready Oviya implementation using 4-layer hybrid architecture:
**Emotion Detector + Brain (LLM) + Emotion Controller + Hybrid Voice (CSM + OpenVoiceV2)**

## Architecture

```
User Input
    ↓
[1] Emotion Detector
    - Analyzes user's emotional state
    - Provides context to brain
    ↓
[2] Oviya's Brain (Qwen2.5:7B)
    - Understands context and intent
    - Generates response text + emotion label
    ↓
[3] Emotion Controller
    - Maps emotion labels to acoustic parameters
    - Creates style tokens and embeddings
    ↓
[4] Hybrid Voice Engine
    - CSM: Conversational consistency and context awareness
    - OpenVoiceV2: Voice cloning and fine emotion control
    - Auto-selects best engine for each situation
    ↓
Audio Output
```

## Directory Structure

```
oviya-production/
├── emotion_detector/     # User emotion detection
│   ├── detector.py      # Emotion detection from text
│   └── models/          # Emotion detection models
├── brain/                  # LLM brain (Qwen2.5)
│   ├── llm_brain.py       # Brain implementation
│   └── prompts.py         # System prompts and schemas
├── emotion_controller/     # Emotion mapping layer
│   ├── controller.py      # Emotion controller
│   └── mappings.py        # Emotion → acoustic params
├── voice/                  # Hybrid Voice Engine (CSM + OpenVoiceV2)
│   ├── openvoice_tts.py   # Hybrid voice integration
│   └── adapters/          # LoRA adapters for Oviya
├── config/                 # Configuration files
│   ├── emotions.json      # Emotion labels and parameters
│   └── oviya_persona.json # Oviya's personality config
├── utils/                  # Utilities
│   ├── audio_utils.py     # Audio processing
│   └── logging_utils.py   # Logging and monitoring
├── data/                   # Data storage
│   └── voice_samples/     # Oviya voice recordings
├── external/               # External dependencies
│   └── OpenVoice/         # OpenVoiceV2 repository
├── tests/                  # Test suites
│   └── integration_test.py
├── pipeline.py             # Main orchestrator
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up CSM (already running on Vast.ai):**
   ```bash
   # CSM is already running on your Vast.ai server (port 6006)
   # No additional setup needed!
   ```

3. **Set up OpenVoiceV2 (optional):**
   ```bash
   # Clone OpenVoiceV2 repository
   git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice
   cd external/OpenVoice
   pip install -e .
   
   # Download OpenVoiceV2 model
   huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ./models/OpenVoiceV2
   ```

4. **Configure Oviya:**
   ```bash
   cp config/oviya_persona.example.json config/oviya_persona.json
   # Edit config/oviya_persona.json with your settings
   ```

5. **Run Stage 0 Evaluation (Recommended First):**
   ```bash
   python stage0_emotion_test.py
   ```
   This tests if CSM can reproduce emotions from OpenVoice V2 before any training.

6. **Run Oviya:**
   ```bash
   python pipeline.py
   ```

## Features

- ✅ **Emotionally Intelligent:** 8 emotion labels with proper acoustic mapping
- ✅ **Hybrid Voice Engine:** CSM for consistency + OpenVoiceV2 for cloning
- ✅ **User Emotion Detection:** Analyzes user's emotional state
- ✅ **Conversational Context:** CSM maintains conversation flow
- ✅ **Stage 0 Evaluation:** Test emotion transfer before training
- ✅ **Production Ready:** Proper error handling, logging, monitoring
- ✅ **Modular Design:** Each layer can be tested and tuned independently
- ✅ **Open Source:** Built on CSM + OpenVoiceV2, fully customizable

## Emotion Labels

- `calm_supportive`
- `empathetic_sad`
- `joyful_excited`
- `playful`
- `confident`
- `concerned_anxious`
- `angry_firm`
- `neutral`

## Development

Run tests:
```bash
pytest tests/
```

Run integration test:
```bash
python tests/integration_test.py
```

## License

See LICENSE file for details.

