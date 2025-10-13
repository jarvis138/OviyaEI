# 📁 Oviya Production - Project Structure

## 🎯 Core Production Files

### Main Pipeline
- **`pipeline.py`** - Main orchestrator (Brain + Voice + Emotion Controller)

### Brain System (`brain/`)
- **`llm_brain.py`** - LLM integration with Ollama (qwen2.5:7b)
  - Prosodic markup generation
  - Emotional memory tracking
  - Backchannel injection
- **`backchannels.py`** - Micro-affirmation system (40+ phrases)
- **`epistemic_prosody.py`** - Uncertainty/confidence detection
- **`emotion_transitions.py`** - Smooth emotion blending

### Voice System (`voice/`)
- **`openvoice_tts.py`** - Hybrid Voice Engine (CSM + OpenVoiceV2)
- **`audio_postprocessor.py`** - Maya-level enhancements
  - Breath injection (5 types)
  - Audio mastering (EQ, compression, loudness)
  - Prosody processing (timing, pauses)

### Emotion System (`emotion_controller/`)
- **`controller.py`** - 49-emotion library manager
  - Non-linear intensity curves
  - Contextual modifiers
  - Blended emotion handling

### Emotion Detection (`emotion_detector/`)
- **`detector.py`** - Text-based emotion detection

---

## 🧪 Test Files

### Current Tests (Keep)
- **`test_beyond_maya.py`** - Comprehensive Beyond-Maya feature testing
- **`test_5_scenarios.py`** - Quick 5-scenario validation
- **`test_diverse_scenarios.py`** - 12 scenarios (flirting, sarcasm, prosody)
- **`test_llm_prosody_5.py`** - LLM prosodic markup validation
- **`test_all_enhancements.py`** - All 6 enhancements testing
- **`production_sanity_tests.py`** - Pre-deployment validation

### Integration Tests (`tests/`)
- **`integration_test.py`** - Full pipeline integration test

---

## 📚 Documentation

### Production Documentation (Essential)
- **`PRODUCTION_READINESS.md`** - Deployment checklist & sanity tests
- **`IMPLEMENTATION_STATUS.md`** - Feature completion status
- **`ENHANCEMENTS_COMPLETE.md`** - Beyond-Maya implementation details
- **`README.md`** - Main project overview

### Quick Start Guides
- **`QUICK_START.md`** - Getting started
- **`FIX_CSM_EMOTION_REFERENCES.md`** - CSM emotion reference troubleshooting

### Specialized Guides
- **`COMPLETE_EMOTION_SYSTEM_GUIDE.md`** - 49-emotion system
- **`EXPANDED_EMOTIONS_GUIDE.md`** - Emotion expansion details
- **`VASTAI_DEPLOYMENT.md`** - Vast.ai deployment guide

---

## 🛠️ Utilities

### Monitoring & Optimization
- **`monitoring.py`** - Production metrics tracking
  - MOS (Mean Opinion Score)
  - Emotion accuracy
  - Persona drift
  - Latency monitoring
- **`optimizations.py`** - Runtime optimizations
  - Prosody template caching
  - Emotion parameter caching
  - Streaming synthesizer
  - Adaptive pause system

### Helper Scripts
- **`generate_emotion_references.py`** - Create emotion reference audio
- **`extract_ravdess_emotions.py`** - Extract RAVDESS dataset emotions
- **`cleanup_project.sh`** - Project cleanup script

### Server Scripts (`scripts/`)
- **`vastai_csm_server_expanded.py`** - CSM server with 49 emotions
- **`generate_expanded_emotions_vastai.py`** - Generate emotion refs on server

---

## 📦 Configuration Files

### Emotion Configs (`config/`)
- **`emotions_49.json`** - 49-emotion library with parameters
- **`emotion_reference_mapping.json`** - Emotion-to-reference mappings
- **`emotion_library.json`** - Blended emotion definitions

### Audio Assets (`audio_assets/`)
- **`breath_samples/`** - Breath audio files (5 types)
- **`emotion_references/`** - Emotion reference WAV files (49 emotions)

---

## 📂 Output Directories

- **`output/`** - Generated audio files
  - `scenarios/` - Test scenario outputs
  - `llm_prosody/` - LLM prosody test outputs
  - `diverse_scenarios/` - Diverse scenario outputs

- **`logs/`** - Runtime logs
  - `metrics/` - Production metrics (MOS, latency, etc.)

- **`cache/`** - Cached data
  - `prosody/` - Prosody template cache

---

## 🗄️ Archive

- **`archive/`** - Archived old files
  - `old_tests/` - Deprecated test files
  - `old_docs/` - Outdated documentation
  - `logs/` - Old log files

---

## 🎯 Key Features Implemented

### ✅ Beyond-Maya Level 1 (Complete)
1. **Backchannel System** - 40+ micro-affirmations
2. **Enhanced Emotion Intensity** - Non-linear curves
3. **Contextual Prosody Memory** - Cross-turn tracking
4. **Micro-pause Predictor** - Smart pause injection
5. **Enhanced Breath System** - Respiratory model
6. **49-Emotion Integration** - 3-tier emotion library
7. **Epistemic Prosody** - Uncertainty detection
8. **Emotion Transitions** - Smooth blending

### ✅ Production Systems (Complete)
1. **Monitoring** - MOS, accuracy, drift, latency tracking
2. **Optimizations** - Caching, streaming, adaptive pauses
3. **Sanity Tests** - 5 pre-deployment validation tests
4. **Error Handling** - Fallbacks, mock responses, graceful degradation

---

## 📊 Project Statistics

- **Total Python files**: ~45 (excluding archive)
- **Core production files**: 15
- **Test files**: 7
- **Documentation files**: 20+
- **Emotion library**: 49 emotions (3 tiers)
- **Backchannel phrases**: 40+
- **Breath samples**: 5 types

---

## 🚀 Deployment Status

**Status**: ✅ **PRODUCTION READY (Beta)**

**Sanity Tests**: 4/5 passed (80%)
- ✅ Prosodic markup validation
- ✅ Audio drift detection
- ✅ Emotion distribution
- ✅ Stability & fallbacks
- ⚠️ Performance (latency needs optimization)

**Recommended Actions**:
1. Deploy to beta environment
2. Collect user feedback (MOS scores)
3. Monitor metrics for 1 week
4. Optimize latency (target: < 5s)
5. Iterate based on feedback

---

## 📝 Maintenance

### Regular Tasks
- Monitor logs in `logs/metrics/`
- Review emotion distribution bias
- Check cache hit rates
- Update Ollama tunnel URLs (localhost.run expires)
- Update CSM ngrok URLs

### Cleanup
```bash
./cleanup_project.sh  # Remove temp files, organize archives
```

### Testing
```bash
# Quick validation
python3 test_5_scenarios.py

# Full Beyond-Maya test
python3 test_beyond_maya.py

# Production sanity tests
python3 production_sanity_tests.py

# LLM prosody validation
python3 test_llm_prosody_5.py
```

---

*Last Updated: 2024*  
*Version: 1.0 (Production Beta)*


