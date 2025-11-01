# Code Cleanup & Migration Guide

## ✅ Completed: Unified VAD+STT Implementation

All core voice processing now uses the unified pipeline:
- **New**: `production/voice/unified_vad_stt.py` - Single source of truth
- **Replaces**: Multiple duplicate implementations

## 📋 Files That Can Be Deprecated (After Testing)

### High Priority (Can be removed after full migration)
1. **`production/voice/silero_vad_adapter.py`**
   - Status: Still used as fallback in `websocket_server.py` and `csm_1b_generator_optimized.py`
   - Action: Keep for now, migrate fallbacks to unified pipeline
   - Replacement: `production/voice/unified_vad_stt.py` → `OptimizedSileroVAD`

2. **`production/voice/whisper_client.py`**
   - Status: Still used in `voice_server_webrtc.py`, `csm_1b_generator_optimized.py`, and experimental files
   - Action: Keep for now, migrate to unified pipeline
   - Replacement: `production/voice/unified_vad_stt.py` → `OptimizedWhisperSTT`

### Medium Priority (Service-specific implementations)
3. **`services/services/asr-realtime/silero_vad_processor.py`**
   - Status: Used by ASR microservice
   - Action: Keep if microservice is active, otherwise migrate to unified
   - Note: This was the best implementation, now consolidated into unified

4. **`services/services/asr-realtime/whisper_stream.py`**
   - Status: Used by ASR microservice
   - Action: Keep if microservice is active, otherwise migrate to unified
   - Note: This was the best implementation, now consolidated into unified

## 🚀 Migration Steps

### Step 1: Update Remaining Fallbacks
```python
# Before (in csm_1b_generator_optimized.py):
from .silero_vad_adapter import SileroVADAdapter
from .whisper_client import WhisperTurboClient

# After:
from .unified_vad_stt import UnifiedVADSTTPipeline, get_unified_pipeline
pipeline = get_unified_pipeline()
```

### Step 2: Update voice_server_webrtc.py Fallback
Already completed ✅ - Now uses unified pipeline with legacy fallback

### Step 3: Update websocket_server.py Fallback
Already completed ✅ - Now uses unified pipeline with legacy fallback

### Step 4: Test All Entry Points
- [ ] Test websocket_server.py with unified pipeline
- [ ] Test voice_server_webrtc.py with unified pipeline
- [ ] Test csm_1b_generator_optimized.py (if still used)
- [ ] Verify all fallbacks work correctly

### Step 5: Remove Deprecated Files (After Testing)
Once all tests pass:
```bash
# Backup first
mkdir -p deprecated/voice
mv production/voice/silero_vad_adapter.py deprecated/voice/
mv production/voice/whisper_client.py deprecated/voice/

# Update imports to use unified pipeline
# Then delete if no longer needed
```

## 📝 Notes

- **Services Directory**: Files in `services/services/` are microservice-specific and should remain if those services are active
- **Experimental Files**: Files in `experimental/` should be updated separately
- **Backward Compatibility**: Legacy implementations kept as fallbacks until full migration

## ✅ Completed Migrations

1. ✅ `production/websocket_server.py` - Uses unified pipeline
2. ✅ `production/voice_server_webrtc.py` - Uses unified pipeline
3. ✅ CSM-1B format fixes - Uses `apply_chat_template()`
4. ✅ Mimi decode fixes - Proper format handling
5. ✅ CUDA graphs optimization - Recording phase fixed
6. ✅ Emotion dataset extraction script created

## 🎯 Next Steps

1. Run comprehensive tests with unified pipeline
2. Migrate remaining fallbacks in `csm_1b_generator_optimized.py`
3. Update experimental files if needed
4. Remove deprecated files after full migration

