## oviya-production — Architecture

### Purpose
Production-grade, batteries-included runtime for realtime voice: WebRTC and WebSocket servers, `OviyaBrain`, CSM-1B streaming TTS, emotion systems, and test suites.

### Major Modules
- Realtime Servers
  - `voice_server_webrtc.py`: SDP/ICE, inbound mic, VAD, STT → LLM → TTS streaming over outbound track, barge-in handling
  - `websocket_server.py`: WS PCM ingest, streaming STT/LLM, TTS chunks (base64@24k)
- Brain & Psych Systems (`brain/`)
  - `llm_brain.py`: persona-driven LLM with safety, backchannels, prosody, streaming
  - Heuristics: `auto_decider.py`; Safety: `safety_router.py`; memory/consistency/backchannels/etc.
- Voice (`voice/`)
  - `csm_1b_client.py`: local/remote streaming RVQ→Mimi; context & reference conditioning
  - `csm_1b_stream.py`: RVQ windowing & drift correction; `openvoice_tts.py` policy wrapper
- Emotion
  - `emotion_controller/controller.py`: map brain emotion → CSM style; `emotion_detector/` optional inputs
- Config
  - `config/oviya_persona.json`: system prompt, matrices, safety fallbacks; `config/service_urls.py`
- Monitoring
  - Prometheus latency/TTFB metrics in WebRTC server
- Tests
  - `test_*` suites for scenarios, pipeline, VAD, timestamps, memory

### Data Flows
- WebRTC: mic track → preprocess+VAD → end-of-speech → Whisper Turbo/WhisperX → `OviyaBrain` (stream first sentence) → humanlike timing → CSM streaming → outbound audio track; barge-in fade/stop
- WebSocket: PCM chunks → `StreamingSTT` → early sentence → TTS streaming JSON → final response

### Key Entrypoints
- WebRTC turn: `OviyaVoiceConnection.process_audio_frame`, `handle_user_utterance`
- WebRTC signaling: `/api/voice/offer`
- WS turns: `ConversationSession.generate_response`, `generate_response_streaming`
- Brain: `OviyaBrain.think`, `think_streaming`
- CSM: `CSM1BClient.generate_streaming` (local/remote)

### Operational Notes
- Fallbacks: Ollama retry/backoff → simplified → mock; STT Turbo → WhisperX; CSM local/HF/remote
- Performance: early sentence gating; CUDA decode overlap; volume normalize; jitter-smoothing in WS client demo


