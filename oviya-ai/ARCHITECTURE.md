## oviya-ai — Architecture

### Purpose
Service-oriented implementation: standalone microservices for ASR, VAD, CSM streaming, orchestrator, and integrated pipeline; infra for deployment.

### Services
- ASR realtime: `services/asr-realtime/` (Silero VAD + faster-whisper), `server.py`
- VAD service: `services/vad/silero_vad_service.py`
- CSM streaming: `services/csm-streaming/` (model pool, queues), `server.py`
- Orchestrator: `services/orchestrator/server.py` (rate limiting, moderation, performance monitor, streaming WS route)
- Pipeline: `services/pipeline/integrated_pipeline.py` (end-to-end async states)
- Context manager: `services/context/`

### Infra
- `infrastructure/docker/` Dockerfiles per service; `docker-compose.yml`; nginx; monitoring

### Data Flow (Typical)
Client → Orchestrator WS → VAD/ASR → Context → LLM stream → CSM streaming → Client

### Notes
- Complementary to `oviya-production` monolith; enables horizontal scaling and isolation.


