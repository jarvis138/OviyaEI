"""
WebSocket Server for Real-Time Oviya Conversations
Provides streaming audio input/output via WebSocket for web clients
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import asyncio
import numpy as np
import json
import base64
from typing import Dict, Optional, List
import torch
from pathlib import Path
import time
from collections import deque
import os
import sys

# Add parent directory to path to find core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional JWT auth (fallback to allow if PyJWT missing or secret unset)
try:
    import jwt  # PyJWT
    def verify_jwt(token: str) -> bool:
        secret = os.getenv("OVIYA_SECRET", "")
        if not secret:
            return True
        try:
            jwt.decode(token, secret, algorithms=["HS256"])  # type: ignore
            return True
        except Exception:
            return False
except Exception:
    def verify_jwt(token: str) -> bool:
        return True

# Simple token-bucket rate limit per IP
BUCKET: Dict[str, tuple] = {}
CAP = 20
REFILL = CAP
WINDOW = 1.0
def _client_ip(ws: WebSocket) -> str:
    xf = ws.headers.get('x-forwarded-for')
    if xf:
        return xf.split(',')[0].strip()
    return ws.client.host if ws.client else 'unknown'
def allow(ip: str) -> bool:
    t = time.time()
    c, ts = BUCKET.get(ip, (CAP, t))
    c = min(CAP, c + (t - ts) * REFILL / WINDOW)
    if c < 1:
        BUCKET[ip] = (c, t)
        return False
    BUCKET[ip] = (c - 1, t)
    return True

# Metrics
try:
    from prometheus_client import Summary, Histogram, Counter, CONTENT_TYPE_LATEST, generate_latest
    STT_LATENCY = Summary('oviya_stt_latency_seconds', 'STT processing time per turn')
    LLM_LATENCY = Summary('oviya_llm_latency_seconds', 'LLM processing time per turn')
    TTS_LATENCY = Summary('oviya_tts_latency_seconds', 'TTS generation time per turn')
    TIME_TO_FIRST_AUDIO = Summary('oviya_time_to_first_audio_seconds', 'Time to first audio chunk')
    STT_H = Histogram('oviya_stt_seconds', 'STT latency', buckets=(.05,.1,.2,.3,.5,.75,1,2))
    LLM_H = Histogram('oviya_llm_seconds', 'LLM latency', buckets=(.05,.1,.2,.3,.5,.75,1,2))
    TTS_H = Histogram('oviya_tts_seconds', 'TTS latency', buckets=(.05,.1,.2,.3,.5,.75,1,2))
    TTFB_H = Histogram('oviya_ttfb_seconds', 'TTFB', buckets=(.05,.1,.2,.3,.5,.75,1))
    ERRORS = Counter('oviya_ws_errors_total', 'WebSocket errors')
except Exception:
    STT_LATENCY = LLM_LATENCY = TTS_LATENCY = TIME_TO_FIRST_AUDIO = None
    STT_H = LLM_H = TTS_H = TTFB_H = ERRORS = None

# Import Oviya components
# from .voice.realtime_input import RealTimeVoiceInput  # Local WhisperX processing - disabled for now
from .emotion_detector.detector import EmotionDetector
from .brain.llm_brain import OviyaBrain
from .brain.secure_base import SecureBaseSystem
from .brain.bids import BidResponseSystem
from .emotion_controller.controller import EmotionController
from .voice.openvoice_tts import HybridVoiceEngine
from .voice.csm_1b_client import CSM1BClient
from .voice.csm_1b_stream import CSMRVQStreamer, BatchedCSMStreamer, get_batched_streamer
from .voice.csm_1b_generator_optimized import OptimizedCSMStreamer, get_optimized_streamer
from core.voice.acoustic_emotion_detector import AcousticEmotionDetector
from .brain.personality_store import PersonalityStore
from .config.service_urls import OLLAMA_URL, CSM_URL
from faster_whisper import WhisperModel

# Import MCP systems
from .brain.mcp_memory_integration import OviyaMemorySystem
from .brain.crisis_detection import CrisisDetectionSystem
from .brain.empathic_thinking import EmpathicThinkingEngine
from .brain.emotional_reciprocity import reciprocal_empathy_integrator
from .voice.strategic_silence import strategic_silence_manager, emotional_pacing_controller

import time

# Prefer local adapter, fallback to WebRTC implementation
try:
    from .voice.silero_vad_adapter import SileroVAD  # decoupled adapter
    _HAS_SILERO_VAD = True
except Exception:
    try:
        from .voice_server_webrtc import SileroVAD  # fallback
        _HAS_SILERO_VAD = True
    except Exception:
        _HAS_SILERO_VAD = False

# Optional session cleanup helper
try:
    from .voice.session_state import cleanup_sessions
    _HAS_SESSION_CLEANUP = True
except Exception:
    _HAS_SESSION_CLEANUP = False

app = FastAPI(title="Oviya WebSocket Server")
api_v1 = APIRouter(prefix="/v1")
try:
    from core.serving.conditioning_api import router as conditioning_router
    app.include_router(conditioning_router)
except Exception:
    pass
@api_v1.post("/conversations")
async def api_create_conversation(payload: Dict):
    # Minimal stub - generate fake conversation_id
    return {"conversation_id": f"c_{int(time.time()*1000)}"}

@api_v1.post("/conversations/{conversation_id}/turns")
async def api_add_turn(conversation_id: str, payload: Dict):
    user_id = payload.get("user_id", "anonymous")
    text = payload.get("text", "")
    ctx = payload.get("context", {})
    session = ConversationSession(user_id)
    # Use brain to think (non-streaming)
    resp = session.brain.think(text, conversation_history=None)
    # Compose timing plan (reuse humanlike where possible via defaults)
    out = {
        "assistant": {
            "text": resp.get("text", ""),
            "emotion": resp.get("emotion", "neutral"),
            "intensity": resp.get("intensity", 0.7),
            "style_hint": resp.get("style_hint", ""),
            "situation": session.brain._last_guidance_category if hasattr(session.brain, "_last_guidance_category") else "",
            "timing_plan": {"pre_tts_delay_ms": 400}
        },
        "safety": {"flag": False},
        "global_soul": {}
    }
    return out

app.include_router(api_v1)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Pre-initialize models on startup"""
    print("🚀 Pre-initializing models...")
    get_global_models()
    print("✅ Server ready for connections")
    if _HAS_SESSION_CLEANUP:
        async def _cleanup_loop():
            while True:
                try:
                    cleanup_sessions()
                except Exception:
                    pass
                await asyncio.sleep(60)
        asyncio.create_task(_cleanup_loop())

# Global instances (in production, use dependency injection)
personality_store = PersonalityStore()

# Global model instances (shared across sessions for faster startup)
_global_voice_input = None
_global_emotion_detector = None
_global_acoustic_emotion = None

def get_global_models():
    """Initialize and return global model instances"""
    global _global_voice_input, _global_emotion_detector, _global_acoustic_emotion
    
    if _global_voice_input is None:
        print("🎤 Initializing global models (one-time setup)...")
        _global_voice_input = RealTimeVoiceInput(enable_diarization=False)
        _global_voice_input.initialize_models()
        _global_emotion_detector = EmotionDetector()
        _global_acoustic_emotion = AcousticEmotionDetector()
        print("✅ Global models initialized")
    
    return _global_voice_input, _global_emotion_detector, _global_acoustic_emotion


class StreamingSTT:
    """
    Lightweight streaming STT using faster-whisper.
    Processes rolling audio buffer and emits partial transcripts.
    """
    def __init__(self, model_size: str = "small.en"):
        device = "cuda" if torch.cuda.is_available() else "auto"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = 16000
        self.buffer = bytearray()
        self.last_emit_time = 0.0
        self.emit_interval = 0.25  # seconds between partial emissions
        self.last_partial = ""
        self.min_window_s = 1.0
        self.max_window_s = 3.0

    def add_audio(self, audio_bytes: bytes):
        self.buffer.extend(audio_bytes)

    def should_commit(self, vad_confidence: float) -> bool:
        # Commit on strong VAD confidence to avoid mid-phoneme cuts
        return vad_confidence >= 0.7 and len(self.buffer) >= int(self.sample_rate * self.min_window_s) * 2

    def get_partial(self) -> str:
        now = time.time()
        if now - self.last_emit_time < self.emit_interval:
            return ""
        self.last_emit_time = now

        # Need at least min_window_s of audio
        min_bytes = int(self.sample_rate * self.min_window_s) * 2
        if len(self.buffer) < min_bytes:
            return ""

        # Use the last window (up to max_window_s)
        max_bytes = int(self.sample_rate * self.max_window_s) * 2
        window = bytes(self.buffer[-max_bytes:]) if len(self.buffer) > max_bytes else bytes(self.buffer)

        # Convert to float32
        audio_array = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            segments, _ = self.model.transcribe(
                audio_array,
                beam_size=1,
                vad_filter=True,
                without_timestamps=True,
                language="en"
            )
            text = " ".join(s.text.strip() for s in segments if getattr(s, 'text', '').strip())
            text = text.strip()
            if text and text != self.last_partial:
                self.last_partial = text
                return text
            return ""
        except Exception:
            return ""


class ConversationSession:
    """Manages a single conversation session"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id

        # Use global shared models (much faster)
        voice_input, emotion_detector, acoustic_emotion = get_global_models()
        self.voice_input = voice_input
        self.emotion_detector = emotion_detector
        self.acoustic_emotion = acoustic_emotion

        # Create per-session instances
        self.brain = OviyaBrain(ollama_url=OLLAMA_URL)
        self.emotion_controller = EmotionController()
        self.tts = HybridVoiceEngine(csm_url=CSM_URL, default_engine="csm")

        # 🆕 INTEGRATE CUDA GRAPHS OPTIMIZED STREAMING FOR LOW-LATENCY THERAPY
        print("🎯 Initializing CUDA Graphs Optimized CSM Streaming...")
        self.optimized_streamer = get_optimized_streamer()  # CUDA graphs optimized

        # 🔥 PRE-WARM CUDA GRAPHS FOR CONSISTENT <2s PERFORMANCE
        print("🔥 Pre-warming CUDA graphs for therapy sessions...")
        try:
            self.optimized_streamer.warmup_for_therapy()
            print("✅ CUDA graphs pre-warmed for <2s consistent performance!")
        except Exception as e:
            print(f"⚠️ CUDA graphs warmup failed: {e}")
            print("   Continuing with standard initialization...")

        print("✅ CUDA graphs optimized streamer ready (<2s latency target)")

        # 🆕 INTEGRATE BATCHED RVQ STREAMING FOR MULTI-USER CONCURRENT AUDIO
        print("🎤 Initializing Batched CSM Streaming for multi-user conversations...")
        self.csm_streaming = get_batched_streamer()  # Global batched streamer

        # Start batch processor in background
        self.batch_processor_task = asyncio.create_task(
            self.csm_streaming.start_batch_processor()
        )
        print("✅ Batched streaming processor started")
        self.stt = StreamingSTT()
        self.is_generating = False

        # Psych systems
        self.secure_base = SecureBaseSystem()
        self.bids = BidResponseSystem()

        # Initialize MCP systems for emotional intelligence
        self.memory_system = OviyaMemorySystem()
        self.crisis_detector = CrisisDetectionSystem()
        self.empathy_engine = EmpathicThinkingEngine()

        # Load user personality
        self.personality = personality_store.load_personality(user_id)
        if self.personality:
            print(f"📚 Loaded personality for {user_id}")
            # Inject personality context into brain
            context = personality_store.get_conversation_summary(user_id, last_n=5)
            self.brain.context = context

        print(f"✅ Conversation session initialized for {user_id} with MCP emotional intelligence")

        # Streaming state
        self._tts_stream_task: Optional[asyncio.Task] = None
        self._tts_cancel_requested: bool = False
        self._reference_audio: Optional[np.ndarray] = None
        self._reference_text: Optional[str] = None
        self._memory_triples: List[Dict] = []  # (q, p, r) triples
        # VAD state for commit (energy-based)
        self._vad_is_speaking: bool = False
        self._vad_silence_ms: float = 0.0
        self._vad_speech_ms: float = 0.0
        # Breath sample (optional)
        self._breath_buf: Optional[np.ndarray] = self._load_breath_sample()
        self._breath_sent: bool = False
        # Silero VAD (if available)
        self.silero_vad = SileroVAD() if _HAS_SILERO_VAD else None
        self._silero_remainder = np.zeros((0,), dtype=np.float32)
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[Dict]:
        """
        Process incoming audio chunk
        
        Args:
            audio_data: Raw audio bytes (PCM, 16-bit, 16kHz, mono)
            
        Returns:
            Transcription result if available
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to legacy voice input buffer (for fallback)
        self.voice_input.add_audio_chunk(audio_array)
        # Also feed streaming STT
        self.stt.add_audio(audio_data)
        
        # Check for transcription
        result = self.voice_input.get_transcription(timeout=0.01)
        
        return result

    def _load_breath_sample(self) -> Optional[np.ndarray]:
        try:
            import torchaudio
            p = Path(__file__).resolve().parent.parent / 'audio_assets' / 'breath_samples' / 'gentle_exhale.wav'
            if not p.exists():
                return None
            wav, sr = torchaudio.load(str(p))
            wav = wav.mean(dim=0)  # mono
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)
            arr = wav.numpy().astype(np.float32)
            arr *= 0.15  # low amplitude
            return arr
        except Exception:
            return None

    async def _compute_personality_vector(self, user_text: str, user_emotion: str) -> Dict[str, float]:
        """
        Compute the 5-pillar personality vector for Oviya's response
        This is a simplified version - would integrate with full personality system
        """
        # Base personality vector (could be loaded from personality store)
        base_vector = {
            "Ma": 0.3,      # Innovation/Creativity
            "Ahimsa": 0.4,  # Non-violence/Safety
            "Jeong": 0.15,  # Deep connection/Empathy
            "Logos": 0.1,   # Reason/Logic
            "Lagom": 0.05   # Balance/Appropriateness
        }

        # Adjust based on user emotion (simplified adaptation)
        emotion_adjustments = {
            "sad": {"Jeong": +0.1, "Ahimsa": +0.1, "Lagom": +0.05},
            "anxious": {"Ahimsa": +0.15, "Lagom": +0.1, "Logos": +0.05},
            "angry": {"Ahimsa": +0.2, "Jeong": +0.1, "Lagom": +0.05},
            "joyful": {"Ma": +0.1, "Jeong": +0.1, "Lagom": +0.05},
            "confused": {"Logos": +0.15, "Jeong": +0.1, "Ahimsa": +0.05}
        }

        if user_emotion in emotion_adjustments:
            for pillar, adjustment in emotion_adjustments[user_emotion].items():
                base_vector[pillar] = min(1.0, max(0.0, base_vector[pillar] + adjustment))

        # Normalize to ensure sum <= 1 (representing energy distribution)
        total = sum(base_vector.values())
        if total > 1.0:
            for pillar in base_vector:
                base_vector[pillar] /= total

        return base_vector
    
    async def generate_response(self, user_text: str, user_emotion: str) -> Dict:
        """
        Generate Oviya's response
        
        Args:
            user_text: Transcribed user text
            user_emotion: Detected user emotion
            
        Returns:
            {
                'text': str,
                'emotion': str,
                'audio_chunks': List[bytes]  # Base64 encoded
            }
        """
        print(f"🧠 Generating response for: '{user_text}' (emotion: {user_emotion})")
        
        # Generate brain response
        response = self.brain.think(user_text, user_emotion)
        print(f"💬 Brain response: '{response['text']}' (emotion: {response['emotion']})")
        
        # Map to CSM emotion
        oviya_emotion_params = self.emotion_controller.map_emotion(
            response['emotion'],
            response['intensity']
        )
        oviya_emotion = oviya_emotion_params['style_token']
        print(f"🎭 Mapped emotion: {oviya_emotion}")
        
        # Generate audio
        print(f"🎤 Generating TTS audio...")
        audio_tensor = self.tts.generate(
            text=response['text'],
            emotion_params=oviya_emotion_params
        )
        # Get audio shape safely
        if hasattr(audio_tensor, 'shape'):
            audio_shape = tuple(audio_tensor.shape)
        elif hasattr(audio_tensor, '__len__'):
            audio_shape = f"len={len(audio_tensor)}"
        else:
            audio_shape = 'unknown'
        print(f"🎵 TTS generated: shape={audio_shape}")
        
        # Convert audio to chunks for streaming
        audio_chunks = self._chunk_audio(audio_tensor)
        print(f"📦 Created {len(audio_chunks)} audio chunks")
        
        # Save conversation turn
        personality_store.add_conversation_turn(self.user_id, {
            'user_message': user_text,
            'oviya_response': response['text'],
            'user_emotion': user_emotion,
            'oviya_emotion': oviya_emotion
        })
        
        # Calculate duration (samples / sample_rate)
        duration = len(audio_tensor) / 24000.0  # CSM outputs at 24kHz
        
        return {
            'text': response['text'],
            'emotion': oviya_emotion,
            'audio_chunks': audio_chunks,
            'duration': duration
        }

    async def generate_response_streaming(self, websocket: WebSocket, user_text: str, user_emotion: str):
        """
        Stream TTS audio chunks as they are generated with MCP emotional intelligence.
        Sends 'audio_chunk' messages and a terminal 'response' with text/emotion.
        """
        await self.cancel_tts_stream()

        # MCP INTEGRATION: Crisis detection first (SAFETY FIRST)
        crisis_assessment = await self.crisis_detector.assess_crisis_risk(
            user_text, [msg.get('text', '') for msg in self._memory_triples[-10:]]
        )

        if crisis_assessment["escalation_needed"]:
            # Handle crisis with immediate intervention
            emergency_resources = await self.crisis_detector.get_emergency_resources()
            crisis_response = crisis_assessment["immediate_response"]

            # Send crisis intervention immediately
            await websocket.send_json({
                'type': 'crisis_intervention',
                'text': crisis_response,
                'emergency_resources': emergency_resources,
                'severity': crisis_assessment["risk_level"]
            })
            return

        # MCP INTEGRATION: Retrieve relevant memories for context
        relevant_memories = await self.memory_system.retrieve_relevant_memories(
            self.user_id, user_text, limit=5
        )

        # MCP INTEGRATION: Compute current personality vector
        personality_vector = await self._compute_personality_vector(user_text, user_emotion)

        # MCP INTEGRATION: Generate deep empathic response
        emotion_context = {
            "emotion": user_emotion,
            "intensity": 0.7,  # Could be enhanced with acoustic analysis
            "patterns": relevant_memories.get("conversation_history", []),
            "conflicts": []  # Could be detected from conversation flow
        }

        empathic_response = await self.empathy_engine.generate_empathic_response(
            user_text, personality_vector, emotion_context
        )

        # 🆕 STRATEGIC SILENCE INTEGRATION: Apply therapeutic silence before response
        ma_weight = personality_vector.get("Ma", 0.3)

        # Build conversation context for adaptive silence
        conversation_context = {
            "depth": len(self._memory_triples),
            "emotion_intensity": emotion_context.get("intensity", 0.7),
            "vulnerability_indicators": len([msg for msg in self._memory_triples[-5:]
                                          if any(word in msg.get('text', '').lower()
                                                for word in ['scared', 'lost', 'hurt', 'ashamed', 'broken'])]),
            "recent_silence_patterns": []  # Could track from previous interactions
        }

        # Apply pre-response therapeutic silence
        silence_metadata = await strategic_silence_manager.apply_therapeutic_silence(
            user_emotion=user_emotion,
            emotion_intensity=emotion_context.get("intensity", 0.7),
            ma_weight=ma_weight,
            websocket=websocket,
            silence_type="pre_response",
            conversation_context=conversation_context
        )

        # 🆕 EMOTIONAL RECIPROCITY INTEGRATION: Add Oviya's internal emotional state
        # Create emotion embedding (simplified - would use real emotion detector)
        emotion_embed = torch.randn(64)  # Placeholder - should come from emotion detector
        oviya_personality_tensor = torch.tensor(list(personality_vector.values()))

        # Enhance response with reciprocal empathy
        enhanced_response_text, reciprocity_metadata = await reciprocal_empathy_integrator.enhance_response_with_reciprocity(
            response_text=empathic_response["response"],
            user_emotion_embed=emotion_embed,
            oviya_personality=oviya_personality_tensor,
            conversation_context={
                "emotion": user_emotion,
                "depth": len(self._memory_triples),
                "emotion_intensity": emotion_context.get("intensity", 0.7)
            }
        )

        # 🆕 THERAPEUTIC PACING INTEGRATION: Add pause markers and voice modulation
        # Apply therapeutic pacing to the enhanced response
        response_with_pauses = emotional_pacing_controller.add_therapeutic_pauses(
            enhanced_response_text,
            ma_weight=ma_weight,
            emotion=user_emotion,
            emotion_intensity=emotion_context.get("intensity", 0.7)
        )

        # Get Ma-weighted voice modulation parameters
        voice_modulation = emotional_pacing_controller.apply_ma_voice_modulation(
            ma_weight=ma_weight,
            emotion=user_emotion
        )

        # Apply to prosody controller (integrate with existing prosody system)
        prosodic_text = response_with_pauses  # Start with pause-enhanced text

        # TODO: Integrate voice_modulation with CSM-1B prosody controller
        emotion = user_emotion or "neutral"

        # Stream LLM tokens while starting TTS ASAP
        start_time = time.time()
        assembled = []
        token_count = 0

        # For MCP-enhanced responses, we generate the full text first
        # (could be optimized for streaming later)
        if not prosodic_text:
            # Fallback to basic brain response if MCP fails
            async for token in self.brain.think_streaming(user_text, user_emotion, conversation_history=None):
                assembled.append(token)
                token_count += 1
                # Early trigger logic
                if not prosodic_text and any(p in ''.join(assembled) for p in ['.', '!', '?']):
                    prosodic_text = ''.join(assembled)
                    break
                if not prosodic_text and token_count >= 20:
                    prosodic_text = ''.join(assembled)
                    break

        if not prosodic_text:
            # Final fallback
            brain_resp_full = self.brain.think(user_text, user_emotion)
            prosodic_text = brain_resp_full.get('prosodic_text') or brain_resp_full.get('text', '')
            emotion = brain_resp_full.get('emotion', emotion)

        intensity = 0.7  # Could be derived from personality vector
        oviya_emotion_params = self.emotion_controller.map_emotion(
            emotion, intensity
        )

        # MCP INTEGRATION: Store conversation in memory system
        await self.memory_system.store_conversation_memory(self.user_id, {
            "user_input": user_text,
            "response": prosodic_text,
            "timestamp": time.time(),
            "emotion": emotion,
            "personality_vector": personality_vector,
            "session_id": f"session_{int(time.time())}"
        })

        # Store memory triple
        self._add_memory_triple(user_text, prosodic_text)

        self._tts_cancel_requested = False

        async def _stream():
            try:
                max_samples = 1920  # ~80ms at 24kHz
                buf, n = [], 0
                ttfb_sent = False
                        # 🆕 BATCHED RVQ STREAMING: Multi-user concurrent audio generation
                # Submit request to batch processor for optimal GPU utilization
                batch_id = await self.csm_streaming.submit_stream_request(
                    user_id=self.user_id,
                    text=prosodic_text,
                    emotion=oviya_emotion_params.get('style_token', emotion),
                    speaker_id=42,  # Oviya's consistent voice
                    conversation_context=self._format_context_for_tts(),
                    priority=1  # Normal priority (could be higher for crisis situations)
                )

                print(f"📦 Submitted to batch processor: {batch_id}")

                # Stream results from batched processor
                async for c24 in self.csm_streaming.get_stream_results(self.user_id):
                    if self._tts_cancel_requested:
                        break
                    # Smooth chunk edge via simple EMA filter
                    c24 = self._ema_smooth(c24, alpha=0.1)
                    buf.append(c24)
                    n += len(c24)
                    if n >= max_samples:
                        arr = np.concatenate(buf)
                        # Insert breath once at start if requested
                        if not ttfb_sent and not self._breath_sent and self._breath_buf is not None and ('<breath' in prosodic_text):
                            arr = np.concatenate([self._breath_buf, arr])
                            self._breath_sent = True
                        await websocket.send_json({
                            'type': 'audio_chunk',
                            'format': 'pcm_s16le',
                            'sample_rate': 24000,
                            'audio_base64': base64.b64encode((arr * 32767).astype(np.int16).tobytes()).decode('utf-8')
                        })
                        if not ttfb_sent:
                            try:
                                await websocket.send_json({'type': 'first_audio_chunk'})
                            except Exception:
                                pass
                            if TIME_TO_FIRST_AUDIO:
                                TIME_TO_FIRST_AUDIO.observe(time.time() - start_time)
                            ttfb_sent = True
                        buf, n = [], 0
                if not self._tts_cancel_requested and buf:
                    arr = np.concatenate(buf)
                    await websocket.send_json({
                        'type': 'audio_chunk',
                        'format': 'pcm_s16le',
                        'sample_rate': 24000,
                        'audio_base64': base64.b64encode((arr * 32767).astype(np.int16).tobytes()).decode('utf-8')
                    })
            except Exception as e:
                await websocket.send_json({'type': 'error', 'message': f'TTS stream error: {str(e)}'})

        self._tts_stream_task = asyncio.create_task(_stream())

        await websocket.send_json({
            'type': 'response',
            'text': prosodic_text,
            'emotion': oviya_emotion_params.get('style_token', emotion)
        })

        # 🆕 POST-RESPONSE THERAPEUTIC SILENCE: Apply contemplative silence after response
        # This creates the "sitting with you" feeling that makes Oviya truly present
        post_silence_metadata = await strategic_silence_manager.apply_therapeutic_silence(
            user_emotion=user_emotion,
            emotion_intensity=emotion_context.get("intensity", 0.7),
            ma_weight=ma_weight,
            websocket=websocket,
            silence_type="post_response",
            conversation_context=conversation_context
        )

        # Log therapeutic silence events for analytics and improvement
        silence_analytics = {
            "pre_response_silence": silence_metadata,
            "post_response_silence": post_silence_metadata,
            "total_silence_duration": silence_metadata["final_duration"] + post_silence_metadata["final_duration"],
            "ma_weight_used": ma_weight,
            "emotion": user_emotion,
            "conversation_depth": len(self._memory_triples),
            "timestamp": time.time()
        }
        # TODO: Send to analytics/logging system

    async def cancel_tts_stream(self):
        self._tts_cancel_requested = True

        # Cancel batched streaming request
        try:
            await self.csm_streaming.cancel_user_stream(self.user_id)
        except Exception:
            pass  # Batched streamer handles cleanup gracefully

        if self._tts_stream_task and not self._tts_stream_task.done():
            try:
                await asyncio.wait_for(self._tts_stream_task, timeout=0.1)
            except asyncio.TimeoutError:
                self._tts_stream_task.cancel()
        self._tts_stream_task = None

    def _add_memory_triple(self, q: str, r: str):
        p = (self.brain.context[:200] if hasattr(self.brain, 'context') and self.brain.context else "")
        self._memory_triples.append({'q': q, 'p': p, 'r': r})
        if len(self._memory_triples) > 50:
            self._memory_triples = self._memory_triples[-50:]

    def _format_context_for_tts(self) -> List[Dict]:
        ctx = []
        for t in self._memory_triples[-3:]:
            ctx.append({'text': t['q'], 'speaker_id': 1})
            ctx.append({'text': t['r'], 'speaker_id': 0})
        return ctx
    
    def _ema_smooth(self, x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        try:
            y = np.copy(x)
            for i in range(1, len(y)):
                y[i] = alpha * y[i] + (1 - alpha) * y[i - 1]
            return y
        except Exception:
            return x
    
    def _chunk_audio(self, audio: torch.Tensor, chunk_size: int = 4096) -> list:
        """
        Split audio into chunks for streaming
        
        Args:
            audio: Audio tensor (1D or 2D)
            chunk_size: Samples per chunk
            
        Returns:
            List of base64-encoded audio chunks
        """
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        # Convert to int16 PCM
        audio_np = (audio.cpu().numpy() * 32767).astype(np.int16)
        
        chunks = []
        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i:i+chunk_size]
            # Encode as base64 for JSON transmission
            chunk_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
            chunks.append(chunk_b64)
        
        return chunks


@app.get("/")
async def get():
    """Serve a simple test page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Oviya Voice Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            #status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background-color: #d4edda; color: #155724; }
            .disconnected { background-color: #f8d7da; color: #721c24; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
            #transcript { border: 1px solid #ccc; padding: 10px; min-height: 200px; margin: 10px 0; }
            .user { color: blue; }
            .oviya { color: green; }
        </style>
    </head>
    <body>
        <h1>🎤 Oviya Voice Chat</h1>
        <div id="status" class="disconnected">Disconnected</div>
        <button id="connectBtn">Connect</button>
        <button id="recordBtn" disabled>Start Recording</button>
        <div id="transcript"></div>
        
        <script>
            // JitterPlayer for adaptive playback
            class JitterPlayer {
                constructor(ctx, dst, baseRate = 1.0) {
                    this.ctx = ctx; this.dst = dst; this.queue = []; this.playing = false;
                    this.targetMs = 300; this.baseRate = baseRate; this.minMs = 150; this.maxMs = 500;
                }
                bufferMs() { return this.queue.reduce((a,b)=>a + (b.length/24000*1000), 0); }
                enqueueFloat32(bufFloat) {
                    const b = this.ctx.createBuffer(1, bufFloat.length, 24000);
                    b.getChannelData(0).set(bufFloat); this.queue.push(b);
                    if (!this.playing) { this.playing = true; this._pump(); }
                }
                _pump() {
                    if (!this.queue.length) { this.playing = false; return; }
                    const buf = this.queue.shift(); const src = this.ctx.createBufferSource(); src.buffer = buf;
                    let rate = this.baseRate; const ms = this.bufferMs();
                    if (ms < this.targetMs) rate *= 1.005; if (ms > this.targetMs) rate *= 0.995;
                    src.playbackRate.value = Math.max(0.98, Math.min(1.02, rate));
                    src.connect(this.dst); src.onended = () => this._pump(); src.start();
                }
            }
            let ws = null;
            let mediaRecorder = null;
            let audioContext = null;
            let isRecording = false;
            let ttsAudioCtx = null;
            let ttsGain = null;
            let jitter = null;
            let userSpeaking = false;
            
            document.getElementById('connectBtn').onclick = connect;
            document.getElementById('recordBtn').onclick = toggleRecording;
            
            function connect() {
                const token = localStorage.getItem('oviya_token') || '';
                ws = new WebSocket(`ws://localhost:8000/ws/conversation?user_id=test_user&token=${encodeURIComponent(token)}`);
                
                ws.onopen = () => {
                    document.getElementById('status').textContent = 'Connected';
                    document.getElementById('status').className = 'connected';
                    document.getElementById('recordBtn').disabled = false;
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = () => {
                    document.getElementById('status').textContent = 'Disconnected';
                    document.getElementById('status').className = 'disconnected';
                    document.getElementById('recordBtn').disabled = true;
                };
            }
            
            async function toggleRecording() {
                if (!isRecording) {
                    await startRecording();
                } else {
                    stopRecording();
                }
            }
            
            async function startRecording() {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new AudioContext({ sampleRate: 24000 });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(2048, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    const audioData = e.inputBuffer.getChannelData(0);
                    // Downsample 24k -> 16k for server compatibility
                    const ratio = 24000 / 16000;
                    const outLen = Math.floor(audioData.length / ratio);
                    const out = new Int16Array(outLen);
                    for (let i = 0; i < outLen; i++) {
                        const v = audioData[Math.floor(i * ratio)];
                        out[i] = Math.max(-32768, Math.min(32767, v * 32768));
                    }
                    ws.send(out.buffer);
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                document.getElementById('recordBtn').textContent = 'Stop Recording';
            }
            
            function stopRecording() {
                if (audioContext) {
                    audioContext.close();
                }
                isRecording = false;
                document.getElementById('recordBtn').textContent = 'Start Recording';
            }
            
            function handleMessage(data) {
                const transcript = document.getElementById('transcript');
                
                if (data.type === 'transcription') {
                    transcript.innerHTML += `<p class="user"><strong>You:</strong> ${data.text}</p>`;
                } else if (data.type === 'response') {
                    transcript.innerHTML += `<p class="oviya"><strong>Oviya:</strong> ${data.text}</p>`;
                } else if (data.type === 'audio_chunk') {
                    // Streaming audio chunk playback
                    enqueueAndPlayChunk(data);
                }
                
                transcript.scrollTop = transcript.scrollHeight;
            }
            
            function initTTSPlayback() {
                if (!ttsAudioCtx) {
                    ttsAudioCtx = new AudioContext({ sampleRate: 24000 });
                    ttsGain = ttsAudioCtx.createGain();
                    ttsGain.gain.setValueAtTime(1.0, ttsAudioCtx.currentTime);
                    ttsGain.connect(ttsAudioCtx.destination);
                    jitter = new JitterPlayer(ttsAudioCtx, ttsGain);
                }
            }

            function enqueueAndPlayChunk(data) {
                initTTSPlayback();
                const bytes = Uint8Array.from(atob(data.audio_base64), c => c.charCodeAt(0));
                const int16 = new Int16Array(bytes.buffer);
                const f32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768.0;
                jitter.enqueueFloat32(f32);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.websocket("/ws/conversation")
async def websocket_endpoint(websocket: WebSocket, user_id: str = "anonymous"):
    """
    WebSocket endpoint for real-time conversation
    
    Protocol:
        Client -> Server: 
            - Raw audio bytes (PCM, 16-bit, 16kHz, mono)
            - JSON messages: {'type': 'greeting', 'text': str}
        Server -> Client: JSON messages
            - {'type': 'transcription', 'text': str, 'emotion': str}
            - {'type': 'response', 'text': str, 'emotion': str}
            - {'type': 'audio_chunk', 'format': 'pcm_s16le', 'sample_rate': 24000, 'audio_base64': str}
            - {'type': 'error', 'message': str}
    """
    # JWT + rate limit
    token = websocket.query_params.get('token', '')
    if not verify_jwt(token):
        await websocket.close(code=4401, reason="unauthorized")
        return
    ip = _client_ip(websocket)
    if not allow(ip):
        await websocket.close(code=4408, reason="rate_limited")
        return
    await websocket.accept()
    
    print(f"🔌 WebSocket connected: {user_id}")
    
    # Create conversation session
    session = ConversationSession(user_id)
    
    try:
        # Heartbeat task to keep tunnel/NAT alive
        async def heartbeat():
            while True:
                try:
                    await websocket.send_json({"type": "ping", "t": time.time()})
                except Exception:
                    break
                await asyncio.sleep(5)
        hb_task = asyncio.create_task(heartbeat())
        while True:
            # Receive data (can be bytes or text)
            message = await websocket.receive()
            
            # Check message type
            if 'text' in message:
                # Handle JSON messages (greeting, etc.)
                try:
                    data = json.loads(message['text'])
                    # Interrupt control path
                    if data.get('type') == 'interrupt':
                        try:
                            session._tts_cancel_requested = True
                            await session.cancel_tts_stream()
                            await websocket.send_json({'type': 'interrupt_ack', 'id': data.get('id')})
                        except Exception as e:
                            await websocket.send_json({'type': 'error', 'message': f'interrupt_failed: {str(e)}'})
                        continue
                    if data.get('type') == 'upload_reference_voice':
                        # Expect base64 PCM16 mono 24k or WAV bytes
                        try:
                            ref_b64 = data.get('audio_base64')
                            transcript = data.get('transcript')
                            if ref_b64:
                                ref_bytes = base64.b64decode(ref_b64)
                                # Set on session CSM client
                                session.csm_streaming.set_reference_voice(ref_bytes, transcript)
                                await websocket.send_json({'type': 'reference_voice_set', 'ok': True})
                            else:
                                await websocket.send_json({'type': 'reference_voice_set', 'ok': False, 'error': 'missing audio_base64'})
                        except Exception as e:
                            await websocket.send_json({'type': 'reference_voice_set', 'ok': False, 'error': str(e)})
                        continue
                    if data.get('type') == 'test_cuda_graphs':
                        # 🎯 CUDA GRAPHS PERFORMANCE TEST ENDPOINT
                        print(f"🧪 CUDA Graphs Performance Test requested by {user_id}")
                        try:
                            test_text = data.get('text', 'Hello! This is a CUDA graphs optimized generation test.')
                            emotion = data.get('emotion', 'joyful')

                            # Measure latency with CUDA graphs
                            start_time = time.time()
                            audio_bytes = session.optimized_streamer.generate_voice(
                                text=test_text,
                                speaker_id=42,  # Oviya's voice
                                emotion=emotion
                            )
                            latency_ms = (time.time() - start_time) * 1000

                            # Send performance results
                            import base64
                            await websocket.send_json({
                                'type': 'cuda_graphs_test_result',
                                'text': test_text,
                                'emotion': emotion,
                                'latency_ms': round(latency_ms, 1),
                                'audio_size_bytes': len(audio_bytes),
                                'audio_base64': base64.b64encode(audio_bytes).decode('utf-8'),
                                'sample_rate': 24000,
                                'target_achieved': latency_ms < 100
                            })

                            print(f"✅ CUDA graphs test: {latency_ms:.1f}ms latency")
                        except Exception as e:
                            await websocket.send_json({
                                'type': 'error',
                                'message': f'CUDA graphs test failed: {str(e)}'
                            })

                    if data.get('type') == 'batch_voice_generation':
                        # 🎵 BATCH PROCESSING FOR MULTI-USER SESSIONS
                        print(f"🎵 Batch voice generation request from {user_id}")
                        try:
                            batch_requests = data.get('requests', [])
                            if not batch_requests:
                                await websocket.send_json({
                                    'type': 'error',
                                    'message': 'No batch requests provided'
                                })
                                continue

                            print(f"🎵 Processing batch of {len(batch_requests)} requests...")

                            # Use optimized streamer for batch processing
                            batch_results = session.optimized_streamer.generate_batch_voice(batch_requests)

                            # Send results back
                            import base64
                            results_data = []
                            for i, audio_bytes in enumerate(batch_results):
                                results_data.append({
                                    'request_index': i,
                                    'audio_base64': base64.b64encode(audio_bytes).decode('utf-8'),
                                    'sample_rate': 24000,
                                    'audio_size_bytes': len(audio_bytes)
                                })

                            await websocket.send_json({
                                'type': 'batch_voice_results',
                                'results': results_data,
                                'total_requests': len(batch_requests),
                                'processed_requests': len(batch_results)
                            })

                            print(f"✅ Batch processing complete: {len(batch_results)}/{len(batch_requests)} requests")

                        except Exception as e:
                            await websocket.send_json({
                                'type': 'error',
                                'message': f'Batch voice generation failed: {str(e)}'
                            })

                    if data.get('type') == 'greeting':
                        print(f"👋 Greeting request from {user_id}")
                        try:
                            # Generate greeting response - simulate user saying "Hello"
                            greeting_response = await session.generate_response(
                                "Hello",  # User's greeting
                                "neutral"  # User emotion
                            )
                            
                            print(f"✅ Greeting generated: {len(greeting_response.get('audio_chunks', []))} chunks")
                            print(f"   Text: {greeting_response.get('text')}")
                            print(f"   Emotion: {greeting_response.get('emotion')}")
                            
                            # Send greeting to client
                            await websocket.send_json({
                                'type': 'response',
                                'text': greeting_response['text'],
                                'emotion': greeting_response['emotion'],
                                'audio_chunks': greeting_response['audio_chunks'],
                                'duration': greeting_response['duration']
                            })
                            print(f"✅ Greeting sent to client")
                        except Exception as e:
                            print(f"❌ Error generating greeting: {e}")
                            import traceback
                            traceback.print_exc()
                            await websocket.send_json({
                                'type': 'error',
                                'message': str(e)
                            })
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON received: {e}")
                continue
            
            elif 'bytes' in message:
                # Handle audio data
                audio_data = message['bytes']
                print(f"🎤 Received audio: {len(audio_data)} bytes")
                
                # Process audio chunk
                result = await session.process_audio_chunk(audio_data)
                
                # Emit partial transcripts from streaming STT
                partial = session.stt.get_partial()
                if partial:
                    await websocket.send_json({
                        'type': 'transcription',
                        'text': partial,
                        'partial': True
                    })
                    # If sentence boundary or token-bucket threshold, trigger early TTS
                    if any(p in partial for p in ['.', '!', '?']) and not session.is_generating:
                        # Bid detection for micro-ack (backchannel is injected in brain already)
                        bid = session.bids.detect_bid(partial, prosody={"energy": 0.05}, pause_ms=300)
                        # We could send a quick ack here if needed
                        session.is_generating = True
                        asyncio.create_task(session.generate_response_streaming(
                            websocket,
                            partial,
                            'neutral'
                        ))
                
                # Prefer Silero VAD if available to decide end-of-speech
                if session.silero_vad is not None:
                    # Convert to float32 16k and process in 512-sample windows
                    pcm16 = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    buf = np.concatenate([session._silero_remainder, pcm16]) if len(session._silero_remainder) else pcm16
                    idx = 0
                    win = 512
                    eos_detected = False
                    eos_audio = None
                    while idx + win <= len(buf):
                        chunk = buf[idx:idx+win]
                        is_speech, end_of_speech, audio_to_process = session.silero_vad.process_chunk(chunk)
                        if end_of_speech and audio_to_process is not None and len(audio_to_process) > 0:
                            eos_detected = True
                            eos_audio = audio_to_process
                        idx += win
                    # Store remainder
                    session._silero_remainder = buf[idx:]
                    if eos_detected and not session.is_generating:
                        session.is_generating = True
                        # Transcribe eos_audio quickly with faster-whisper
                        try:
                            segments, _ = session.stt.model.transcribe(
                                eos_audio.astype(np.float32),
                                beam_size=1,
                                vad_filter=False,
                                without_timestamps=True,
                                language="en"
                            )
                            text = " ".join(s.text.strip() for s in segments if getattr(s, 'text', '').strip()).strip()
                        except Exception:
                            text = partial or (result['text'] if result else '')
                        asyncio.create_task(session.generate_response_streaming(
                            websocket,
                            text,
                            'neutral'
                        ))
                else:
                    # Energy-based VAD commit: if speaking→silence transition, kick generation
                    samples = int(len(audio_data) / 2)
                    dur_ms = samples / 16.0  # since 16kHz
                    pcm16 = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(pcm16 * pcm16) + 1e-8))
                    speaking_now = rms > 0.02
                    if speaking_now:
                        session._vad_speech_ms += dur_ms
                        session._vad_silence_ms = 0.0
                    else:
                        session._vad_silence_ms += dur_ms
                    if not session._vad_is_speaking and speaking_now:
                        session._vad_is_speaking = True
                    if session._vad_is_speaking and not speaking_now and session._vad_speech_ms >= 300 and session._vad_silence_ms >= 250:
                        session._vad_is_speaking = False
                        session._vad_speech_ms = 0.0
                        # Trigger early generation if not already
                        if not session.is_generating:
                            session.is_generating = True
                            asyncio.create_task(session.generate_response_streaming(
                                websocket,
                                partial or (result['text'] if result else ''),
                                'neutral'
                            ))

                if not result:
                    # No full transcription yet
                    continue
                
                print(f"📝 Got transcription: '{result['text'][:50]}...'")
                    
                # Send transcription to client
                await websocket.send_json({
                    'type': 'transcription',
                    'text': result['text'],
                    'speakers': result.get('speakers', ['user']),
                    'word_timestamps': result.get('word_timestamps', [])
                })
                
                # Detect emotion from audio + text
                acoustic_emotion = session.acoustic_emotion.detect_emotion(
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                )
                
                text_emotion = session.emotion_detector.detect_emotion(result['text'])
                
                # Combine emotions
                combined = session.acoustic_emotion.combine_with_text_emotion(
                    acoustic_emotion,
                    text_emotion,
                    acoustic_weight=0.6
                )
                
                # If not already generating from partials, start now
                if not session.is_generating:
                    session.is_generating = True
                    await session.generate_response_streaming(
                        websocket,
                        result['text'],
                        combined['emotion']
                    )
                    session.is_generating = False
    
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {user_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
        await websocket.close()
    finally:
        try:
            hb_task.cancel()
        except Exception:
            pass

        # Cleanup batched streaming resources
        try:
            await session.cancel_tts_stream()
            # Note: Batched streamer handles global cleanup automatically
        except Exception:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Oviya WebSocket Server",
        "version": "1.0.0"
    }


@app.get('/worklet_ws.js')
async def worklet_ws():
    js = """
class WSRecorder extends AudioWorkletProcessor {
  constructor() { super(); this.decim = 24000/16000; }
  process(inputs) {
    const chan = inputs[0]?.[0]; if (!chan) return true;
    const outLen = Math.floor(chan.length / this.decim);
    const pcm16 = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const v = chan[Math.floor(i * this.decim)];
      pcm16[i] = Math.max(-32768, Math.min(32767, v * 32768));
    }
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    return true;
  }
}
registerProcessor('ws-recorder', WSRecorder);
"""
    return Response(content=js, media_type="application/javascript")

@app.get('/metrics')
async def metrics():
    try:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        return Response("", media_type='text/plain')


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting Oviya WebSocket Server")
    print("=" * 60)
    print(f"   Ollama URL: {OLLAMA_URL}")
    print(f"   CSM URL: {CSM_URL}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

