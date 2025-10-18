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

# Optional JWT auth (fallback to allow if PyJWT missing or secret unset)
try:
    import jwt  # PyJWT
    def verify_jwt(token: str) -> bool:
        """
        Validate a JWT token using the OVIYA_SECRET environment variable.
        
        If the OVIYA_SECRET environment variable is unset or empty, verification is skipped and the function will accept any token.
        
        Parameters:
            token (str): JWT string to verify.
        
        Returns:
            `true` if the token is valid or verification is skipped, `false` otherwise.
        """
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
        """
        Validate the provided JWT token against the configured secret or accept it when verification is unavailable.
        
        Parameters:
            token (str): JWT token string to verify.
        
        Returns:
            bool: `True` if the token is considered valid, `False` otherwise.
        """
        return True

# Simple token-bucket rate limit per IP
BUCKET: Dict[str, tuple] = {}
CAP = 20
REFILL = CAP
WINDOW = 1.0
def _client_ip(ws: WebSocket) -> str:
    """
    Extract the client's IP address from a WebSocket connection.
    
    Parameters:
        ws (WebSocket): The connected WebSocket instance whose headers and client info are inspected.
    
    Returns:
        client_ip (str): The first IP from the `X-Forwarded-For` header if present; otherwise `ws.client.host` if available; `'unknown'` if no client information is available.
    """
    xf = ws.headers.get('x-forwarded-for')
    if xf:
        return xf.split(',')[0].strip()
    return ws.client.host if ws.client else 'unknown'
def allow(ip: str) -> bool:
    """
    Determine whether a request from the given client IP is permitted by the per-IP token bucket rate limiter and update the bucket state.
    
    Parameters:
        ip (str): Client IP address used as the rate-limiting key.
    
    Returns:
        bool: `True` if the request is allowed (a token was consumed), `False` otherwise.
    """
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
from voice.realtime_input_remote import RealTimeVoiceInput  # Using remote WhisperX API on Vast.ai
from emotion_detector.detector import EmotionDetector
from brain.llm_brain import OviyaBrain
from brain.secure_base import SecureBaseSystem
from brain.bids import BidResponseSystem
from emotion_controller.controller import EmotionController
from voice.openvoice_tts import HybridVoiceEngine
from voice.csm_1b_client import CSM1BClient
from voice.acoustic_emotion_detector import AcousticEmotionDetector
from brain.personality_store import PersonalityStore
from config.service_urls import OLLAMA_URL, CSM_URL
from faster_whisper import WhisperModel
import time

# Prefer local adapter, fallback to WebRTC implementation
try:
    from voice.silero_vad_adapter import SileroVAD  # decoupled adapter
    _HAS_SILERO_VAD = True
except Exception:
    try:
        from voice_server_webrtc import SileroVAD  # fallback
        _HAS_SILERO_VAD = True
    except Exception:
        _HAS_SILERO_VAD = False

# Optional session cleanup helper
try:
    from voice.session_state import cleanup_sessions
    _HAS_SESSION_CLEANUP = True
except Exception:
    _HAS_SESSION_CLEANUP = False

app = FastAPI(title="Oviya WebSocket Server")
api_v1 = APIRouter(prefix="/v1")
try:
    from serving.conditioning_api import router as conditioning_router
    app.include_router(conditioning_router)
except Exception:
    pass
@api_v1.post("/conversations")
async def api_create_conversation(payload: Dict):
    # Minimal stub - generate fake conversation_id
    """
    Create a new conversation and return its identifier.
    
    Parameters:
        payload (Dict): Request body for conversation creation; fields are accepted but ignored by this stub implementation.
    
    Returns:
        dict: A mapping with key `conversation_id` containing the new conversation's identifier as a string.
    """
    return {"conversation_id": f"c_{int(time.time()*1000)}"}

@api_v1.post("/conversations/{conversation_id}/turns")
async def api_add_turn(conversation_id: str, payload: Dict):
    """
    Handle a single non-streaming user turn and produce a structured assistant response.
    
    Parameters:
        conversation_id (str): Identifier for the conversation (used for routing/context; not otherwise modified).
        payload (Dict): Incoming turn data. Recognized keys:
            - "user_id" (str): Optional user identifier; defaults to "anonymous".
            - "text" (str): User utterance to process.
            - "context" (Dict): Optional contextual metadata (passed through but not required).
    
    Returns:
        Dict: A response object containing:
            - "assistant": {
                "text": assistant reply text,
                "emotion": mapped emotion label (e.g., "neutral"),
                "intensity": emotion intensity (0.0-1.0),
                "style_hint": optional style hint string,
                "situation": guidance category from the brain if available,
                "timing_plan": playback timing hints (e.g., {"pre_tts_delay_ms": 400})
              }
            - "safety": safety flags (e.g., {"flag": False})
            - "global_soul": placeholder for global personality/state
    """
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
    """
    Preload global models and start optional session cleanup task.
    
    Pre-initializes shared model instances used by the server. If session cleanup support is available, schedules a background task that calls `cleanup_sessions()` approximately every 60 seconds.
    """
    print("🚀 Pre-initializing models...")
    get_global_models()
    print("✅ Server ready for connections")
    if _HAS_SESSION_CLEANUP:
        async def _cleanup_loop():
            """
            Periodically invokes the session cleanup routine in a background loop.
            
            Calls cleanup_sessions() once every 60 seconds and suppresses any exceptions raised by that call. Runs indefinitely until the surrounding task is cancelled.
            """
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
        """
        Initialize the streaming speech-to-text processor with a Whisper model.
        
        Parameters:
            model_size (str): Whisper model identifier (e.g., "small.en") used to load the transcription model.
        
        Description:
            Loads a WhisperModel for streaming transcription, selects a GPU device when available (falls back otherwise), and configures compute precision accordingly. Initializes audio sampling rate (16000 Hz), an internal byte buffer for incoming PCM data, timing controls for partial-transcript emissions (emit interval and last emission timestamp), last emitted partial transcript tracker, and minimum/maximum window durations (in seconds) used when extracting audio slices for partial transcription.
        """
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
        """
        Append raw audio bytes to the instance's internal buffer.
        
        Parameters:
            audio_bytes (bytes): Raw PCM audio bytes to append to the buffer.
        """
        self.buffer.extend(audio_bytes)

    def should_commit(self, vad_confidence: float) -> bool:
        # Commit on strong VAD confidence to avoid mid-phoneme cuts
        """
        Decides whether accumulated audio should be committed as speech based on VAD confidence and buffer length.
        
        Parameters:
            vad_confidence (float): Voice activity detection confidence in the range [0.0, 1.0].
        
        Returns:
            `true` if `vad_confidence` is at least 0.7 and the internal audio buffer contains at least sample_rate * min_window_s * 2 samples, `false` otherwise.
        """
        return vad_confidence >= 0.7 and len(self.buffer) >= int(self.sample_rate * self.min_window_s) * 2

    def get_partial(self) -> str:
        """
        Return an updated partial transcription derived from the most recent buffered audio, or an empty string when no new partial is available.
        
        Enforces an emission cadence (does nothing if called sooner than the configured emit interval), requires at least the configured minimum audio window, and transcribes using up to the configured maximum window from the buffer. If the transcribed text is non-empty and different from the last emitted partial, the new partial is returned and stored; otherwise an empty string is returned. Errors during transcription are swallowed and result in an empty string.
        
        Returns:
            str: The new partial transcript if it changed since the last emission, otherwise an empty string.
        """
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
        """
        Initialize a conversation session for the given user, preparing shared models, per-session components, and streaming state.
        
        Creates or attaches shared/global models, instantiates per-session systems (LLM brain, emotion controller, TTS and STT clients, psych systems), loads the user's personality (if present) into the session context, and initializes fields used for streaming audio, VAD, memory, and optional components such as a breath sample and Silero VAD.
        
        Parameters:
            user_id (str): Unique identifier of the user for whom the session is created.
        """
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
        self.csm_streaming = CSM1BClient(use_local_model=False, remote_url=CSM_URL)
        self.stt = StreamingSTT()
        self.is_generating = False
        # Psych systems
        self.secure_base = SecureBaseSystem()
        self.bids = BidResponseSystem()
        
        # Load user personality
        self.personality = personality_store.load_personality(user_id)
        if self.personality:
            print(f"📚 Loaded personality for {user_id}")
            # Inject personality context into brain
            context = personality_store.get_conversation_summary(user_id, last_n=5)
            self.brain.context = context
        
        print(f"✅ Conversation session initialized for {user_id}")

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
        Process a PCM16LE audio chunk by feeding it to both the legacy voice-input buffer and the streaming STT, and return any available transcription.
        
        Parameters:
            audio_data (bytes): Raw PCM16LE audio bytes, 16-bit little-endian, 16 kHz, mono.
        
        Returns:
            Optional[dict]: Transcription result dictionary if a transcription is available, `None` otherwise.
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
        """
        Load a bundled breath audio sample for use in TTS.
        
        Attempts to load "audio_assets/breath_samples/gentle_exhale.wav", convert it to mono, resample to 24000 Hz if necessary, and scale its amplitude for subtle mixing. 
        
        Returns:
            numpy.ndarray: 1-D float32 PCM samples at 24000 Hz (shape: (n,)) scaled to low amplitude, or `None` if the file is missing or cannot be loaded.
        """
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
    
    async def generate_response(self, user_text: str, user_emotion: str) -> Dict:
        """
        Create an assistant response text and corresponding TTS audio chunks for a user turn.
        
        Returns:
            dict: {
                'text' (str): The assistant's response text.
                'emotion' (str): The mapped CSM emotion/style token used for TTS.
                'audio_chunks' (List[str]): Base64-encoded PCM audio chunks ready for streaming.
                'duration' (float): Estimated audio duration in seconds (based on 24 kHz sample rate).
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
        Stream synthesized TTS audio to the websocket while generating prosodic text from the brain.
        
        Cancels any active TTS stream, attempts to begin TTS as soon as a prosodic boundary is available from streaming LLM tokens (falls back to a full brain response if needed), and streams generated audio as JSON messages. Sends 'audio_chunk' messages containing base64 PCM_s16le frames (24 kHz), emits a 'first_audio_chunk' marker when the first audio is sent, and finally sends a 'response' message with the final text and emotion. Persists a memory triple for the turn and respects cancellation requests.
        """
        await self.cancel_tts_stream()

        # Stream LLM tokens while starting TTS ASAP
        start_time = time.time()
        prosodic_text = ""
        emotion = user_emotion or "neutral"
        assembled = []
        token_count = 0
        async for token in self.brain.think_streaming(user_text, user_emotion, conversation_history=None):
            assembled.append(token)
            token_count += 1
            # Kick off when first sentence boundary appears
            if not prosodic_text and any(p in ''.join(assembled) for p in ['.', '!', '?']):
                prosodic_text = ''.join(assembled)
                break
            # Token-bucket early trigger
            if not prosodic_text and token_count >= 20:
                prosodic_text = ''.join(assembled)
                break
        if not prosodic_text:
            # Fallback to full think if no early boundary
            brain_resp_full = self.brain.think(user_text, user_emotion)
            prosodic_text = brain_resp_full.get('prosodic_text') or brain_resp_full.get('text', '')
            emotion = brain_resp_full.get('emotion', emotion)
        intensity = brain_resp_full.get('intensity', 0.7) if 'brain_resp_full' in locals() else 0.7
        oviya_emotion_params = self.emotion_controller.map_emotion(
            emotion, intensity
        )

        # Store memory triple
        self._add_memory_triple(user_text, prosodic_text)

        self._tts_cancel_requested = False

        async def _stream():
            """
            Stream TTS audio from the CSM client to the websocket as base64-encoded PCM chunks.
            
            Continuously consumes audio frames produced by self.csm_streaming.generate_streaming, applies a short EMA smoothing, accumulates samples into ~80ms batches, and sends each batch to the websocket as a JSON message with type 'audio_chunk' (format 'pcm_s16le', sample_rate 24000). If a breath sample is configured and requested in the prosodic text, inserts the breath once at the start of the first sent audio. Sends a single 'first_audio_chunk' message the first time audio is delivered and records the time-to-first-audio metric when available. Stops early if self._tts_cancel_requested is set. On unexpected errors, sends an error JSON message describing the failure.
            """
            try:
                max_samples = 1920  # ~80ms at 24kHz
                buf, n = [], 0
                ttfb_sent = False
                async for c24 in self.csm_streaming.generate_streaming(
                    text=prosodic_text,
                    emotion=oviya_emotion_params.get('style_token', emotion),
                    speaker_id=0,
                    conversation_context=self._format_context_for_tts(),
                    reference_audio=self._reference_audio
                ):
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

    async def cancel_tts_stream(self):
        self._tts_cancel_requested = True
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
        """
        Build a short conversational context for TTS from the most recent memory triples.
        
        Returns:
            context (List[Dict]): Sequence of dictionaries each containing:
                - 'text' (str): utterance text.
                - 'speaker_id' (int): 1 for the user utterance (question), 0 for the assistant response.
            The list contains up to the last three memory triples, with each triple expanded into two entries
            in the order: user utterance then assistant response.
        """
        ctx = []
        for t in self._memory_triples[-3:]:
            ctx.append({'text': t['q'], 'speaker_id': 1})
            ctx.append({'text': t['r'], 'speaker_id': 0})
        return ctx
    
    def _ema_smooth(self, x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Apply exponential moving average smoothing along a 1-D numeric array.
        
        Parameters:
            x (np.ndarray): 1-D array of numeric samples to smooth.
            alpha (float): Smoothing factor in (0,1]; higher values weight recent samples more. Defaults to 0.1.
        
        Returns:
            np.ndarray: Smoothed array of the same shape as `x`. If smoothing cannot be performed, returns the original input `x`.
        """
        try:
            y = np.copy(x)
            for i in range(1, len(y)):
                y[i] = alpha * y[i] + (1 - alpha) * y[i - 1]
            return y
        except Exception:
            return x
    
    def _chunk_audio(self, audio: torch.Tensor, chunk_size: int = 4096) -> list:
        """
        Split a tensor of audio samples into base64-encoded PCM16 chunks suitable for JSON transmission.
        
        Parameters:
            audio (torch.Tensor): 1D or 2D float32 tensor of audio samples in the range [-1.0, 1.0].
            chunk_size (int): Number of samples per chunk.
        
        Returns:
            list: List of base64-encoded strings, each containing a PCM16 (signed 16-bit little-endian) audio chunk.
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
    """
    Return a simple HTML test page containing a client-side UI and JavaScript for connecting to the WebSocket voice chat.
    
    The page includes controls to connect, record audio (downsampling from 24k to 16k for transport), display transcripts and responses, and play streaming TTS audio via an in-page jitter-managed playback pipeline.
    
    Returns:
        HTMLResponse: Response containing the test page HTML.
    """
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
            """
            Periodically sends ping messages over the websocket to keep the connection alive.
            
            Sends a JSON message {"type": "ping", "t": <unix timestamp>} every 5 seconds and stops when sending fails (e.g., connection closed).
            """
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


@app.get("/health")
async def health_check():
    """
    Return a simple service health summary.
    
    Returns:
        dict: A dictionary with keys:
            - "status": service health status string, e.g., "healthy".
            - "service": human-readable service name.
            - "version": service version string.
    """
    return {
        "status": "healthy",
        "service": "Oviya WebSocket Server",
        "version": "1.0.0"
    }


@app.get('/worklet_ws.js')
async def worklet_ws():
    """
    Serve a small Web Audio Worklet module that captures audio, downsamples it, and forwards 16 kHz PCM16 buffers to the main thread.
    
    Returns:
    	A FastAPI `Response` containing the JavaScript worklet source with media type "application/javascript".
    """
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
    """
    Expose Prometheus metrics as an HTTP response.
    
    Attempts to return the current Prometheus metrics in the Prometheus text exposition format; if metrics generation fails, returns an empty plain-text response.
    
    Returns:
        Response: HTTP response with Prometheus metrics (Content-Type: prometheus) or an empty plain-text response on error.
    """
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
