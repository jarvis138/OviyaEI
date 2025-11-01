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
from typing import Dict, Optional, List, Tuple
import torch
from pathlib import Path
import time
from collections import deque
import os
import sys

# Add parent directory to path to find core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Session management: Removed session_manager.py (functionality merged into ConversationSession)
session_mgr = None
SESSION_MANAGER_AVAILABLE = False

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

# Import Oviya components with fail-safe
try:
    from .emotion_detector.detector import EmotionDetector
except ImportError:
    EmotionDetector = None

try:
    from .brain.llm_brain import OviyaBrain
except ImportError:
    OviyaBrain = None

try:
    from .brain.secure_base import SecureBaseSystem
except ImportError:
    SecureBaseSystem = None

try:
    from .brain.bids import BidResponseSystem
except ImportError:
    BidResponseSystem = None

try:
    from .emotion_controller.controller import EmotionController
except ImportError:
    EmotionController = None

try:
    from .voice.openvoice_tts import HybridVoiceEngine
except ImportError:
    HybridVoiceEngine = None

try:
    from .voice.csm_1b_client import CSM1BClient
except ImportError:
    CSM1BClient = None

try:
    from .voice.csm_1b_stream import CSMRVQStreamer, BatchedCSMStreamer, get_batched_streamer
except ImportError:
    CSMRVQStreamer = None
    BatchedCSMStreamer = None
    get_batched_streamer = None
# Removed: csm_1b_generator_optimized.py deleted (functionality in csm_1b_stream.py)
OptimizedCSMStreamer = None
get_optimized_streamer = None

try:
    from core.voice.acoustic_emotion_detector import AcousticEmotionDetector
except ImportError:
    AcousticEmotionDetector = None

try:
    from .brain.personality_store import PersonalityStore
except ImportError:
    PersonalityStore = None

try:
    from .config.service_urls import OLLAMA_URL, CSM_URL
except ImportError:
    OLLAMA_URL = "http://localhost:11434"
    CSM_URL = "http://localhost:8001"

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# Import MCP systems with fail-safe
try:
    from .brain.mcp_memory_integration import OviyaMemorySystem
except ImportError:
    OviyaMemorySystem = None

try:
    from .brain.crisis_detection import CrisisDetectionSystem
except ImportError:
    CrisisDetectionSystem = None

try:
    from .brain.empathic_thinking import EmpathicThinkingEngine
except ImportError:
    EmpathicThinkingEngine = None
try:
    from .brain.emotional_reciprocity import reciprocal_empathy_integrator
except ImportError:
    reciprocal_empathy_integrator = None

try:
    from .voice.strategic_silence import strategic_silence_manager, emotional_pacing_controller
except ImportError:
    strategic_silence_manager = None
    emotional_pacing_controller = None

# 🆕 EMOTION BLENDER & LIBRARY: Expand emotion range to 28+ emotions
try:
    from .voice.emotion_blender import EmotionBlender
except ImportError:
    EmotionBlender = None

try:
    from .voice.emotion_library import EmotionLibrary, get_emotion_library
except ImportError:
    EmotionLibrary = None
    get_emotion_library = None

# 🆕 EMOTION DISTRIBUTION MONITOR: Track and adapt emotion selection
try:
    from .shared.utils.emotion_monitor import EmotionDistributionMonitor, get_emotion_monitor
except ImportError:
    try:
        from ..shared.utils.emotion_monitor import EmotionDistributionMonitor, get_emotion_monitor
    except ImportError:
        try:
            from utils.emotion_monitor import EmotionDistributionMonitor, get_emotion_monitor
        except ImportError:
            EmotionDistributionMonitor = None
            get_emotion_monitor = None

# 🆕 GLOBAL SOUL: Cultural emotion wisdom integration
try:
    from .brain.global_soul import OviyaGlobalSoul
except ImportError:
    OviyaGlobalSoul = None

import time

# Prefer unified VAD+STT, fallback to legacy implementations
try:
    from .voice.unified_vad_stt import UnifiedVADSTTPipeline, get_unified_pipeline
    _HAS_UNIFIED_VAD_STT = True
except Exception:
    _HAS_UNIFIED_VAD_STT = False

# Legacy fallback
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
personality_store = PersonalityStore() if PersonalityStore else None

# Global model instances (shared across sessions for faster startup)
_global_voice_input = None
_global_emotion_detector = None
_global_acoustic_emotion = None

def get_global_models():
    """Initialize and return global model instances"""
    global _global_voice_input, _global_emotion_detector, _global_acoustic_emotion

    if _global_voice_input is None:
        print("🎤 Initializing global models (one-time setup)...")

        # Initialize voice input if available
        if hasattr(__import__('sys').modules[__name__], 'RealTimeVoiceInput') and 'RealTimeVoiceInput' in globals():
            _global_voice_input = RealTimeVoiceInput(enable_diarization=False)
            if hasattr(_global_voice_input, 'initialize_models'):
                _global_voice_input.initialize_models()
        else:
            _global_voice_input = None
            print("⚠️ RealTimeVoiceInput not available")

        # Initialize emotion detector if available
        if EmotionDetector:
            _global_emotion_detector = EmotionDetector()
        else:
            _global_emotion_detector = None
            print("⚠️ EmotionDetector not available")

        # Initialize acoustic emotion detector if available
        if AcousticEmotionDetector:
            _global_acoustic_emotion = AcousticEmotionDetector()
        else:
            _global_acoustic_emotion = None
            print("⚠️ AcousticEmotionDetector not available")

        print("✅ Global models initialization completed")

    return _global_voice_input, _global_emotion_detector, _global_acoustic_emotion


# StreamingSTT class removed - now using UnifiedVADSTTPipeline


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

        # Removed: optimized_streamer from deleted csm_1b_generator_optimized.py
        # CUDA graphs optimization is now handled in csm_1b_stream.py
        self.optimized_streamer = None

        # 🆕 INTEGRATE BATCHED RVQ STREAMING FOR MULTI-USER CONCURRENT AUDIO
        print("🎤 Initializing Batched CSM Streaming for multi-user conversations...")
        self.csm_streaming = get_batched_streamer()  # Global batched streamer

        # Start batch processor in background
        self.batch_processor_task = asyncio.create_task(
            self.csm_streaming.start_batch_processor()
        )
        print("✅ Batched streaming processor started")
        
        # Use unified VAD+STT pipeline
        if _HAS_UNIFIED_VAD_STT:
            self.vad_stt_pipeline = get_unified_pipeline()
            print("✅ Unified VAD+STT pipeline initialized")
        else:
            self.vad_stt_pipeline = None
            # Legacy fallback
            if _HAS_SILERO_VAD:
                from .voice.unified_vad_stt import OptimizedWhisperSTT
                self.stt = OptimizedWhisperSTT()
            else:
                self.stt = None
            print("⚠️ Using legacy STT implementation")
        
        self.is_generating = False

        # Psych systems
        self.secure_base = SecureBaseSystem()
        self.bids = BidResponseSystem()

        # Initialize MCP systems for emotional intelligence
        self.memory_system = OviyaMemorySystem()
        self.crisis_detector = CrisisDetectionSystem()
        self.empathy_engine = EmpathicThinkingEngine()
        
        # 🆕 TEMPORAL EMOTION TRACKER: Track emotion patterns over time
        try:
            from .brain.temporal_emotion_tracker import TemporalEmotionTracker
            self.temporal_emotion_tracker = TemporalEmotionTracker()
            print("✅ Temporal Emotion Tracker initialized")
        except Exception as e:
            print(f"⚠️ Temporal Emotion Tracker not available: {e}")
            self.temporal_emotion_tracker = None
        
        # 🆕 EMOTIONAL REASONING ENGINE: Advanced emotional reasoning
        try:
            from .brain.emotional_reasoning import EmotionalReasoningEngine
            self.emotional_reasoning = EmotionalReasoningEngine()
            print("✅ Emotional Reasoning Engine initialized")
        except Exception as e:
            print(f"⚠️ Emotional Reasoning Engine not available: {e}")
            self.emotional_reasoning = None
        
        # 🆕 PROSODY ENGINE: Initialize for personality-driven voice modulation
        try:
            from prosody_engine import ProsodyEngine
            self.prosody_engine = ProsodyEngine()
            print("✅ Prosody Engine initialized (personality-driven voice modulation)")
        except ImportError:
            try:
                from .prosody_engine import ProsodyEngine
                self.prosody_engine = ProsodyEngine()
                print("✅ Prosody Engine initialized (personality-driven voice modulation)")
            except Exception as e:
                print(f"⚠️ Prosody Engine not available: {e}")
                self.prosody_engine = None

        # 🆕 EMOTION BLENDER: Initialize for 28+ emotion expansion
        try:
            self.emotion_blender = EmotionBlender()
            print("✅ Emotion Blender initialized (28+ emotions)")
        except Exception as e:
            print(f"⚠️ Emotion Blender not available: {e}")
            self.emotion_blender = None

        # 🆕 EMOTION LIBRARY: Initialize for validation and tier-based selection
        try:
            self.emotion_library = get_emotion_library() if get_emotion_library else EmotionLibrary()
            print("✅ Emotion Library initialized (49-emotion taxonomy)")
        except Exception as e:
            print(f"⚠️ Emotion Library not available: {e}")
            self.emotion_library = None

        # 🆕 EMOTION DISTRIBUTION MONITOR: Track emotion usage
        try:
            self.emotion_monitor = get_emotion_monitor() if get_emotion_monitor else EmotionDistributionMonitor()
            print("✅ Emotion Distribution Monitor initialized")
        except Exception as e:
            print(f"⚠️ Emotion Monitor not available: {e}")
            self.emotion_monitor = None

        # 🆕 GLOBAL SOUL: Cultural emotion wisdom
        try:
            persona_config = self.brain.persona_config if hasattr(self.brain, 'persona_config') else {}
            self.global_soul = OviyaGlobalSoul(persona_config)
            print("✅ Global Soul initialized (Cultural wisdom: Ma, Jeong, Ahimsa, Logos, Lagom)")
        except Exception as e:
            print(f"⚠️ Global Soul not available: {e}")
            self.global_soul = None

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
        # 🆕 SPEECH-TO-SPEECH: Store user audio for CSM-1B
        self._user_audio_buffer: List[np.ndarray] = []  # Buffer user audio chunks (16kHz)
        self._current_user_audio: Optional[np.ndarray] = None  # Current user speech audio
        # VAD state for commit (energy-based)
        self._vad_is_speaking: bool = False
        self._vad_silence_ms: float = 0.0
        self._vad_speech_ms: float = 0.0
        # Breath sample (optional)
        self._breath_buf: Optional[np.ndarray] = self._load_breath_sample()
        self._breath_sent: bool = False
        # Silero VAD (if available, for legacy fallback)
        if not _HAS_UNIFIED_VAD_STT:
            self.silero_vad = SileroVAD() if _HAS_SILERO_VAD else None
            self._silero_remainder = np.zeros((0,), dtype=np.float32)
        else:
            self.silero_vad = None
            self._silero_remainder = None
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[Dict]:
        """
        Process incoming audio chunk using unified VAD+STT pipeline
        
        🆕 SPEECH-TO-SPEECH: Also stores user audio for CSM-1B
        
        Args:
            audio_data: Raw audio bytes (PCM, 16-bit, 16kHz, mono)
            
        Returns:
            Transcription result if available
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 🆕 SPEECH-TO-SPEECH: Store user audio for CSM-1B
        # Only store if user is speaking (not AI)
        if not self.is_generating:
            self._user_audio_buffer.append(audio_array.copy())
        
        # Use unified VAD+STT pipeline if available
        if self.vad_stt_pipeline:
            result = await self.vad_stt_pipeline.process_audio_chunk(audio_array)
            
            # Set AI speaking state for interrupt detection
            self.vad_stt_pipeline.set_ai_speaking_state(self.is_generating)
            
            # Return formatted result
            if result.get('final_text'):
                # 🆕 SPEECH-TO-SPEECH: Capture complete user audio when speech ends
                if self._user_audio_buffer:
                    self._current_user_audio = np.concatenate(self._user_audio_buffer)
                    self._user_audio_buffer.clear()  # Reset buffer
                return {
                    'text': result['final_text'],
                    'is_final': True,
                    'partial': False
                }
            elif result.get('partial_text'):
                return {
                    'text': result['partial_text'],
                    'is_final': False,
                    'partial': True
                }
        else:
            # Legacy fallback
            self.voice_input.add_audio_chunk(audio_array)
            if hasattr(self, 'stt') and self.stt:
                if hasattr(self.stt, 'add_audio'):
                    self.stt.add_audio(audio_array)
            result = self.voice_input.get_transcription(timeout=0.01)
            return result
        
        return None

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

    def _convert_personality_vector_for_prosody(
        self,
        personality_vector: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Convert personality vector format to ProsodyEngine expected format
        
        Args:
            personality_vector: Dict with keys "Ma", "Ahimsa", "Jeong", "Logos", "Lagom"
            
        Returns:
            Dict with 'pillars' key containing lowercase pillar names
        """
        # Handle both dict and list formats
        if isinstance(personality_vector, dict):
            # Extract values, handling both uppercase and lowercase keys
            pillars = {
                'ma': personality_vector.get('Ma', personality_vector.get('ma', 0.3)),
                'ahimsa': personality_vector.get('Ahimsa', personality_vector.get('ahimsa', 0.4)),
                'jeong': personality_vector.get('Jeong', personality_vector.get('jeong', 0.15)),
                'logos': personality_vector.get('Logos', personality_vector.get('logos', 0.1)),
                'lagom': personality_vector.get('Lagom', personality_vector.get('lagom', 0.05))
            }
        else:
            # List format [Ma, Ahimsa, Jeong, Logos, Lagom]
            pillars = {
                'ma': personality_vector[0] if len(personality_vector) > 0 else 0.3,
                'ahimsa': personality_vector[1] if len(personality_vector) > 1 else 0.4,
                'jeong': personality_vector[2] if len(personality_vector) > 2 else 0.15,
                'logos': personality_vector[3] if len(personality_vector) > 3 else 0.1,
                'lagom': personality_vector[4] if len(personality_vector) > 4 else 0.05
            }
        
        # Find dominant pillar
        dominant_pillar = max(pillars.items(), key=lambda x: x[1])[0]
        
        return {
            'pillars': pillars,
            'dominant_pillar': dominant_pillar
        }
    
    def _expand_emotion_with_blender(
        self,
        base_emotion: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Expand emotion using EmotionBlender to 28+ emotions
        
        CSM-1B Compatible: Returns expanded emotion that can be used with CSM-1B
        """
        if not self.emotion_blender:
            return base_emotion
        
        try:
            # Check if emotion is in blend recipes
            if base_emotion in self.emotion_blender.BLEND_RECIPES:
                # Already a blended emotion, return as-is
                return base_emotion
            
            # Check if we should use a blended emotion based on context
            # For now, use base emotion, but could add logic to select blended emotions
            # based on intensity, context, etc.
            return base_emotion
        except Exception as e:
            print(f"⚠️ Emotion blending failed: {e}")
            return base_emotion

    def _validate_and_resolve_emotion(
        self,
        emotion: str
    ) -> Tuple[str, str]:
        """
        Validate and resolve emotion using EmotionLibrary
        
        Returns:
            (resolved_emotion, tier)
        """
        if not self.emotion_library:
            return emotion, "tier1_core"
        
        try:
            # Resolve emotion (handles aliases)
            resolved_emotion = self.emotion_library.get_emotion(emotion)
            
            # Get tier
            tier = self.emotion_library.get_tier(resolved_emotion)
            
            return resolved_emotion, tier
        except Exception as e:
            print(f"⚠️ Emotion validation failed: {e}")
            return emotion, "tier1_core"
    
    def _compute_prosody_for_csm(
        self,
        emotion: str,
        intensity: float,
        personality_vector: Dict[str, float],
        response_text: str,
        reciprocal_emotion_metadata: Optional[Dict] = None,
        user_emotion_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute prosody parameters using ProsodyEngine and convert to CSM-1B format
        
        🆕 CSM-1B Compatible:
        - Uses VAD dimensions from emotion embeddings to enhance prosody
        - Integrates temporal patterns and emotional reasoning
        
        Args:
            emotion: Primary emotion
            intensity: Emotion intensity (0-1)
            personality_vector: Oviya's personality vector (Ma, Ahimsa, Jeong, Logos, Lagom)
            response_text: Text being synthesized
            reciprocal_emotion_metadata: Optional reciprocal emotion metadata
            user_emotion_embedding: Optional emotion embedding for VAD extraction
            
        Returns:
            Dict with CSM-1B compatible prosody parameters
        """
        if not self.prosody_engine:
            # Fallback: return basic prosody from emotion_controller
            return {
                'pitch_scale': 1.0,
                'rate_scale': 1.0,
                'energy_scale': 1.0
            }
        
        try:
            # 🆕 CULTURAL WISDOM INTEGRATION: Get cultural prosody hints
            cultural_prosody_hints = {}
            if self.global_soul:
                try:
                    ctx = {
                        "emotional_weight": intensity,
                        "intensity": intensity,
                        "draft": response_text[:100],  # First 100 chars
                        "session_seconds": len(self._memory_triples) * 30,  # Estimate
                        "vulnerability": intensity if intensity > 0.7 else 0.0,
                        "regular_checkin": len(self._memory_triples) > 5
                    }
                    cultural_plan = self.global_soul.plan(self.user_id, ctx)
                    
                    # Extract cultural prosody hints
                    if cultural_plan.get("ma"):
                        ma_pause = cultural_plan["ma"].get("pause_before_ms", 0)
                        cultural_prosody_hints["ma_pause_ms"] = ma_pause
                        cultural_prosody_hints["pause_quality"] = cultural_plan["ma"].get("pause_quality", "natural")
                    
                    if cultural_plan.get("sattva"):
                        sattva = cultural_plan["sattva"]
                        cultural_prosody_hints["energy"] = sattva.get("energy", 1.0)
                        cultural_prosody_hints["tone"] = sattva.get("tone", "steady_presence")
                        cultural_prosody_hints["prosody"] = sattva.get("prosody", "balanced")
                    
                    if cultural_plan.get("lagom"):
                        cultural_prosody_hints["lagom_balance"] = cultural_plan["lagom"]
                except Exception as e:
                    print(f"⚠️ Cultural wisdom integration failed: {e}")
            
            # 🆕 VAD ENHANCEMENT: Extract VAD dimensions from emotion embedding
            vad_dimensions = None
            if user_emotion_embedding is not None:
                try:
                    from .brain.emotion_embeddings import get_emotion_embedding_generator
                    emotion_embed_gen = get_emotion_embedding_generator()
                    vad_dimensions = emotion_embed_gen.embedding_to_vad(user_emotion_embedding)
                    
                    # Enhance intensity with valence
                    if vad_dimensions:
                        intensity = intensity * (0.5 + vad_dimensions['valence'])
                        intensity = max(0.0, min(1.0, intensity))
                except Exception as e:
                    print(f"⚠️ VAD extraction failed: {e}")
            
            # Convert personality vector format
            personality_for_prosody = self._convert_personality_vector_for_prosody(personality_vector)
            
            # Prepare reciprocal emotion dict
            reciprocal_emotion_dict = {
                'ovi_emotion': 'neutral',
                'intensity': 0.5
            }
            if reciprocal_emotion_metadata:
                reciprocal_emotion_dict = {
                    'ovi_emotion': reciprocal_emotion_metadata.get('shared_emotion', 'neutral'),
                    'intensity': reciprocal_emotion_metadata.get('confidence', 0.7)
                }
            
            # 🆕 VAD-AWARE PROSODY: Add VAD dimensions to prosody computation
            prosody_kwargs = {
                'emotion': emotion,
                'intensity': intensity,
                'personality_vector': personality_for_prosody,
                'response_text': response_text,
                'reciprocal_emotion': reciprocal_emotion_dict
            }
            
            # Add VAD dimensions if available
            if vad_dimensions:
                prosody_kwargs['valence'] = vad_dimensions['valence']
                prosody_kwargs['arousal'] = vad_dimensions['arousal']
                prosody_kwargs['dominance'] = vad_dimensions['dominance']
            
            # Compute prosody parameters
            prosody_result = self.prosody_engine.compute_prosody_parameters(**prosody_kwargs)
            
            prosody_params = prosody_result['parameters']
            
            # Convert to CSM-1B format
            # CSM-1B uses relative scales (1.0 = neutral)
            csm_prosody = {
                'pitch_scale': 1.0 + prosody_params.get('f0_mean', 0.0),  # Convert semitones to scale
                'rate_scale': prosody_params.get('speech_rate', 1.0),
                'energy_scale': 1.0 + prosody_params.get('energy', 0.0),  # Convert relative to scale
                'pause_probability': prosody_params.get('pause_probability', 0.1),
                'intonation_curve': prosody_params.get('intonation_curve', 'neutral'),
                'f0_range': prosody_params.get('f0_range', 0.0),
                'prosody_explanation': prosody_result.get('explanation', ''),
                'personality_influence': prosody_result.get('personality_influence', 'balanced')
            }
            
            # 🆕 CULTURAL WISDOM: Merge cultural prosody hints
            if cultural_prosody_hints:
                # Adjust prosody based on cultural wisdom
                if "ma_pause_ms" in cultural_prosody_hints:
                    # Ma (contemplative space) → slower speech, more pauses
                    csm_prosody['pause_probability'] = min(0.3, csm_prosody['pause_probability'] + 0.1)
                    csm_prosody['rate_scale'] *= 0.95  # Slightly slower
                
                if "energy" in cultural_prosody_hints:
                    # Sattva balance → adjust energy
                    csm_prosody['energy_scale'] = cultural_prosody_hints["energy"]
                
                if "prosody" in cultural_prosody_hints:
                    prosody_type = cultural_prosody_hints["prosody"]
                    if prosody_type == "slower_softer":
                        csm_prosody['rate_scale'] *= 0.9
                        csm_prosody['energy_scale'] *= 0.85
                    elif prosody_type == "warmer_gentler":
                        csm_prosody['pitch_scale'] *= 1.05
                        csm_prosody['energy_scale'] *= 0.9
                
                print(f"🌍 Cultural prosody applied: {cultural_prosody_hints.get('pause_quality', 'natural')}")
            
            # Clamp values to safe ranges
            csm_prosody['pitch_scale'] = max(0.5, min(1.5, csm_prosody['pitch_scale']))
            csm_prosody['rate_scale'] = max(0.5, min(1.5, csm_prosody['rate_scale']))
            csm_prosody['energy_scale'] = max(0.5, min(1.5, csm_prosody['energy_scale']))
            
            print(f"🎭 Computed prosody: pitch={csm_prosody['pitch_scale']:.2f}, rate={csm_prosody['rate_scale']:.2f}, energy={csm_prosody['energy_scale']:.2f}")
            print(f"   Personality influence: {csm_prosody['personality_influence']}")
            
            return csm_prosody
            
        except Exception as e:
            print(f"⚠️ Prosody computation failed: {e}")
            # Fallback to neutral prosody
            return {
                'pitch_scale': 1.0,
                'rate_scale': 1.0,
                'energy_scale': 1.0
            }
    
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
        
        # 🆕 ENHANCE: Use brain's personality vector if available (more accurate)
        try:
            # Try to get personality vector from brain
            brain_result = self.brain.think(user_text, user_emotion)
            if 'personality_vector' in brain_result:
                brain_personality = brain_result['personality_vector']
                if isinstance(brain_personality, list) and len(brain_personality) >= 5:
                    base_vector = {
                        "Ma": float(brain_personality[0]),
                        "Ahimsa": float(brain_personality[1]),
                        "Jeong": float(brain_personality[2]),
                        "Logos": float(brain_personality[3]),
                        "Lagom": float(brain_personality[4])
                    }
                    print(f"🧠 Using brain's personality vector: Ma={base_vector['Ma']:.2f}, Jeong={base_vector['Jeong']:.2f}")
        except Exception as e:
            print(f"⚠️ Could not get brain personality vector: {e}")
        
        # 🆕 ENHANCE: Use ChromaDB memory evolution if available
        try:
            # Get personality evolution from ChromaDB
            memory_results = await self.memory_system.retrieve_relevant_memories(
                self.user_id, user_text, limit=10
            )
            if memory_results and 'personality_evolution' in memory_results:
                evolution = memory_results['personality_evolution']
                if evolution and len(evolution) > 0:
                    # Use recent personality trend (last 3 interactions)
                    recent_personality = evolution[-1] if evolution else None
                    if recent_personality:
                        # Blend current with historical trend (70% current, 30% historical)
                        for pillar in base_vector:
                            historical_weight = 0.3
                            current_weight = 0.7
                            # Extract from ChromaDB metadata if available
                            # This creates consistency over time
                            base_vector[pillar] = (
                                base_vector[pillar] * current_weight +
                                base_vector.get(pillar, 0.5) * historical_weight
                            )
        except Exception as e:
            print(f"⚠️ Could not retrieve personality evolution: {e}")

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

    async def generate_response_streaming(self, websocket: WebSocket, user_text: str, user_emotion: str, user_audio: Optional[np.ndarray] = None):
        """
        Stream TTS audio chunks as they are generated with MCP emotional intelligence.
        Sends 'audio_chunk' messages and a terminal 'response' with text/emotion.
        
        🆕 SPEECH-TO-SPEECH: Accepts user audio for CSM-1B native speech-to-speech
        
        Args:
            websocket: WebSocket connection
            user_text: Transcribed user text
            user_emotion: Detected user emotion
            user_audio: Optional user audio waveform (16kHz) for speech-to-speech
        """
        await self.cancel_tts_stream()
        
        # 🆕 SPEECH-TO-SPEECH: Use current user audio if not provided
        if user_audio is None and self._current_user_audio is not None:
            user_audio = self._current_user_audio

        # MCP INTEGRATION: Crisis detection first (SAFETY FIRST)
        crisis_assessment = await self.crisis_detector.assess_crisis_risk(
            user_text, [msg.get('text', '') for msg in self._memory_triples[-10:]]
        )

        if crisis_assessment["escalation_needed"]:
            # Handle crisis with immediate intervention
            emergency_resources = await self.crisis_detector.get_emergency_resources()
            crisis_response = crisis_assessment["immediate_response"]
            
            # 🆕 CSM-1B COMPATIBLE: Crisis responses need calmer, measured prosody
            # Generate crisis response audio with appropriate prosody
            crisis_emotion = "calm_supportive"  # Calm for crisis situations
            crisis_prosody = {
                'pitch_scale': 0.95,  # Slightly lower, more measured
                'rate_scale': 0.85,   # Slower, more deliberate
                'energy_scale': 0.9,   # Softer, more supportive
                'pause_probability': 0.15,  # More pauses for emphasis
                'personality_influence': 'crisis_support'
            }
            
            # Format conversation context for CSM-1B (even for crisis)
            conversation_context = self._format_context_for_tts()
            
            # Submit crisis response to CSM-1B with special prosody
            batch_id = await self.csm_streaming.submit_stream_request(
                user_id=self.user_id,
                text=crisis_response,
                emotion=crisis_emotion,
                speaker_id=42,
                conversation_context=conversation_context,
                user_audio=None,  # No user audio for crisis responses
                reference_audio=None,
                prosody_params=crisis_prosody,
                priority=10  # Highest priority for crisis
            )
            
            # Stream crisis audio with calm prosody
            async def _stream_crisis():
                max_samples = 1920
                buf, n = [], 0
                async for c24 in self.csm_streaming.get_stream_results(self.user_id):
                    c24 = self._ema_smooth(c24, alpha=0.1)
                    buf.append(c24)
                    n += len(c24)
                    if n >= max_samples:
                        arr = np.concatenate(buf)
                        await websocket.send_json({
                            'type': 'crisis_intervention',
                            'format': 'pcm_s16le',
                            'sample_rate': 24000,
                            'audio_base64': base64.b64encode((arr * 32767).astype(np.int16).tobytes()).decode('utf-8'),
                            'text': crisis_response,
                            'emergency_resources': emergency_resources,
                            'severity': crisis_assessment["risk_level"]
                        })
                        buf, n = [], 0
                if buf:
                    arr = np.concatenate(buf)
                    await websocket.send_json({
                        'type': 'crisis_intervention',
                        'format': 'pcm_s16le',
                        'sample_rate': 24000,
                        'audio_base64': base64.b64encode((arr * 32767).astype(np.int16).tobytes()).decode('utf-8'),
                        'text': crisis_response,
                        'emergency_resources': emergency_resources,
                        'severity': crisis_assessment["risk_level"]
                    })
            
            await _stream_crisis()
            return

        # MCP INTEGRATION: Retrieve relevant memories for context
        relevant_memories = await self.memory_system.retrieve_relevant_memories(
            self.user_id, user_text, limit=5
        )

        # MCP INTEGRATION: Compute current personality vector
        personality_vector = await self._compute_personality_vector(user_text, user_emotion)

        # 🆕 TEMPORAL EMOTION TRACKING: Track emotion state
        emotion_intensity = 0.7  # Default, will be enhanced
        if self.temporal_emotion_tracker:
            try:
                # Get emotion detector for intensity
                emotion_result = self.emotion_detector.detect_emotion(user_text)
                emotion_intensity = emotion_result.get('intensity', 0.7)
                
                # 🆕 ACOUSTIC EMOTION DETECTOR: Enhance emotion detection with audio features
                if user_audio is not None and self.acoustic_emotion:
                    try:
                        acoustic_result = self.acoustic_emotion.detect_emotion(user_audio, sample_rate=16000)
                        # Get Oviya emotion from acoustic result (mapped to 49-emotion taxonomy)
                        acoustic_emotions = acoustic_result.get('oviya_emotions', [])
                        if acoustic_emotions:
                            acoustic_emotion = acoustic_emotions[0]  # Use first mapped emotion
                        else:
                            # Fallback to base emotion mapping
                            base_emotion = acoustic_result.get('emotion', user_emotion)
                            acoustic_emotion = base_emotion
                        
                        acoustic_arousal = acoustic_result.get('arousal', 0.5)
                        acoustic_valence = acoustic_result.get('valence', 0.5)
                        acoustic_confidence = acoustic_result.get('confidence', 0.5)
                        
                        # Blend acoustic and text-based emotion
                        if acoustic_emotion != user_emotion:
                            # Prefer acoustic emotion if confidence is high
                            if acoustic_confidence > 0.7:
                                user_emotion = acoustic_emotion
                                print(f"🎵 Acoustic emotion override: {acoustic_emotion} (confidence: {acoustic_confidence:.2f})")
                        
                        # Enhance intensity with acoustic features
                        emotion_intensity = max(emotion_intensity, (acoustic_arousal + abs(acoustic_valence)) / 2.0)
                    except Exception as e:
                        print(f"⚠️ Acoustic emotion detection failed: {e}")
                
                # Get VAD dimensions if available
                valence = 0.5
                arousal = 0.5
                dominance = 0.5
                
                # Extract emotion embedding for VAD
                try:
                    from .brain.emotion_embeddings import get_emotion_embedding_generator
                    emotion_embed_gen = get_emotion_embedding_generator()
                    user_emotion_embed = emotion_embed_gen.extract_combined_emotion_embedding(
                        audio=user_audio if user_audio is not None else None,
                        text=user_text,
                        sample_rate=16000
                    )
                    vad_dimensions = emotion_embed_gen.embedding_to_vad(user_emotion_embed)
                    valence = vad_dimensions['valence']
                    arousal = vad_dimensions['arousal']
                    dominance = vad_dimensions['dominance']
                    emotion_intensity = (valence + arousal + dominance) / 3.0
                except Exception as e:
                    print(f"⚠️ VAD extraction failed: {e}")
                
                # Add emotion state to tracker
                self.temporal_emotion_tracker.add_emotion_state(
                    emotion=user_emotion,
                    intensity=emotion_intensity,
                    confidence=0.8,
                    valence=valence,
                    arousal=arousal,
                    dominance=dominance,
                    context=user_text[:100]  # First 100 chars
                )
                
                # Get temporal patterns for context
                temporal_context = self.temporal_emotion_tracker.get_context_for_csm()
            except Exception as e:
                print(f"⚠️ Temporal tracking failed: {e}")
                temporal_context = None
        else:
            temporal_context = None
        
        # 🆕 EMOTIONAL REASONING: Reason about emotional cause and goals
        emotional_reasoning = None
        if self.emotional_reasoning:
            try:
                conversation_context = [
                    {'text': turn.get('q', turn.get('r', '')), 'speaker_id': turn.get('speaker_id', 1)}
                    for turn in self._memory_triples[-5:]
                ]
                
                emotional_reasoning = self.emotional_reasoning.get_reasoning_for_csm(
                    current_emotion=user_emotion,
                    intensity=emotion_intensity,
                    conversation_context=conversation_context,
                    temporal_patterns=temporal_context
                )
            except Exception as e:
                print(f"⚠️ Emotional reasoning failed: {e}")
        
        # MCP INTEGRATION: Generate deep empathic response
        emotion_context = {
            "emotion": user_emotion,
            "intensity": emotion_intensity,
            "patterns": relevant_memories.get("conversation_history", []),
            "conflicts": [],  # Could be detected from conversation flow
            "temporal_patterns": temporal_context,  # 🆕 Add temporal context
            "emotional_reasoning": emotional_reasoning  # 🆕 Add emotional reasoning
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
        # 🆕 REAL EMOTION EMBEDDINGS: Use EmotionEmbeddingGenerator
        from .brain.emotion_embeddings import get_emotion_embedding_generator
        emotion_embed_gen = get_emotion_embedding_generator()
        
        # Extract combined emotion embedding from audio and text
        user_emotion_embed = emotion_embed_gen.extract_combined_emotion_embedding(
            audio=user_audio if user_audio is not None else None,
            text=user_text,
            sample_rate=16000,
            audio_weight=0.6  # Favor audio for emotional prosody
        )
        
        oviya_personality_tensor = torch.tensor(list(personality_vector.values()))

        # Enhance response with reciprocal empathy
        enhanced_response_text, reciprocity_metadata = await reciprocal_empathy_integrator.enhance_response_with_reciprocity(
            response_text=empathic_response["response"],
            user_emotion_embed=user_emotion_embed,
            oviya_personality=oviya_personality_tensor,
            conversation_context={
                "emotion": user_emotion,
                "depth": len(self._memory_triples),
                "emotion_intensity": emotion_context.get("intensity", 0.7)
            }
        )
        
        # 🆕 POSITIVE AFFIRMATIONS: Use AI Therapist MCP for targeted affirmations
        # Check if user might benefit from affirmations (low self-worth, high vulnerability)
        should_add_affirmations = (
            emotion_context.get("intensity", 0.7) < 0.4 or  # Low confidence
            any(word in user_text.lower() for word in ['worthless', 'not good enough', 'stupid', 'failure', 'hate myself']) or
            len(self._memory_triples) > 10  # Established relationship
        )
        
        affirmation_text = None
        if should_add_affirmations:
            try:
                # Get AI Therapist MCP client
                from .brain.mcp_client import get_mcp_client
                ai_therapist = get_mcp_client("ai-therapist")
                
                if ai_therapist:
                    # Initialize if needed
                    if not hasattr(ai_therapist, '_initialized') or not ai_therapist._initialized:
                        await ai_therapist.initialize()
                    
                    # Map user emotion to affirmation focus area
                    affirmation_focus = {
                        'sad': 'self_worth',
                        'anxious': 'capabilities',
                        'angry': 'resilience',
                        'fearful': 'purpose',
                        'lonely': 'relationships'
                    }.get(user_emotion, 'self_worth')
                    
                    # Get targeted affirmation
                    affirmation_result = await ai_therapist.call_tool("positive_affirmations", {
                        "focus_area": affirmation_focus,
                        "specific_concerns": [user_text[:100]],  # First 100 chars
                        "tone": "gentle"  # Gentle tone for Oviya
                    })
                    
                    # Extract affirmation text
                    if isinstance(affirmation_result, dict):
                        affirmation_text = affirmation_result.get("affirmation", affirmation_result.get("text", ""))
                        if affirmation_text:
                            # Integrate affirmation naturally into response
                            enhanced_response_text = f"{enhanced_response_text} {affirmation_text}"
                            print(f"💝 Added positive affirmation: {affirmation_text[:50]}...")
            except Exception as e:
                print(f"⚠️ Positive affirmations failed: {e}")
        
        # Store reciprocity_metadata for prosody computation
        # This will be used by ProsodyEngine to modulate voice based on Oviya's reciprocal emotion

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

        # 🆕 PROSODY ENGINE INTEGRATION: Compute personality-driven prosody
        # This integrates all 5 personality pillars (Ma, Jeong, Ahimsa, Logos, Lagom)
        # with emotional reciprocity and therapeutic context
        # 🆕 VAD ENHANCEMENT: Include emotion embedding for VAD-based prosody
        personality_prosody = self._compute_prosody_for_csm(
            emotion=user_emotion or "neutral",
            intensity=emotion_context.get("intensity", 0.7),
            personality_vector=personality_vector,
            response_text=prosodic_text,
            reciprocal_emotion_metadata=reciprocity_metadata if 'reciprocity_metadata' in locals() else None,
            user_emotion_embedding=user_emotion_embed if 'user_emotion_embed' in locals() else None
        )
        
        # 🆕 EMOTION LIBRARY: Validate and resolve emotion
        validated_emotion, emotion_tier = self._validate_and_resolve_emotion(emotion)
        
        # 🆕 EMOTION BLENDER: Expand to 28+ emotions if appropriate
        expanded_emotion = self._expand_emotion_with_blender(
            validated_emotion,
            context=emotion_context
        )
        
        # 🆕 EMOTION DISTRIBUTION MONITOR: Record emotion usage
        if self.emotion_monitor:
            try:
                self.emotion_monitor.record_emotion(expanded_emotion)
                # Check distribution health and adapt if needed
                health = self.emotion_monitor.check_distribution_health()
                if health.get("status") == "unhealthy":
                    # Could adjust emotion selection based on distribution
                    print(f"⚠️ Emotion distribution unhealthy: {health.get('issues', [])}")
            except Exception as e:
                print(f"⚠️ Emotion monitoring failed: {e}")
        
        # Map emotion to CSM-1B format and compute prosody
        oviya_emotion_params = self.emotion_controller.map_emotion(
            expanded_emotion,  # Use expanded emotion
            emotion_context.get("intensity", 0.7)
        )
        
        # 🆕 MERGE PROSODY: Merge personality-driven prosody with emotion params
        # personality_prosody contains pitch_scale, rate_scale, energy_scale from ProsodyEngine
        # Merge into oviya_emotion_params for CSM-1B
        if personality_prosody:
            # Merge prosody parameters
            oviya_emotion_params.update({
                'pitch_scale': personality_prosody.get('pitch_scale', 1.0),
                'rate_scale': personality_prosody.get('rate_scale', 1.0),
                'energy_scale': personality_prosody.get('energy_scale', 1.0),
                'pause_probability': personality_prosody.get('pause_probability', 0.1),
                'intonation_curve': personality_prosody.get('intonation_curve', 'neutral'),
                'f0_range': personality_prosody.get('f0_range', 0.0),
                'personality_influence': personality_prosody.get('personality_influence', 'balanced'),
                'prosody_explanation': personality_prosody.get('prosody_explanation', '')
            })
        
        print(f"🎭 Final emotion params: {oviya_emotion_params.get('style_token', emotion)}")
        print(f"   Prosody: pitch={oviya_emotion_params.get('pitch_scale', 1.0):.2f}, rate={oviya_emotion_params.get('rate_scale', 1.0):.2f}, energy={oviya_emotion_params.get('energy_scale', 1.0):.2f}")

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
            # Use brain's personality vector if available
            if 'personality_vector' in brain_resp_full:
                personality_vector = {
                    "Ma": brain_resp_full['personality_vector'][0],
                    "Ahimsa": brain_resp_full['personality_vector'][1],
                    "Jeong": brain_resp_full['personality_vector'][2],
                    "Logos": brain_resp_full['personality_vector'][3],
                    "Lagom": brain_resp_full['personality_vector'][4]
                }
            
            # 🆕 PROSODY COMPUTATION FOR FALLBACK PATH
            # Even in fallback, compute prosody for personality-driven voice
            if self.prosody_engine:
                try:
                    personality_prosody = self._compute_prosody_for_csm(
                        emotion=emotion,
                        intensity=0.7,
                        personality_vector=personality_vector,
                        response_text=prosodic_text,
                        reciprocal_emotion_metadata=reciprocity_metadata if 'reciprocity_metadata' in locals() else None
                    )
                    # Merge prosody into emotion params
                    oviya_emotion_params.update(personality_prosody)
                except Exception as e:
                    print(f"⚠️ Fallback prosody computation failed: {e}")

        # Map emotion to CSM-1B format using brain's helper
        try:
            csm_emotion = self.brain.map_emotion_to_csm_format(
                oviya_emotion_params.get('style_token', emotion)
            )
            oviya_emotion_params['csm_emotion'] = csm_emotion
        except Exception:
            oviya_emotion_params['csm_emotion'] = oviya_emotion_params.get('style_token', emotion)

        # MCP INTEGRATION: Store conversation in memory system
        await self.memory_system.store_conversation_memory(self.user_id, {
            "user_input": user_text,
            "response": prosodic_text,
            "timestamp": time.time(),
            "emotion": emotion,
            "personality_vector": personality_vector,
            "session_id": f"session_{int(time.time())}"
        })

        # Store memory triple with emotion info and user audio
        self._add_memory_triple(user_text, prosodic_text, user_emotion, oviya_emotion_params.get('style_token', emotion), user_audio=user_audio)

        self._tts_cancel_requested = False

        async def _stream():
            try:
                max_samples = 1920  # ~80ms at 24kHz
                buf, n = [], 0
                ttfb_sent = False
                        # 🆕 BATCHED RVQ STREAMING: Multi-user concurrent audio generation
                # Format conversation context with audio references
                conversation_context = self._format_context_for_tts()
                
                # Get reference audio for current emotion if available
                # 🆕 HANDLE EMOTION REFERENCE DIRECTORY PATHS: Support both local and VastAI paths
                emotion_ref_dirs = [
                    Path("data/emotion_references"),  # Local development
                    Path("/workspace/emotion_references"),  # VastAI deployment
                    Path("/workspace/data/emotion_references"),  # Alternative VastAI path
                ]
                emotion_map_file = None
                emotion_ref_dir_found = None
                reference_audio = None
                
                # Try to find emotion_map.json in any of the possible directories
                for emotion_ref_dir in emotion_ref_dirs:
                    potential_file = emotion_ref_dir / "emotion_map.json"
                    if potential_file.exists():
                        emotion_map_file = potential_file
                        emotion_ref_dir_found = emotion_ref_dir
                        break
                
                if emotion_map_file:
                    try:
                        with open(emotion_map_file, 'r') as f:
                            emotion_map = json.load(f)
                        current_emotion = oviya_emotion_params.get('style_token', emotion)
                        if current_emotion in emotion_map and emotion_map[current_emotion] and emotion_ref_dir_found:
                            ref_file = emotion_ref_dir_found / emotion_map[current_emotion][0]['file']
                            if ref_file.exists():
                                import torchaudio
                                audio_tensor, sr = torchaudio.load(str(ref_file))
                                reference_audio = audio_tensor.squeeze().numpy().astype(np.float32)
                    except Exception:
                        pass
                
                # 🆕 SPEECH-TO-SPEECH: Submit request with user audio for native speech-to-speech
                # Resample user audio to 24kHz if provided (CSM-1B expects 24kHz)
                user_audio_24k = None
                if user_audio is not None:
                    try:
                        import torchaudio
                        audio_tensor = torch.tensor(user_audio, dtype=torch.float32)
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        # Resample from 16kHz to 24kHz
                        user_audio_24k = torchaudio.functional.resample(audio_tensor, 16000, 24000)
                        user_audio_24k = user_audio_24k.squeeze(0).numpy().astype(np.float32)
                        print(f"🎤 Resampled user audio: {len(user_audio)} samples (16kHz) → {len(user_audio_24k)} samples (24kHz)")
                    except Exception as e:
                        print(f"⚠️ Failed to resample user audio: {e}")
                        user_audio_24k = None
                
                # Submit request to batch processor for optimal GPU utilization
                # 🆕 PASS PROSODY PARAMS AND REFERENCE AUDIO
                batch_id = await self.csm_streaming.submit_stream_request(
                    user_id=self.user_id,
                    text=prosodic_text,
                    emotion=oviya_emotion_params.get('style_token', emotion),
                    speaker_id=42,  # Oviya's consistent voice
                    conversation_context=conversation_context,
                    user_audio=user_audio_24k,  # 🆕 SPEECH-TO-SPEECH: Pass user audio
                    reference_audio=reference_audio,  # 🆕 REFERENCE AUDIO: Pass emotion reference audio
                    prosody_params=oviya_emotion_params,  # 🆕 PROSODY: Pass prosody parameters (pitch_scale, rate_scale, energy_scale)
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

    def _add_memory_triple(self, q: str, r: str, user_emotion: str = "neutral", oviya_emotion: str = "neutral", user_audio: Optional[np.ndarray] = None):
        """
        Add memory triple with emotion tracking
        
        🆕 SPEECH-TO-SPEECH: Store user audio for CSM-1B context
        """
        p = (self.brain.context[:200] if hasattr(self.brain, 'context') and self.brain.context else "")
        triple = {
            'q': q, 
            'p': p, 
            'r': r,
            'emotion': user_emotion,
            'oviya_emotion': oviya_emotion
        }
        # 🆕 SPEECH-TO-SPEECH: Store user audio if available
        if user_audio is not None:
            triple['q_audio'] = user_audio
        self._memory_triples.append(triple)
        if len(self._memory_triples) > 50:
            self._memory_triples = self._memory_triples[-50:]

    def _format_context_for_tts(self) -> List[Dict]:
        """
        Format conversation context for CSM-1B
        
        Uses brain's format_conversation_context_for_csm() method for consistency.
        
        According to Sesame's format:
        - Include text and optional audio references
        - Last 3 turns for optimal context
        - Speaker IDs: 1 = user, 42 = Oviya
        """
        # Use brain's method to format context (preferred)
        try:
            return self.brain.format_conversation_context_for_csm(
                memory_triples=self._memory_triples,
                include_audio=True
            )
        except Exception as e:
            print(f"⚠️ Brain formatting failed, using fallback: {e}")
            # Fallback to manual formatting
            ctx = []
            
            # 🆕 HANDLE EMOTION REFERENCE DIRECTORY PATHS: Support both local and VastAI paths
            emotion_ref_dirs = [
                Path("data/emotion_references"),  # Local development
                Path("/workspace/emotion_references"),  # VastAI deployment
                Path("/workspace/data/emotion_references"),  # Alternative VastAI path
            ]
            emotion_map_file = None
            emotion_ref_dir_found = None
            
            # Try to find emotion_map.json in any of the possible directories
            for emotion_ref_dir in emotion_ref_dirs:
                potential_file = emotion_ref_dir / "emotion_map.json"
                if potential_file.exists():
                    emotion_map_file = potential_file
                    emotion_ref_dir_found = emotion_ref_dir
                    break
            
            emotion_map = {}
            if emotion_map_file:
                try:
                    with open(emotion_map_file, 'r') as f:
                        emotion_map = json.load(f)
                except Exception:
                    pass
            
            for t in self._memory_triples[-3:]:
                # User turn
                user_turn = {
                    'text': t['q'],
                    'speaker_id': 1
                }
                
                # Add audio reference if available for this emotion
                user_emotion = t.get('emotion', 'neutral')
                if user_emotion in emotion_map and emotion_map[user_emotion] and emotion_ref_dir_found:
                    ref_file = emotion_ref_dir_found / emotion_map[user_emotion][0]['file']
                    if ref_file.exists():
                        try:
                            import torchaudio
                            audio_tensor, sr = torchaudio.load(str(ref_file))
                            audio_np = audio_tensor.squeeze().numpy().astype(np.float32)
                            user_turn['audio'] = audio_np
                        except Exception:
                            pass
                
                ctx.append(user_turn)
                
                # Oviya turn
                oviya_turn = {
                    'text': t['r'],
                    'speaker_id': 42  # Oviya's consistent speaker ID
                }
                
                # Add audio reference for Oviya's emotion if available
                oviya_emotion = t.get('oviya_emotion', 'neutral')
                if oviya_emotion in emotion_map and emotion_map[oviya_emotion] and emotion_ref_dir_found:
                    ref_file = emotion_ref_dir_found / emotion_map[oviya_emotion][0]['file']
                    if ref_file.exists():
                        try:
                            import torchaudio
                            audio_tensor, sr = torchaudio.load(str(ref_file))
                            audio_np = audio_tensor.squeeze().numpy().astype(np.float32)
                            oviya_turn['audio'] = audio_np
                        except Exception:
                            pass
                
                ctx.append(oviya_turn)
            
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

    # Initialize session state tracking
    session_state = None
    if SESSION_MANAGER_AVAILABLE and user_id != "anonymous":
        try:
            session_state = session_mgr.get_or_create(user_id)
            print(f"📊 Session state initialized: {user_id}")
        except Exception as e:
            print(f"⚠️ Session management failed: {e}")
            session_state = None
    
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
                            # Removed: optimized_streamer.generate_voice (use csm_streaming instead)
                            # Use csm_streaming for voice generation
                            batch_id = await session.csm_streaming.submit_stream_request(
                                user_id=session.user_id,
                                text=test_text,
                                emotion=emotion,
                                speaker_id=42,
                                conversation_context=[],
                                user_audio=None,
                                reference_audio=None,
                                prosody_params=None
                            )
                            audio_bytes = b""  # Streamed via get_stream_results
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
                            # Removed: optimized_streamer.generate_batch_voice (use csm_streaming instead)
                            # Batch generation is handled by csm_streaming's batch processor
                            batch_results = []  # Use csm_streaming batch API

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

                    # Update session activity
                    # Removed: session_manager.py deleted (functionality merged into ConversationSession)
                    # If sentence boundary or token-bucket threshold, trigger early TTS
                    if any(p in partial for p in ['.', '!', '?']) and not session.is_generating:
                        # Bid detection for micro-ack (backchannel is injected in brain already)
                        bid = session.bids.detect_bid(partial, prosody={"energy": 0.05}, pause_ms=300)
                        # We could send a quick ack here if needed
                        session.is_generating = True
                        # 🆕 SPEECH-TO-SPEECH: Pass user audio if available
                        user_audio_for_response = session._current_user_audio if session._current_user_audio is not None else None
                        asyncio.create_task(session.generate_response_streaming(
                            websocket,
                            partial,
                            'neutral',
                            user_audio=user_audio_for_response
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
                            'neutral',
                            user_audio=eos_audio if eos_audio is not None else session._current_user_audio
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
                                'neutral',
                                user_audio=session._current_user_audio
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
                        combined['emotion'],
                        user_audio=session._current_user_audio
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

        # Cleanup session state
        # Removed: session_manager.py deleted (functionality merged into ConversationSession)


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

