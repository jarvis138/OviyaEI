"""
Oviya Voice Server - ChatGPT-Style WebRTC Implementation
Ultra-low latency (<300ms) voice mode using WebRTC + Vast.ai services

Architecture:
- WebRTC for P2P audio streaming (no buffering)
- Silero VAD for instant speech detection (<100ms)
- Remote WhisperX on Vast.ai for transcription
- Ollama on Vast.ai for LLM responses
- CSM on Vast.ai for emotional TTS
"""

import asyncio
import numpy as np
import torch
import requests
import base64
import io
import time
import re
from collections import deque
from typing import Optional, Dict, AsyncGenerator, List
import os
import torchaudio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
import os
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
try:
    from prometheus_client import Summary, CONTENT_TYPE_LATEST, generate_latest
    STT_LATENCY = Summary('oviya_rtc_stt_latency_seconds', 'STT processing time per turn')
    LLM_LATENCY = Summary('oviya_rtc_llm_latency_seconds', 'LLM processing time per turn')
    TTS_LATENCY = Summary('oviya_rtc_tts_latency_seconds', 'TTS generation time per turn')
    TIME_TO_FIRST_AUDIO = Summary('oviya_rtc_ttfb_seconds', 'Time to first audio chunk')
except Exception:
    STT_LATENCY = LLM_LATENCY = TTS_LATENCY = TIME_TO_FIRST_AUDIO = None

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer
)
try:
    from aiortc.rtcrtpsender import RTCRtpSender
    RTCRtpSender._RETRANSMISSION_MAX_DELAY = 0.25
except Exception:
    pass
from av import AudioFrame

# Import Oviya components
from .config.service_urls import OLLAMA_URL, CSM_URL, WHISPERX_URL
from .brain.llm_brain import OviyaBrain
from .emotion_controller.controller import EmotionController
from .voice.csm_1b_client import CSM1BClient
from .voice.whisper_client import WhisperTurboClient
from .voice.csm_1b_stream import CSMRVQStreamer
from .voice.humanlike_prosody import HumanlikeProsodyEngine

app = FastAPI(title="Oviya WebRTC Voice Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SileroVAD:
    """
    ChatGPT-level Voice Activity Detection
    Detects speech start/end in <100ms for instant response
    """
    
    def __init__(self):
        print("🎤 Loading Silero VAD...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps, _, _, _, _) = utils
        
        self.sample_rate = 16000
        self.speech_threshold = 0.5
        self.silence_threshold = 0.35
        
        # ChatGPT-like timing
        self.min_speech_duration_ms = 250   # 250ms to detect speech start
        self.min_silence_duration_ms = 700  # 700ms to detect speech end
        
        self.speech_buffer = deque(maxlen=100)  # ~3 seconds at 30ms chunks
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        
        print("✅ VAD ready")
    
    def process_chunk(self, audio_chunk: np.ndarray):
        """
        Process 30ms audio chunk (512 samples at 16kHz)
        Returns: (is_speech_active, end_of_speech, audio_to_transcribe)
        """
        # Silero VAD requires exactly 512 samples (32ms at 16kHz)
        # Pad or truncate if needed
        if len(audio_chunk) < 512:
            audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
        elif len(audio_chunk) > 512:
            audio_chunk = audio_chunk[:512]
        
        audio_tensor = torch.FloatTensor(audio_chunk)
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        self.speech_buffer.append(audio_chunk)
        
        # Detect speech start
        if speech_prob > self.speech_threshold:
            self.speech_frames += 1
            self.silence_frames = 0
            
            # Speech started (250ms of consecutive speech)
            if not self.is_speaking and self.speech_frames >= 8:  # 8 * 30ms ≈ 240ms
                self.is_speaking = True
                print("🗣️  Speech detected")
                return True, False, None
        
        # Detect silence
        else:
            if self.is_speaking:
                self.silence_frames += 1
                
                # End of speech detected (700ms of silence)
                if self.silence_frames >= 23:  # 23 * 30ms ≈ 690ms
                    self.is_speaking = False
                    self.speech_frames = 0
                    self.silence_frames = 0
                    
                    # Return all buffered audio for transcription
                    audio_to_process = np.concatenate(list(self.speech_buffer))
                    self.speech_buffer.clear()
                    
                    duration = len(audio_to_process) / self.sample_rate
                    print(f"🔇 Speech ended ({duration:.1f}s)")
                    return False, True, audio_to_process
            else:
                self.speech_frames = 0
        
        return self.is_speaking, False, None


class RemoteWhisperXClient:
    """Client for WhisperX on Vast.ai (via Cloudflare tunnel)"""
    
    def __init__(self):
        self.transcribe_url = f"{WHISPERX_URL}/transcribe"
        print(f"🎤 WhisperX: {self.transcribe_url}")
    
    async def transcribe(self, audio: np.ndarray) -> Dict:
        """Fast transcription using remote WhisperX"""
        try:
            # Convert to base64
            audio_base64 = base64.b64encode(audio.tobytes()).decode('utf-8')
            
            # Call remote WhisperX
            response = requests.post(
                self.transcribe_url,
                json={
                    'audio': audio_base64,
                    'batch_size': 8,
                    'language': 'en'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get('text', ''),
                    "words": result.get('word_timestamps', []),
                    "processing_time": result.get('processing_time', 0)
                }
            else:
                print(f"⚠️  WhisperX error: {response.status_code}")
                return {"text": "", "words": [], "processing_time": 0}
                
        except Exception as e:
            print(f"❌ WhisperX error: {e}")
            return {"text": "", "words": [], "processing_time": 0}


class RemoteCSMClient:
    """
    WebRTC-optimized CSM-1B Client
    Uses proper RVQ/Mimi pipeline for ChatGPT-level quality
    """
    
    def __init__(self):
        self.csm_url = CSM_URL
        print(f"🎵 Initializing CSM-1B for WebRTC...")

        # Prefer local RVQ/Mimi streamer when explicitly enabled
        self.use_local_streamer = os.getenv("OVIYA_USE_LOCAL_CSM_STREAMER", "0") == "1"

        if self.use_local_streamer:
            print("   🎛️ Using local RVQ/Mimi streamer (CSMRVQStreamer)")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 2 RVQ frames flush (~160ms) for low latency; can tune to 2-4
            self.streamer = CSMRVQStreamer(device=device, flush_rvq_frames=2)
            self.csm_client = None
        else:
            print("   🌐 Using remote CSM API")
            # Initialize remote CSM-1B client
            self.csm_client = CSM1BClient(
                use_local_model=False,
                remote_url=self.csm_url
            )
            self.streamer = None
        
        print(f"   ✅ CSM-1B ready")
    
    async def generate_streaming(
        self,
        text: str,
        emotion: str = "calm",
        conversation_context: Optional[List[Dict]] = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Generate TTS audio with CSM-1B streaming
        
        Now using proper RVQ/Mimi pipeline:
        - RVQ tokens generated incrementally
        - Mimi decoder for high-quality audio
        - Conversational context conditioning
        - Prosody/emotion control
        
        Yields int16 PCM audio chunks at 16kHz (resampled from 24kHz)
        """
        print(f"🎵 Generating with CSM-1B RVQ/Mimi pipeline...")

        if self.use_local_streamer and self.streamer is not None:
            # Local RVQ/Mimi streaming
            async for audio_chunk_24k in self.streamer.generate_streaming(
                text=text,
                emotion=emotion,
                conversation_context=conversation_context
            ):
                # Resample to 16kHz for WebRTC and convert to int16
                audio_16k = self._resample_24_to_16(audio_chunk_24k)
                audio_int16 = (audio_16k * 32767).astype(np.int16)
                yield audio_int16
        else:
            # Remote CSM streaming (already decodes internally)
            async for audio_chunk_24k in self.csm_client.generate_streaming(
                text=text,
                emotion=emotion,
                speaker_id=0,
                conversation_context=conversation_context
            ):
                audio_16k = self._resample_24_to_16(audio_chunk_24k)
                audio_int16 = (audio_16k * 32767).astype(np.int16)
                yield audio_int16
    
    def _resample_24_to_16(self, audio_24k: np.ndarray) -> np.ndarray:
        """
        Resample 24kHz -> 16kHz (CSM output -> WebRTC)
        
        Uses scipy for high-quality resampling
        """
        try:
            from scipy import signal
            # High-quality resampling (3:2 ratio)
            num_samples_16k = int(len(audio_24k) * 16 / 24)
            return signal.resample(audio_24k, num_samples_16k)
        except ImportError:
            # Fallback: simple decimation
            print("   ⚠️  scipy not available, using simple decimation")
            return audio_24k[::3][:len(audio_24k)*2//3]


class AudioStreamTrack(MediaStreamTrack):
    """
    Custom WebRTC audio track that sends Oviya's voice back to client
    This is the "magic" that makes it feel instant - no buffering!
    """
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.channels = 1
        self.audio_queue = asyncio.Queue()
        self._timestamp = 0
        self.samples_per_frame = 480  # 30ms of audio at 16kHz
        self._stopped = False
        # Fade-out state
        self._fade_active = False
        self._fade_duration_samples = 0
        self._fade_progress_samples = 0
        
    async def recv(self):
        """
        Called by WebRTC to get next audio frame
        This runs in real-time, so must be FAST
        """
        if self._stopped:
            raise Exception("Track stopped")
        
        # Get audio from queue (generated by TTS)
        audio_samples = await self.audio_queue.get()
        
        if audio_samples is None:  # Stop signal
            self._stopped = True
            raise Exception("Track ended")
        
        # Convert to AudioFrame
        frame = AudioFrame(
            format='s16',
            layout='mono',
            samples=len(audio_samples)
        )
        
        # Fill frame with audio data
        frame.planes[0].update(audio_samples.tobytes())
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = 1 / self.sample_rate
        
        self._timestamp += len(audio_samples)
        
        return frame
    
    async def send_audio(self, audio: np.ndarray):
        """
        Add audio to streaming queue (called by TTS)
        Audio should be 16-bit PCM at 16kHz
        """
        # Split into 30ms chunks for smooth streaming
        chunk_size = self.samples_per_frame
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]

            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Apply fade-out ramp if active
            if self._fade_active and self._fade_duration_samples > 0:
                # Compute per-sample gain ramp (linear to ~0.001)
                s0 = self._fade_progress_samples
                indices = np.arange(len(chunk)) + s0
                frac = np.clip(indices.astype(np.float32) / float(self._fade_duration_samples), 0.0, 1.0)
                gains = 1.0 + (0.001 - 1.0) * frac  # from 1.0 down to 0.001

                # Ensure chunk is float for scaling
                chunk_float = chunk.astype(np.float32)
                chunk_scaled = chunk_float * gains
                chunk = np.clip(chunk_scaled, -32768.0, 32767.0).astype(np.int16)

                self._fade_progress_samples += len(chunk)
                if self._fade_progress_samples >= self._fade_duration_samples:
                    self._fade_active = False
                    self._fade_duration_samples = 0
                    self._fade_progress_samples = 0

            await self.audio_queue.put(chunk)

    def clear_queue(self):
        """Clear any buffered audio frames to stop playback quickly"""
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except Exception:
            pass

    def start_fade_out(self, duration_ms: int = 500):
        """Start fading out audio over the given duration (milliseconds)"""
        self._fade_duration_samples = int(self.sample_rate * (duration_ms / 1000.0))
        if self._fade_duration_samples <= 0:
            self._fade_active = False
            self._fade_duration_samples = 0
            self._fade_progress_samples = 0
            return
        self._fade_active = True
        self._fade_progress_samples = 0
    
    def stop(self):
        """Stop the track"""
        self._stopped = True
        try:
            self.audio_queue.put_nowait(None)
        except:
            pass


class OviyaVoiceConnection:
    """
    Main connection handler - manages the WebRTC peer connection
    With conversational context for CSM-1B
    """
    
    def __init__(self, pc: RTCPeerConnection):
        self.pc = pc
        self.vad = SileroVAD()
        self.whisperx = RemoteWhisperXClient()
        # Local Whisper Turbo for low-latency STT
        try:
            self.whisper_turbo = WhisperTurboClient()
        except Exception:
            self.whisper_turbo = None
        self.csm = RemoteCSMClient()
        self.brain = OviyaBrain(ollama_url=OLLAMA_URL)
        self.emotion_controller = EmotionController()
        self.humanlike = HumanlikeProsodyEngine(enable_fillers=True)
        
        # Audio streaming
        self.audio_output_track = AudioStreamTrack()
        self.is_processing = False
        self.interrupt_requested = False
        self.is_closed = False  # Track if connection is closed
        
        # Conversation history for CSM-1B context
        self.conversation_history = []
        
        # Add audio track to peer connection
        self.pc.addTrack(self.audio_output_track)
        
        print("✅ Oviya voice connection initialized")
    
    async def close(self):
        """Clean up connection resources"""
        print("🔌 Closing Oviya connection...")
        self.is_closed = True
        self.interrupt_requested = True  # Stop any ongoing generation
        self.is_processing = False
    
    async def process_audio_frame(self, frame: AudioFrame):
        """
        Process incoming audio frame from user's microphone
        This is called for EVERY audio frame (every 20-30ms)
        """
        # Don't process if connection is closed
        if self.is_closed:
            return
        
        # Convert frame to numpy array (float32, normalized)
        audio_data = np.frombuffer(
            frame.planes[0].to_bytes(),
            dtype=np.int16
        ).astype(np.float32) / 32768.0

        # Basic preprocessing: high-pass filter + light noise reduction + RMS normalization
        try:
            # High-pass at ~80 Hz to remove rumble
            b, a = torchaudio.functional.highpass_biquad(torch.tensor(audio_data), self.vad.sample_rate, 80)
            audio_data = b.numpy() if hasattr(b, 'numpy') else audio_data
        except Exception:
            pass
        
        # Very light noise reduction using spectral gating on short window
        try:
            # Convert to torch tensor for ops
            audio_tensor = torch.tensor(audio_data)
            # Short STFT
            stft = torch.stft(audio_tensor, n_fft=256, hop_length=128, win_length=256, return_complex=True)
            mag = stft.abs()
            # Estimate noise floor from 10th percentile
            noise_floor = torch.quantile(mag, 0.10)
            # Soft gate
            gated_mag = torch.clamp(mag - 0.6 * noise_floor, min=0.0)
            # Preserve phase
            phase = torch.angle(stft)
            stft_denoised = gated_mag * torch.exp(1j * phase)
            # iSTFT
            denoised = torch.istft(stft_denoised, n_fft=256, hop_length=128, win_length=256, length=audio_tensor.shape[0])
            audio_data = denoised.numpy()
        except Exception:
            # If anything fails, continue with original
            pass
        
        # RMS normalization to target -20 dBFS equivalent (~0.1 RMS)
        try:
            rms = float(np.sqrt(np.mean(np.square(audio_data)) + 1e-8))
            target_rms = 0.1
            if rms > 0:
                gain = np.clip(target_rms / rms, 0.25, 4.0)
                audio_data = np.clip(audio_data * gain, -1.0, 1.0)
        except Exception:
            pass
        
        # Run VAD
        is_speech, end_of_speech, audio_to_process = self.vad.process_chunk(audio_data)
        
        # User started speaking - interrupt Oviya if needed
        if is_speech and self.is_processing:
            self.interrupt_requested = True
            print("🚫 User interrupted Oviya")
            # Begin server-side fade-out and purge queue to stop quickly
            try:
                self.audio_output_track.start_fade_out(duration_ms=500)
                self.audio_output_track.clear_queue()
            except Exception as _e:
                pass
        
        # User stopped speaking - transcribe and respond
        if end_of_speech and audio_to_process is not None and len(audio_to_process) > 0:
            print(f"🎤 Processing {len(audio_to_process)/16000:.2f}s of audio")
            asyncio.create_task(self.handle_user_utterance(audio_to_process))
    
    async def handle_user_utterance(self, audio: np.ndarray):
        """
        Complete turn: transcribe → think → speak
        """
        if self.is_closed:
            print("⚠️  Connection closed, skipping...")
            return
        
        if self.is_processing:
            print("⚠️  Already processing, skipping...")
            return
        
        self.is_processing = True
        start_time = time.time()
        
        try:
            # 1. Transcribe with WhisperX (~200-400ms)
            print("🔍 Transcribing...")
            transcription = {"text": "", "words": [], "processing_time": 0}
            # Prefer local Whisper Turbo if available
            if self.whisper_turbo is not None:
                try:
                    # audio is float32 0..1 at 16kHz
                    stt_t0 = time.time()
                    result = await self.whisper_turbo.transcribe_audio(audio.astype(np.float32))
                    if STT_LATENCY:
                        STT_LATENCY.observe(time.time() - stt_t0)
                    transcription["text"] = result.get("text", "")
                except Exception:
                    pass
            # Fallback to remote WhisperX
            if not transcription["text"]:
                stt_t0 = time.time()
                transcription = await self.whisperx.transcribe(audio)
                if STT_LATENCY:
                    STT_LATENCY.observe(time.time() - stt_t0)
            text = transcription["text"]
            
            if not text:
                print("⚠️  Empty transcription")
                return
            
            print(f"💬 User: {text}")
            stt_time = time.time() - start_time
            
            # Add user message to conversation history
            self.conversation_history.append({
                "text": text,
                "speaker_id": 1,  # User
                "timestamp": time.time()
            })
            
            # 2. Get Oviya's response (~300-800ms)
            print("🧠 Thinking (streaming)...")
            token_bucket = 0
            assembled = []
            resp_start = time.time()
            response_text = ""
            try:
                async for token in self.brain.think_streaming(text, conversation_history=self.conversation_history):
                    assembled.append(token)
                    token_bucket += 1
                    if not response_text and (any(p in ''.join(assembled) for p in ['.', '!', '?']) or token_bucket >= 20):
                        response_text = ''.join(assembled)
                        break
            except Exception:
                pass
            if not response_text:
                brain_out = self.brain.think(text, conversation_history=self.conversation_history)
                response_text = brain_out.get("prosodic_text") or brain_out.get("text") or ""
            else:
                # Ensure downstream has a valid structure even when we streamed tokens
                brain_out = {"emotion": "calm", "intensity": 0.7}
            if LLM_LATENCY:
                LLM_LATENCY.observe(time.time() - resp_start)
            print(f"💭 Oviya: {response_text}")
            llm_time = time.time() - start_time - stt_time
            
            # Detect emotion from brain output
            emotion = brain_out.get("emotion", "calm")
            
            # Add Oviya response to conversation history
            self.conversation_history.append({
                "text": response_text,
                "speaker_id": 0,  # Oviya
                "timestamp": time.time()
            })
            
            # Keep only last 10 turns (5 exchanges)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # 3. Prosody timing plan and TTS streaming
            print("🎵 Speaking...")
            print(f"   Context: {len(self.conversation_history)} turns")
            tts_start = time.time()
            ttfb_sent = False
            # Apply humanlike pre-tts delay
            try:
                adj_text, timing = self.humanlike.enhance(response_text, emotion, ctx={})
            except Exception:
                adj_text, timing = response_text, {"pre_tts_delay_ms": 300, "insert_breath": False}
            try:
                await asyncio.sleep(max(0.0, float(timing.get("pre_tts_delay_ms", 300)) / 1000.0))
            except Exception:
                pass
            
            # Personality conditioning vector (if computed by brain)
            try:
                style_vec = getattr(self.brain, "_last_personality_vector", None)
            except Exception:
                style_vec = None

            async for audio_chunk in self.csm.generate_streaming(
                text=adj_text,
                emotion=emotion,
                conversation_context=self.conversation_history[:-1],  # Exclude current response
                style_vec=style_vec
            ):
                # Stop if interrupted or connection closed
                if self.interrupt_requested or self.is_closed:
                    print("🚫 Speech interrupted")
                    # Send a final faded frame to avoid abrupt cut
                    try:
                        self.audio_output_track.start_fade_out(duration_ms=300)
                    except Exception:
                        pass
                    break
                
                # Stream audio to WebRTC immediately
                await self.audio_output_track.send_audio(audio_chunk)
                if not ttfb_sent and TIME_TO_FIRST_AUDIO:
                    TIME_TO_FIRST_AUDIO.observe(time.time() - tts_start)
                    ttfb_sent = True
            
            tts_time = time.time() - tts_start
            total_time = time.time() - start_time
            
            print(f"⚡ Latency: STT={stt_time:.2f}s, LLM={llm_time:.2f}s, TTS={tts_time:.2f}s, Total={total_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error handling utterance: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
            self.interrupt_requested = False


# Global connections storage
peer_connections = set()


@app.post("/api/voice/offer")
async def handle_offer(request: Request):
    """
    WebRTC signaling endpoint - handles peer connection setup
    Client sends SDP offer, server responds with SDP answer
    """
    try:
        params = await request.json()
        # Optional JWT auth
        token = params.get("token", "")
        if not verify_jwt(token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        print("📡 Received WebRTC offer")
        
        # Create peer connection with configurable ICE servers (STUN/TURN)
        ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        turn_url = os.getenv("TURN_URL")  # e.g., turn:YOUR_IP:3478?transport=tcp
        turn_user = os.getenv("TURN_USER")
        turn_pass = os.getenv("TURN_PASS")
        # Optional time-limited HMAC auth
        turn_secret = os.getenv("TURN_SECRET")
        if turn_secret and not (turn_user and turn_pass):
            try:
                import base64, hmac, hashlib, time as _t
                expiry = int(_t.time()) + 3600
                user = f"{expiry}:oviya"
                key = base64.b64encode(hmac.new(turn_secret.encode(), user.encode(), hashlib.sha1).digest()).decode()
                turn_user, turn_pass = user, key
            except Exception:
                pass
        turns_url = os.getenv("TURNS_URL")  # e.g., turns:YOUR_IP:5349?transport=tcp
        if turn_url and turn_user and turn_pass:
            ice_servers.append(RTCIceServer(urls=[turn_url], username=turn_user, credential=turn_pass))
        if turns_url and turn_user and turn_pass:
            ice_servers.append(RTCIceServer(urls=[turns_url], username=turn_user, credential=turn_pass))

        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        peer_connections.add(pc)
        
        # Create connection handler
        connection = OviyaVoiceConnection(pc)
        
        @pc.on("track")
        async def on_track(track):
            """
            Called when client starts sending audio
            """
            print(f"📡 Receiving {track.kind} track")
            
            if track.kind == "audio":
                # Process audio frames in real-time
                while True:
                    try:
                        frame = await track.recv()
                        await connection.process_audio_frame(frame)
                    except Exception as e:
                        print(f"Track ended: {e}")
                        break
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                # Clean up Oviya connection resources
                await connection.close()
                await pc.close()
                peer_connections.discard(pc)
                print("🔌 Connection cleaned up")
        
        # Set remote description
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        print("✅ WebRTC connection established")
        
        return JSONResponse({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
        
    except Exception as e:
        print(f"❌ Error handling offer: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
async def root():
    """Serve the WebRTC client HTML"""
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oviya Voice - WebRTC Mode</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            border-radius: 24px;
            padding: 48px;
            box-shadow: 0 24px 72px rgba(0,0,0,0.3);
            text-align: center;
            max-width: 600px;
            width: 90%;
        }
        h1 {
            margin: 0 0 12px;
            color: #333;
            font-size: 32px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 32px;
            font-size: 16px;
        }
        .voice-button {
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            font-size: 64px;
            transition: all 0.3s ease;
            margin: 24px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        .voice-button.inactive {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .voice-button.inactive:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
        }
        .voice-button.active {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4); }
            50% { transform: scale(1.08); box-shadow: 0 12px 36px rgba(239, 68, 68, 0.6); }
        }
        .status {
            margin: 24px 0;
            font-size: 18px;
            color: #666;
            min-height: 32px;
            font-weight: 500;
        }
        .visualizer {
            width: 100%;
            height: 80px;
            background: #f8f9fa;
            border-radius: 12px;
            margin: 24px 0;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: flex-end;
            justify-content: space-evenly;
            padding: 8px;
        }
        .bar {
            width: 4px;
            background: linear-gradient(to top, #667eea, #764ba2);
            border-radius: 2px;
            transition: height 0.1s ease;
            height: 0;
        }
        .transcript {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            text-align: left;
            min-height: 80px;
        }
        .transcript-label {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .transcript-text {
            color: #333;
            line-height: 1.6;
            font-size: 16px;
        }
        .latency {
            font-size: 13px;
            color: #999;
            margin-top: 16px;
            font-family: 'Monaco', 'Courier New', monospace;
        }
        .tech-info {
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid #e5e7eb;
            font-size: 12px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Oviya Voice</h1>
        <p class="subtitle">ChatGPT-style WebRTC Mode</p>
        
        <button id="voiceButton" class="voice-button inactive">🎤</button>
        
        <div class="status" id="status">Click microphone to start</div>
        
        <div class="visualizer" id="visualizer"></div>
        
        <div class="transcript">
            <div class="transcript-label">You said:</div>
            <div class="transcript-text" id="userTranscript">...</div>
        </div>
        
        <div class="transcript">
            <div class="transcript-label">Oviya:</div>
            <div class="transcript-text" id="oviyaResponse">...</div>
        </div>
        
        <div class="latency" id="latency"></div>
        
        <div class="tech-info">
            <strong>WebRTC P2P</strong> • Silero VAD • WhisperX • Ollama • CSM TTS<br>
            Running on Vast.ai RTX 5880 Ada via Cloudflare Tunnels
        </div>
    </div>

    <script>
        class OviyaVoiceWebRTC {
            constructor() {
                this.pc = null;
                this.localStream = null;
                this.isActive = false;
                
                this.voiceButton = document.getElementById('voiceButton');
                this.statusEl = document.getElementById('status');
                this.userTranscriptEl = document.getElementById('userTranscript');
                this.oviyaResponseEl = document.getElementById('oviyaResponse');
                this.latencyEl = document.getElementById('latency');
                this.visualizerEl = document.getElementById('visualizer');
                
                this.audioContext = null;
                this.analyser = null;
                this.visualizerBars = [];
                this.workletNode = null;
                this.level = 0;
                
                this.setupUI();
                this.createVisualizer();
            }
            
            setupUI() {
                this.voiceButton.addEventListener('click', () => {
                    if (this.isActive) {
                        this.stop();
                    } else {
                        this.start();
                    }
                });
            }
            
            createVisualizer() {
                for (let i = 0; i < 48; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'bar';
                    this.visualizerEl.appendChild(bar);
                    this.visualizerBars.push(bar);
                }
            }
            
            updateVisualizer() {
                if (!this.analyser) return;
                
                // Use worklet-computed level if available
                const base = Math.max(0, Math.min(1, this.level));
                for (let i = 0; i < this.visualizerBars.length; i++) {
                    const factor = (i + 1) / this.visualizerBars.length;
                    const height = 100 * base * Math.pow(factor, 0.5);
                    this.visualizerBars[i].style.height = `${height}%`;
                }
                
                if (this.isActive) {
                    requestAnimationFrame(() => this.updateVisualizer());
                }
            }
            
            async start() {
                try {
                    this.statusEl.textContent = '🔄 Connecting...';
                    
                    this.localStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            channelCount: 1,
                            sampleRate: 16000,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });
                    
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const source = this.audioContext.createMediaStreamSource(this.localStream);
                    // Load AudioWorklet for ring buffer + level metering
                    try {
                        await this.audioContext.audioWorklet.addModule('/worklet.js');
                        this.workletNode = new AudioWorkletNode(this.audioContext, 'oviya-meter');
                        source.connect(this.workletNode);
                        this.workletNode.port.onmessage = (e) => {
                            if (e.data && typeof e.data.level === 'number') {
                                this.level = e.data.level;
                            }
                        };
                    } catch (e) {
                        // Fallback to basic analyser if worklet fails
                        this.analyser = this.audioContext.createAnalyser();
                        this.analyser.fftSize = 256;
                        source.connect(this.analyser);
                    }
                    this.updateVisualizer();
                    
                    this.pc = new RTCPeerConnection({ iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' }
                    ]});
                    
                    this.localStream.getTracks().forEach(track => {
                        this.pc.addTrack(track, this.localStream);
                    });
                    
                    this.pc.ontrack = (event) => {
                        console.log('📡 Receiving audio from Oviya');
                        const audio = new Audio();
                        audio.srcObject = event.streams[0];
                        audio.play().catch(e => console.error('Audio play error:', e));
                        // Adaptive playbackRate nudge (±2%) to hide jitter
                        try {
                            setInterval(() => {
                                audio.playbackRate = Math.max(0.98, Math.min(1.02, audio.playbackRate));
                            }, 1000);
                        } catch (e) {}
                    };
                    
                    const offer = await this.pc.createOffer();
                    await this.pc.setLocalDescription(offer);
                    
                    const response = await fetch('/api/voice/offer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            sdp: this.pc.localDescription.sdp,
                            type: this.pc.localDescription.type
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`Server error: ${response.status}`);
                    }
                    
                    const answer = await response.json();
                    await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
                    
                    this.isActive = true;
                    this.voiceButton.classList.remove('inactive');
                    this.voiceButton.classList.add('active');
                    this.voiceButton.textContent = '🔴';
                    this.statusEl.textContent = '🎤 Listening... (speak naturally)';
                    
                    console.log('✅ WebRTC connection established');
                    
                } catch (error) {
                    console.error('Error starting voice:', error);
                    this.statusEl.textContent = `❌ Error: ${error.message}`;
                    this.stop();
                }
            }
            
            stop() {
                if (this.localStream) {
                    this.localStream.getTracks().forEach(track => track.stop());
                }
                if (this.pc) {
                    this.pc.close();
                }
                if (this.audioContext) {
                    this.audioContext.close();
                }
                
                this.isActive = false;
                this.voiceButton.classList.remove('active');
                this.voiceButton.classList.add('inactive');
                this.voiceButton.textContent = '🎤';
                this.statusEl.textContent = 'Disconnected. Click to talk again.';
                
                this.visualizerBars.forEach(bar => bar.style.height = '0');
                
                console.log('🔌 Disconnected');
            }
        }
        
        const voiceClient = new OviyaVoiceWebRTC();
    </script>
</body>
</html>
    """)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Oviya WebRTC Voice Server",
        "endpoints": {
            "whisperx": WHISPERX_URL,
            "ollama": OLLAMA_URL,
            "csm": CSM_URL
        }
    }


@app.get("/worklet.js")
async def worklet_js():
    js = """
class OviyaMeterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._rms = 0;
    this._alpha = 0.2; // smoothing
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const chan = input[0];
      let sum = 0.0;
      for (let i = 0; i < chan.length; i++) {
        sum += chan[i] * chan[i];
      }
      const rms = Math.sqrt(sum / chan.length);
      this._rms = this._alpha * rms + (1 - this._alpha) * this._rms;
      // post a soft-clipped level for UI
      const level = Math.tanh(this._rms * 3.0);
      this.port.postMessage({ level });
    }
    return true;
  }
}
registerProcessor('oviya-meter', OviyaMeterProcessor);
"""
    return Response(content=js, media_type="application/javascript")

@app.get('/metrics')
async def metrics():
    try:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        return Response("", media_type='text/plain')


@app.on_event("shutdown")
async def on_shutdown():
    """Clean up connections on shutdown"""
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros)
    peer_connections.clear()
    print("🛑 Server shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🎤 OVIYA VOICE SERVER - ChatGPT Style (WebRTC)")
    print("=" * 70)
    print(f"WhisperX: {WHISPERX_URL}")
    print(f"Ollama:   {OLLAMA_URL}")
    print(f"CSM TTS:  {CSM_URL}")
    print("=" * 70)
    print("Server:   http://localhost:8000")
    print("WebRTC:   POST /api/voice/offer")
    print("Client:   http://localhost:8000/")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

