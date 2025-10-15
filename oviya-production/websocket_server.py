"""
WebSocket Server for Real-Time Oviya Conversations
Provides streaming audio input/output via WebSocket for web clients
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
import numpy as np
import json
import base64
from typing import Dict, Optional, List
import torch
from pathlib import Path

# Import Oviya components
from voice.realtime_input_remote import RealTimeVoiceInput  # Using remote WhisperX API on Vast.ai
from emotion_detector.detector import EmotionDetector
from brain.llm_brain import OviyaBrain
from emotion_controller.controller import EmotionController
from voice.openvoice_tts import HybridVoiceEngine
from voice.csm_1b_client import CSM1BClient
from voice.csm_1b_client import CSM1BClient
from voice.acoustic_emotion_detector import AcousticEmotionDetector
from brain.personality_store import PersonalityStore
from config.service_urls import OLLAMA_URL, CSM_URL
from faster_whisper import WhisperModel
import time

app = FastAPI(title="Oviya WebSocket Server")

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
        self.model = WhisperModel(model_size, device="auto", compute_type="int8")
        self.sample_rate = 16000
        self.buffer = bytearray()
        self.last_emit_time = 0.0
        self.emit_interval = 0.25  # seconds between partial emissions
        self.last_partial = ""
        self.min_window_s = 1.0
        self.max_window_s = 3.0

    def add_audio(self, audio_bytes: bytes):
        self.buffer.extend(audio_bytes)

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
        self.csm_streaming = CSM1BClient(use_local_model=False, remote_url=CSM_URL)
        self.stt = StreamingSTT()
        self.is_generating = False
        
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
        Stream TTS audio chunks as they are generated.
        Sends 'audio_chunk' messages and a terminal 'response' with text/emotion.
        """
        await self.cancel_tts_stream()

        # Stream LLM tokens while starting TTS ASAP
        prosodic_text = ""
        emotion = user_emotion or "neutral"
        assembled = []
        async for token in self.brain.think_streaming(user_text, user_emotion, conversation_history=None):
            assembled.append(token)
            # Kick off when first sentence boundary appears
            if not prosodic_text and any(p in ''.join(assembled) for p in ['.', '!', '?']):
                prosodic_text = ''.join(assembled)
                break
        if not prosodic_text:
            # Fallback to full think if no early boundary
            brain_resp_full = self.brain.think(user_text, user_emotion)
            prosodic_text = brain_resp_full.get('prosodic_text') or brain_resp_full.get('text', '')
            emotion = brain_resp_full.get('emotion', emotion)
        oviya_emotion_params = self.emotion_controller.map_emotion(
            emotion, brain_resp.get('intensity', 0.7)
        )

        # Store memory triple
        self._add_memory_triple(user_text, prosodic_text)

        self._tts_cancel_requested = False

        async def _stream():
            try:
                max_samples = int(24000 * 1.5)  # 1.5s chunk
                buf, n = [], 0
                async for c24 in self.csm_streaming.generate_streaming(
                    text=prosodic_text,
                    emotion=oviya_emotion_params.get('style_token', emotion),
                    speaker_id=0,
                    conversation_context=self._format_context_for_tts(),
                    reference_audio=self._reference_audio
                ):
                    if self._tts_cancel_requested:
                        break
                    buf.append(c24)
                    n += len(c24)
                    if n >= max_samples:
                        arr = np.concatenate(buf)
                        await websocket.send_json({
                            'type': 'audio_chunk',
                            'format': 'pcm_s16le',
                            'sample_rate': 24000,
                            'audio_base64': base64.b64encode((arr * 32767).astype(np.int16).tobytes()).decode('utf-8')
                        })
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
        ctx = []
        for t in self._memory_triples[-3:]:
            ctx.append({'text': t['q'], 'speaker_id': 1})
            ctx.append({'text': t['r'], 'speaker_id': 0})
        return ctx
    
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
            let ws = null;
            let mediaRecorder = null;
            let audioContext = null;
            let isRecording = false;
            let ttsAudioCtx = null;
            let ttsGain = null;
            let ttsQueue = [];
            let ttsPlaying = false;
            let userSpeaking = false;
            
            document.getElementById('connectBtn').onclick = connect;
            document.getElementById('recordBtn').onclick = toggleRecording;
            
            function connect() {
                ws = new WebSocket('ws://localhost:8000/ws/conversation?user_id=test_user');
                
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
                }
            }

            function enqueueAndPlayChunk(data) {
                initTTSPlayback();
                const bytes = Uint8Array.from(atob(data.audio_base64), c => c.charCodeAt(0));
                const int16 = new Int16Array(bytes.buffer);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;
                const buf = ttsAudioCtx.createBuffer(1, float32.length, 24000);
                buf.getChannelData(0).set(float32);
                ttsQueue.push(buf);
                if (!ttsPlaying) { ttsPlaying = true; playQueue(); }
            }

            function playQueue() {
                if (!ttsQueue.length) { ttsPlaying = false; return; }
                const buf = ttsQueue.shift();
                const src = ttsAudioCtx.createBufferSource();
                src.buffer = buf;
                src.connect(ttsGain);
                src.onended = () => playQueue();
                // Fade-in for smoothness
                ttsGain.gain.cancelScheduledValues(ttsAudioCtx.currentTime);
                ttsGain.gain.setValueAtTime(0.001, ttsAudioCtx.currentTime);
                ttsGain.gain.exponentialRampToValueAtTime(1.0, ttsAudioCtx.currentTime + 0.05);
                src.start();
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
    await websocket.accept()
    
    print(f"🔌 WebSocket connected: {user_id}")
    
    # Create conversation session
    session = ConversationSession(user_id)
    
    try:
        while True:
            # Receive data (can be bytes or text)
            message = await websocket.receive()
            
            # Check message type
            if 'text' in message:
                # Handle JSON messages (greeting, etc.)
                try:
                    data = json.loads(message['text'])
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
                    # If sentence boundary, start early generation (pipeline-parallel)
                    if any(p in partial for p in ['.', '!', '?']) and not session.is_generating:
                        session.is_generating = True
                        asyncio.create_task(session.generate_response_streaming(
                            websocket,
                            partial,
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Oviya WebSocket Server",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting Oviya WebSocket Server")
    print("=" * 60)
    print(f"   Ollama URL: {OLLAMA_URL}")
    print(f"   CSM URL: {CSM_URL}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

