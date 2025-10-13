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
from typing import Dict, Optional
import torch
from pathlib import Path

# Import Oviya components
from voice.realtime_input import RealTimeVoiceInput
from emotion_detector.detector import EmotionDetector
from brain.llm_brain import OviyaBrain
from emotion_controller.controller import EmotionController
from voice.openvoice_tts import HybridVoiceEngine
from voice.acoustic_emotion_detector import AcousticEmotionDetector
from brain.personality_store import PersonalityStore
from config.service_urls import OLLAMA_URL, CSM_URL

app = FastAPI(title="Oviya WebSocket Server")

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (in production, use dependency injection)
personality_store = PersonalityStore()


class ConversationSession:
    """Manages a single conversation session"""
    
    def __init__(self, user_id: str, enable_diarization: bool = False):
        self.user_id = user_id
        self.voice_input = RealTimeVoiceInput(enable_diarization=enable_diarization)
        self.emotion_detector = EmotionDetector()
        self.acoustic_emotion = AcousticEmotionDetector()
        self.brain = OviyaBrain(ollama_url=OLLAMA_URL)
        self.emotion_controller = EmotionController()
        self.tts = HybridVoiceEngine(csm_url=CSM_URL, default_engine="csm")
        
        # Load user personality
        self.personality = personality_store.load_personality(user_id)
        if self.personality:
            print(f"📚 Loaded personality for {user_id}")
            # Inject personality context into brain
            context = personality_store.get_conversation_summary(user_id, last_n=5)
            self.brain.context = context
        
        # Initialize models
        self.voice_input.initialize_models()
        
        print(f"✅ Conversation session initialized for {user_id}")
    
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
        
        # Add to voice input buffer
        self.voice_input.add_audio_chunk(audio_array)
        
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
        # Generate brain response
        response = self.brain.think(user_text, user_emotion)
        
        # Map to CSM emotion
        oviya_emotion = self.emotion_controller.map_to_csm_emotion(
            response['emotion'],
            response['intensity']
        )
        
        # Generate audio
        audio_result = self.tts.generate_emotional_speech(
            response['text'],
            oviya_emotion,
            reference_audio=None
        )
        
        # Convert audio to chunks for streaming
        audio_chunks = self._chunk_audio(audio_result['audio'])
        
        # Save conversation turn
        personality_store.add_conversation_turn(self.user_id, {
            'user_message': user_text,
            'oviya_response': response['text'],
            'user_emotion': user_emotion,
            'oviya_emotion': oviya_emotion
        })
        
        return {
            'text': response['text'],
            'emotion': oviya_emotion,
            'audio_chunks': audio_chunks,
            'duration': audio_result.get('duration', 0)
        }
    
    def _chunk_audio(self, audio: torch.Tensor, chunk_size: int = 4096) -> list:
        """
        Split audio into chunks for streaming
        
        Args:
            audio: Audio tensor
            chunk_size: Samples per chunk
            
        Returns:
            List of base64-encoded audio chunks
        """
        # Convert to int16 PCM
        audio_np = (audio.numpy() * 32767).astype(np.int16)
        
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
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    const audioData = e.inputBuffer.getChannelData(0);
                    const int16Data = new Int16Array(audioData.length);
                    for (let i = 0; i < audioData.length; i++) {
                        int16Data[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
                    }
                    ws.send(int16Data.buffer);
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
                    playAudio(data.audio_chunks);
                }
                
                transcript.scrollTop = transcript.scrollHeight;
            }
            
            function playAudio(chunks) {
                // Decode and play audio chunks
                const audioContext = new AudioContext({ sampleRate: 24000 });
                let offset = 0;
                
                chunks.forEach(chunk => {
                    const audioData = Uint8Array.from(atob(chunk), c => c.charCodeAt(0));
                    const int16Data = new Int16Array(audioData.buffer);
                    const float32Data = new Float32Array(int16Data.length);
                    
                    for (let i = 0; i < int16Data.length; i++) {
                        float32Data[i] = int16Data[i] / 32768.0;
                    }
                    
                    const audioBuffer = audioContext.createBuffer(1, float32Data.length, 24000);
                    audioBuffer.getChannelData(0).set(float32Data);
                    
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    source.start(audioContext.currentTime + offset);
                    
                    offset += audioBuffer.duration;
                });
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
        Client -> Server: Raw audio bytes (PCM, 16-bit, 16kHz, mono)
        Server -> Client: JSON messages
            - {'type': 'transcription', 'text': str, 'emotion': str}
            - {'type': 'response', 'text': str, 'emotion': str, 'audio_chunks': List[str]}
            - {'type': 'error', 'message': str}
    """
    await websocket.accept()
    
    print(f"🔌 WebSocket connected: {user_id}")
    
    # Create conversation session
    session = ConversationSession(user_id)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process audio chunk
            result = await session.process_audio_chunk(data)
            
            if result:
                # Send transcription to client
                await websocket.send_json({
                    'type': 'transcription',
                    'text': result['text'],
                    'speakers': result.get('speakers', ['user']),
                    'word_timestamps': result.get('word_timestamps', [])
                })
                
                # Detect emotion from audio + text
                acoustic_emotion = session.acoustic_emotion.detect_emotion(
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                )
                
                text_emotion = session.emotion_detector.detect_emotion(result['text'])
                
                # Combine emotions
                combined = session.acoustic_emotion.combine_with_text_emotion(
                    acoustic_emotion,
                    text_emotion,
                    acoustic_weight=0.6
                )
                
                # Generate response
                response = await session.generate_response(
                    result['text'],
                    combined['emotion']
                )
                
                # Send response to client
                await websocket.send_json({
                    'type': 'response',
                    'text': response['text'],
                    'emotion': response['emotion'],
                    'audio_chunks': response['audio_chunks'],
                    'duration': response['duration']
                })
    
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

