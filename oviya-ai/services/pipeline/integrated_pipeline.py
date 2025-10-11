#!/usr/bin/env python3
"""
Integrated Pipeline Orchestrator
User Mic → Silero VAD → Whisper ASR → Context Manager → Gemini LLM → CSM TTS → Audio Stream
"""
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp
import base64
import numpy as np
from collections import deque
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineState:
    """Represents the current pipeline state"""
    session_id: str
    user_id: str
    is_listening: bool
    is_processing: bool
    is_speaking: bool
    current_emotion: str
    audio_buffer: deque
    last_vad_time: float
    last_activity: float

class IntegratedPipeline:
    """Integrated pipeline orchestrator"""
    
    def __init__(self):
        # Service URLs
        self.vad_url = "http://localhost:8001"
        self.asr_url = "http://localhost:8002"
        self.context_url = "http://localhost:8003"
        self.csm_url = "http://157.157.221.29:8000"  # RunPod CSM
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Pipeline state
        self.active_sessions: Dict[str, PipelineState] = {}
        self.http_session = None
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.vad_threshold = 0.5
        self.silence_timeout = 2.0  # seconds
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.http_session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.http_session:
            await self.http_session.close()
    
    async def create_session(self, user_id: str) -> str:
        """Create a new pipeline session"""
        session_id = str(uuid.uuid4())
        
        # Create context session
        async with self.http_session.post(
            f"{self.context_url}/context/session/create",
            json={"user_id": user_id}
        ) as response:
            if response.status == 200:
                context_data = await response.json()
                context_session_id = context_data["session_id"]
            else:
                raise Exception("Failed to create context session")
        
        # Create pipeline state
        pipeline_state = PipelineState(
            session_id=session_id,
            user_id=user_id,
            is_listening=False,
            is_processing=False,
            is_speaking=False,
            current_emotion="empathetic",
            audio_buffer=deque(maxlen=int(self.sample_rate * 30)),  # 30 seconds
            last_vad_time=0,
            last_activity=time.time()
        )
        
        self.active_sessions[session_id] = pipeline_state
        
        logger.info(f"✅ Created pipeline session {session_id} for user {user_id}")
        return session_id
    
    async def process_audio_chunk(self, session_id: str, audio_data: bytes) -> Optional[Dict]:
        """Process audio chunk through the pipeline"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        state = self.active_sessions[session_id]
        state.last_activity = time.time()
        
        # Convert audio to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        state.audio_buffer.extend(audio_array)
        
        # Step 1: Voice Activity Detection
        vad_result = await self._detect_voice_activity(audio_data)
        
        if vad_result["is_speech"]:
            state.last_vad_time = time.time()
            state.is_listening = True
        
        # Step 2: Check for speech end (silence timeout)
        if (state.is_listening and 
            time.time() - state.last_vad_time > self.silence_timeout):
            
            # Process complete speech segment
            return await self._process_speech_segment(session_id)
        
        return None
    
    async def _detect_voice_activity(self, audio_data: bytes) -> Dict:
        """Detect voice activity using Silero VAD"""
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            async with self.http_session.post(
                f"{self.vad_url}/vad/detect",
                json={"audio_base64": audio_base64}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"is_speech": False, "confidence": 0.0}
        except Exception as e:
            logger.error(f"❌ VAD error: {e}")
            return {"is_speech": False, "confidence": 0.0}
    
    async def _process_speech_segment(self, session_id: str) -> Dict:
        """Process complete speech segment through the pipeline"""
        state = self.active_sessions[session_id]
        state.is_listening = False
        state.is_processing = True
        
        try:
            # Step 3: Speech Recognition (Whisper ASR)
            transcription = await self._transcribe_audio(session_id)
            
            if not transcription["text"]:
                state.is_processing = False
                return {"status": "no_speech", "text": ""}
            
            # Step 4: Get Context
            context = await self._get_relevant_context(session_id, transcription["text"])
            
            # Step 5: Generate Response (Gemini LLM)
            ai_response = await self._generate_response(
                session_id, transcription["text"], context, state.current_emotion
            )
            
            # Step 6: Generate Audio (CSM TTS)
            audio_response = await self._generate_audio(
                session_id, ai_response["text"], ai_response["emotion"]
            )
            
            # Step 7: Update Context
            await self._update_context(
                session_id, transcription["text"], ai_response["text"], ai_response["emotion"]
            )
            
            state.is_processing = False
            state.is_speaking = True
            state.current_emotion = ai_response["emotion"]
            
            return {
                "status": "success",
                "user_text": transcription["text"],
                "ai_text": ai_response["text"],
                "emotion": ai_response["emotion"],
                "audio_base64": audio_response["audio_base64"],
                "context": context,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            state.is_processing = False
            return {"status": "error", "error": str(e)}
    
    async def _transcribe_audio(self, session_id: str) -> Dict:
        """Transcribe audio using Whisper ASR"""
        state = self.active_sessions[session_id]
        
        # Convert buffer to audio data
        audio_array = np.array(list(state.audio_buffer))
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        async with self.http_session.post(
            f"{self.asr_url}/asr/transcribe",
            json={"audio_base64": audio_base64}
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                return {"text": "", "confidence": 0.0}
    
    async def _get_relevant_context(self, session_id: str, query: str) -> List[Dict]:
        """Get relevant context using Context Manager"""
        async with self.http_session.get(
            f"{self.context_url}/context/session/{session_id}/relevant",
            params={"query": query, "limit": 5}
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("relevant_context", [])
            else:
                return []
    
    async def _generate_response(self, session_id: str, user_text: str, 
                               context: List[Dict], emotion: str) -> Dict:
        """Generate response using Gemini LLM"""
        # Prepare context for LLM
        context_text = ""
        if context:
            context_text = "\n".join([
                f"Previous: {ctx.get('metadata', {}).get('text', '')}"
                for ctx in context[-3:]  # Last 3 relevant turns
            ])
        
        # Create prompt
        prompt = f"""
You are Oviya, an empathetic AI companion. Respond to the user's message with the specified emotion.

Context:
{context_text}

User: {user_text}
Emotion: {emotion}

Respond naturally and empathetically. Keep responses concise (1-2 sentences).
"""
        
        # Call Gemini API
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            response = model.generate_content(prompt)
            ai_text = response.text.strip()
            
            return {
                "text": ai_text,
                "emotion": emotion,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Gemini error: {e}")
            return {
                "text": "I'm sorry, I'm having trouble processing that right now.",
                "emotion": "concerned",
                "timestamp": time.time()
            }
    
    async def _generate_audio(self, session_id: str, text: str, emotion: str) -> Dict:
        """Generate audio using CSM TTS"""
        try:
            async with self.http_session.post(
                f"{self.csm_url}/tts",
                json={
                    "text": text,
                    "emotion": emotion,
                    "session_id": session_id,
                    "priority": "normal"
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    raise Exception(f"CSM service error: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ CSM error: {e}")
            raise Exception(f"CSM service unavailable: {e}")
    
    async def _update_context(self, session_id: str, user_text: str, 
                            ai_text: str, emotion: str):
        """Update conversation context"""
        async with self.http_session.post(
            f"{self.context_url}/context/session/{session_id}/turn",
            json={
                "user_text": user_text,
                "ai_text": ai_text,
                "emotion": emotion
            }
        ) as response:
            if response.status != 200:
                logger.error(f"❌ Context update failed: {response.status}")

# FastAPI app
app = FastAPI(title="Integrated Pipeline Orchestrator", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline
pipeline = None

@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup"""
    global pipeline
    pipeline = IntegratedPipeline()
    await pipeline.__aenter__()

@app.on_event("shutdown")
async def shutdown():
    """Cleanup pipeline on shutdown"""
    global pipeline
    if pipeline:
        await pipeline.__aexit__(None, None, None)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "integrated-pipeline",
        "active_sessions": len(pipeline.active_sessions) if pipeline else 0,
        "services": {
            "vad": pipeline.vad_url,
            "asr": pipeline.asr_url,
            "context": pipeline.context_url,
            "csm": pipeline.csm_url
        }
    }

@app.post("/pipeline/session/create")
async def create_session(request: dict):
    """Create a new pipeline session"""
    user_id = request.get("user_id", "anonymous")
    session_id = await pipeline.create_session(user_id)
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": time.time()
    }

@app.websocket("/pipeline/stream/{session_id}")
async def pipeline_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time pipeline streaming"""
    await websocket.accept()
    logger.info(f"🔗 Pipeline WebSocket connection established for session {session_id}")
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process through pipeline
            result = await pipeline.process_audio_chunk(session_id, data)
            
            if result and result.get("status") == "success":
                # Send complete response
                await websocket.send_text(json.dumps(result))
                
                # Mark as finished speaking
                if session_id in pipeline.active_sessions:
                    pipeline.active_sessions[session_id].is_speaking = False
            
    except WebSocketDisconnect:
        logger.info(f"🔌 Pipeline WebSocket connection closed for session {session_id}")
    except Exception as e:
        logger.error(f"❌ Pipeline WebSocket error: {e}")
        await websocket.close()

@app.get("/pipeline/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get session status"""
    if session_id not in pipeline.active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = pipeline.active_sessions[session_id]
    
    return {
        "session_id": session_id,
        "user_id": state.user_id,
        "is_listening": state.is_listening,
        "is_processing": state.is_processing,
        "is_speaking": state.is_speaking,
        "current_emotion": state.current_emotion,
        "last_activity": state.last_activity,
        "buffer_size": len(state.audio_buffer)
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Integrated Pipeline Orchestrator...")
    uvicorn.run(app, host="0.0.0.0", port=8004)
