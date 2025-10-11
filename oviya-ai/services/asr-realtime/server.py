#!/usr/bin/env python3
"""
FastAPI WebSocket server for real-time ASR with Silero VAD + Whisper
"""
import asyncio
import json
import logging
import time
from typing import Dict, Optional
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from asr_pipeline import WhisperSileroASR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Oviya ASR Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ASR pipeline
asr_pipeline = None

class TranscriptionRequest(BaseModel):
    audio: str  # Base64 encoded audio
    sample_rate: int = 16000
    language: Optional[str] = "en"

class TranscriptionResponse(BaseModel):
    text: str
    confidence: float
    latency_ms: float
    language: str
    segments: list

@app.on_event("startup")
async def startup_event():
    """Initialize ASR pipeline on startup"""
    global asr_pipeline
    try:
        asr_pipeline = WhisperSileroASR(whisper_model_size="small.en")
        await asr_pipeline.initialize()
        logger.info("ASR service started successfully")
    except Exception as e:
        logger.error(f"Failed to start ASR service: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="ASR pipeline not initialized")
    
    stats = asr_pipeline.get_performance_stats()
    return {
        "status": "healthy",
        "service": "asr-realtime",
        "version": "1.0.0",
        "stats": stats
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """Transcribe audio from base64 string"""
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="ASR pipeline not initialized")
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio)
        
        # Process audio
        result = await asr_pipeline.whisper.transcribe_audio(audio_bytes, request.sample_rate)
        
        return TranscriptionResponse(
            text=result['text'],
            confidence=result['confidence'],
            latency_ms=result['latency_ms'],
            language=result['language'],
            segments=result['segments']
        )
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/asr/stream")
async def asr_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time ASR streaming"""
    await websocket.accept()
    
    if asr_pipeline is None:
        await websocket.close(code=1011, reason="ASR pipeline not initialized")
        return
    
    try:
        logger.info("ASR WebSocket connection established")
        
        # Create audio stream generator
        async def audio_stream():
            while True:
                try:
                    # Receive audio chunk
                    data = await websocket.receive_bytes()
                    yield data
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected")
                    break
                except Exception as e:
                    logger.error(f"Audio stream error: {e}")
                    break
        
        # Process audio stream
        async for result in asr_pipeline.process_stream(audio_stream()):
            try:
                # Send result back to client
                await websocket.send_text(json.dumps(result))
                
                # Handle interrupt
                if result['type'] == 'interrupt':
                    logger.info("Interrupt detected, stopping processing")
                    break
                
            except WebSocketDisconnect:
                logger.info("Client disconnected during processing")
                break
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                break
    
    except Exception as e:
        logger.error(f"ASR stream error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    
    finally:
        asr_pipeline.stop_processing()
        logger.info("ASR stream ended")

@app.websocket("/asr/stream/{session_id}")
async def asr_stream_with_session(websocket: WebSocket, session_id: str):
    """WebSocket endpoint with session tracking"""
    await websocket.accept()
    
    if asr_pipeline is None:
        await websocket.close(code=1011, reason="ASR pipeline not initialized")
        return
    
    logger.info(f"ASR WebSocket connection established for session: {session_id}")
    
    try:
        # Send session confirmation
        await websocket.send_text(json.dumps({
            'type': 'session_ready',
            'session_id': session_id,
            'timestamp': time.time()
        }))
        
        # Create audio stream generator
        async def audio_stream():
            while True:
                try:
                    # Receive audio chunk
                    data = await websocket.receive_bytes()
                    yield data
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for session: {session_id}")
                    break
                except Exception as e:
                    logger.error(f"Audio stream error for session {session_id}: {e}")
                    break
        
        # Process audio stream
        async for result in asr_pipeline.process_stream(audio_stream()):
            try:
                # Add session info to result
                result['session_id'] = session_id
                
                # Send result back to client
                await websocket.send_text(json.dumps(result))
                
                # Handle interrupt
                if result['type'] == 'interrupt':
                    logger.info(f"Interrupt detected for session {session_id}, stopping processing")
                    break
                
            except WebSocketDisconnect:
                logger.info(f"Client disconnected during processing for session: {session_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket send error for session {session_id}: {e}")
                break
    
    except Exception as e:
        logger.error(f"ASR stream error for session {session_id}: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    
    finally:
        asr_pipeline.stop_processing()
        logger.info(f"ASR stream ended for session: {session_id}")

@app.get("/stats")
async def get_stats():
    """Get ASR performance statistics"""
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="ASR pipeline not initialized")
    
    return asr_pipeline.get_performance_stats()

@app.post("/reset_stats")
async def reset_stats():
    """Reset performance statistics"""
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="ASR pipeline not initialized")
    
    asr_pipeline.reset_stats()
    return {"message": "Stats reset successfully"}

@app.post("/set_ai_speaking")
async def set_ai_speaking(is_speaking: bool):
    """Set AI speaking state for interrupt detection"""
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="ASR pipeline not initialized")
    
    asr_pipeline.set_ai_speaking_state(is_speaking)
    return {"ai_speaking": is_speaking}

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )