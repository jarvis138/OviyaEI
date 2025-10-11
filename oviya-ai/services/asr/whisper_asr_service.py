#!/usr/bin/env python3
"""
Whisper ASR Service
Real-time Speech Recognition using OpenAI Whisper
"""
import asyncio
import numpy as np
import torch
import torchaudio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import time
import io
import base64
import whisper
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperASRProcessor:
    """Whisper ASR processor for real-time speech recognition"""
    
    def __init__(self, model_size: str = "base", sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model_size = model_size
        self.model = None
        self.audio_buffer = deque(maxlen=int(sample_rate * 30))  # 30 second buffer
        self.is_processing = False
        
        # Load Whisper model
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"🔄 Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            logger.info("✅ Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to buffer"""
        self.audio_buffer.extend(audio_data)
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio data using Whisper
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Transcription result with text and metadata
        """
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcribe with Whisper
            start_time = time.time()
            result = self.model.transcribe(audio_data)
            processing_time = time.time() - start_time
            
            # Extract text and metadata
            text = result["text"].strip()
            
            # Calculate confidence (Whisper doesn't provide direct confidence)
            # Use segment-level confidence if available
            confidence = 0.8  # Default confidence
            if "segments" in result and result["segments"]:
                confidences = [seg.get("avg_logprob", 0) for seg in result["segments"]]
                if confidences:
                    confidence = max(0, min(1, np.mean(confidences) + 1))  # Convert logprob to confidence
            
            return {
                "text": text,
                "confidence": confidence,
                "processing_time": processing_time,
                "language": result.get("language", "en"),
                "segments": result.get("segments", []),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in transcription: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def transcribe_buffer(self) -> Dict[str, Any]:
        """Transcribe current audio buffer"""
        if len(self.audio_buffer) < self.sample_rate * 0.5:  # Minimum 0.5 seconds
            return {
                "text": "",
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": "Insufficient audio data",
                "timestamp": time.time()
            }
        
        # Convert buffer to numpy array
        audio_data = np.array(list(self.audio_buffer))
        
        # Transcribe
        result = self.transcribe_audio(audio_data)
        
        # Clear buffer after transcription
        self.audio_buffer.clear()
        
        return result
    
    def reset(self):
        """Reset ASR state"""
        self.audio_buffer.clear()
        self.is_processing = False

# FastAPI app
app = FastAPI(title="Whisper ASR Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ASR processor
asr_processor = WhisperASRProcessor()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "whisper-asr",
        "model_loaded": asr_processor.model is not None,
        "model_size": asr_processor.model_size,
        "sample_rate": asr_processor.sample_rate
    }

@app.post("/asr/transcribe")
async def transcribe_audio(request: dict):
    """Transcribe audio chunk"""
    try:
        # Get audio data
        audio_base64 = request.get("audio_base64", "")
        if not audio_base64:
            return {"error": "No audio data provided"}
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        result = asr_processor.transcribe_audio(audio_data)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in transcription: {e}")
        return {"error": str(e)}

@app.post("/asr/transcribe_buffer")
async def transcribe_buffer():
    """Transcribe current audio buffer"""
    try:
        result = asr_processor.transcribe_buffer()
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in buffer transcription: {e}")
        return {"error": str(e)}

@app.post("/asr/add_chunk")
async def add_audio_chunk(request: dict):
    """Add audio chunk to buffer"""
    try:
        # Get audio data
        audio_base64 = request.get("audio_base64", "")
        if not audio_base64:
            return {"error": "No audio data provided"}
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        asr_processor.add_audio_chunk(audio_data)
        
        return {
            "status": "added",
            "buffer_size": len(asr_processor.audio_buffer),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Error adding audio chunk: {e}")
        return {"error": str(e)}

@app.websocket("/asr/stream")
async def asr_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time ASR streaming"""
    await websocket.accept()
    logger.info("🔗 ASR WebSocket connection established")
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Add to buffer
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            asr_processor.add_audio_chunk(audio_data)
            
            # Send buffer status
            result = {
                "buffer_size": len(asr_processor.audio_buffer),
                "timestamp": time.time()
            }
            
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        logger.info("🔌 ASR WebSocket connection closed")
    except Exception as e:
        logger.error(f"❌ ASR WebSocket error: {e}")
        await websocket.close()

@app.post("/asr/reset")
async def reset_asr():
    """Reset ASR state"""
    asr_processor.reset()
    return {"status": "reset", "timestamp": time.time()}

if __name__ == "__main__":
    logger.info("🚀 Starting Whisper ASR Service...")
    uvicorn.run(app, host="0.0.0.0", port=8002)


