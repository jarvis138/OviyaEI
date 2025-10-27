#!/usr/bin/env python3
"""
Silero VAD Service
Real-time Voice Activity Detection using Silero VAD v5
"""
import asyncio
import numpy as np
import torch
import torchaudio
import logging
from typing import AsyncGenerator, Tuple, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import time
from collections import deque
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SileroVADProcessor:
    """Silero VAD processor for real-time voice activity detection"""
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = None
        self.utils = None
        self.audio_buffer = deque(maxlen=int(sample_rate * 0.5))  # 0.5 second buffer
        self.is_speech = False
        self.speech_start_time = None
        self.silence_duration = 0
        
        # Load Silero VAD model
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model"""
        try:
            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='silero-models',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.model = model
            self.utils = utils
            
            logger.info("✅ Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Silero VAD model: {e}")
            raise
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> Tuple[bool, float, dict]:
        """
        Process audio chunk for voice activity detection
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Tuple of (is_speech, confidence, metadata)
        """
        try:
            # Convert to tensor if needed
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).float()
            else:
                audio_tensor = audio_data
            
            # Ensure correct shape
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Resample if needed
            if audio_tensor.shape[1] != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    audio_tensor.shape[1], self.sample_rate
                )
                audio_tensor = resampler(audio_tensor)
            
            # Get speech probability
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            # Determine if speech is detected
            is_speech = speech_prob > self.threshold
            
            # Update state
            current_time = time.time()
            
            if is_speech:
                if not self.is_speech:
                    self.speech_start_time = current_time
                    self.silence_duration = 0
                self.is_speech = True
            else:
                if self.is_speech:
                    self.silence_duration = current_time - self.speech_start_time
                    # End speech if silence duration exceeds threshold
                    if self.silence_duration > 0.5:  # 500ms silence threshold
                        self.is_speech = False
                        self.speech_start_time = None
            
            # Create metadata
            metadata = {
                "speech_probability": speech_prob,
                "is_speech": is_speech,
                "silence_duration": self.silence_duration,
                "timestamp": current_time,
                "threshold": self.threshold
            }
            
            return is_speech, speech_prob, metadata
            
        except Exception as e:
            logger.error(f"❌ Error processing audio chunk: {e}")
            return False, 0.0, {"error": str(e)}
    
    def reset(self):
        """Reset VAD state"""
        self.is_speech = False
        self.speech_start_time = None
        self.silence_duration = 0
        self.audio_buffer.clear()

# FastAPI app
app = FastAPI(title="Silero VAD Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global VAD processor
vad_processor = SileroVADProcessor()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "silero-vad",
        "model_loaded": vad_processor.model is not None,
        "sample_rate": vad_processor.sample_rate,
        "threshold": vad_processor.threshold
    }

@app.post("/vad/detect")
async def detect_speech(request: dict):
    """Detect speech in audio chunk"""
    try:
        # Get audio data
        audio_base64 = request.get("audio_base64", "")
        if not audio_base64:
            return {"error": "No audio data provided"}
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Process with VAD
        is_speech, confidence, metadata = vad_processor.process_audio_chunk(audio_data)
        
        return {
            "is_speech": is_speech,
            "confidence": confidence,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in speech detection: {e}")
        return {"error": str(e)}

@app.websocket("/vad/stream")
async def vad_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time VAD streaming"""
    await websocket.accept()
    logger.info("🔗 VAD WebSocket connection established")
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process audio chunk
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            is_speech, confidence, metadata = vad_processor.process_audio_chunk(audio_data)
            
            # Send VAD result
            result = {
                "is_speech": is_speech,
                "confidence": confidence,
                "metadata": metadata,
                "timestamp": time.time()
            }
            
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        logger.info("🔌 VAD WebSocket connection closed")
    except Exception as e:
        logger.error(f"❌ VAD WebSocket error: {e}")
        await websocket.close()

@app.post("/vad/reset")
async def reset_vad():
    """Reset VAD state"""
    vad_processor.reset()
    return {"status": "reset", "timestamp": time.time()}

if __name__ == "__main__":
    logger.info("🚀 Starting Silero VAD Service...")
    uvicorn.run(app, host="0.0.0.0", port=8001)


