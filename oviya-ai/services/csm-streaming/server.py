#!/usr/bin/env python3
"""
Oviya CSM Streaming Service
Core TTS service with audio context emotions
"""
import asyncio
import json
import time
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TTSRequest:
    text: str
    emotion: str = "empathetic"
    session_id: str = ""
    priority: str = "normal"
    user_audio: Optional[bytes] = None

@dataclass
class TTSResponse:
    audio_chunk: bytes
    text_chunk: str
    emotion: str
    chunk_latency_ms: float
    is_final: bool = False

class AudioContextManager:
    """Manages audio context for emotional speech generation"""
    
    def __init__(self):
        # Pre-recorded emotion samples (3-5 seconds each)
        self.emotion_prompts = {
            "empathetic": "emotion_prompts/empathetic.wav",
            "encouraging": "emotion_prompts/encouraging.wav",
            "calm": "emotion_prompts/calm.wav",
            "concerned": "emotion_prompts/concerned.wav",
            "joyful": "emotion_prompts/joyful.wav"
        }
        
        # Cache loaded audio to reduce I/O
        self.cached_prompts = {}
        self._load_all_prompts()
        
        # Temperature settings for variation
        self.temperature_map = {
            "empathetic": 0.7,
            "encouraging": 0.85,
            "calm": 0.5,
            "concerned": 0.6,
            "joyful": 0.9
        }
    
    def _load_all_prompts(self):
        """Pre-load all emotion prompts into memory"""
        for emotion, path in self.emotion_prompts.items():
            try:
                if os.path.exists(path):
                    waveform, sample_rate = torchaudio.load(path)
                    self.cached_prompts[emotion] = {
                        "waveform": waveform,
                        "sample_rate": sample_rate
                    }
                    logger.info(f"Loaded emotion prompt: {emotion}")
                else:
                    logger.warning(f"Emotion prompt not found: {path}")
            except Exception as e:
                logger.error(f"Error loading emotion prompt {emotion}: {e}")
    
    async def prepare_conversation_context(
        self, 
        text: str, 
        emotion: str,
        user_audio: Optional[bytes] = None,
        previous_context: Optional[List] = None
    ) -> Dict:
        """Prepare CSM conversation format with audio context"""
        conversation = []
        
        # Add emotion prompt as first context (AI speaking in target emotion)
        if emotion in self.cached_prompts:
            conversation.append({
                "role": "1",  # AI speaker
                "content": [
                    {"type": "text", "text": "Speaking warmly"},  # Placeholder text
                    {"type": "audio", "data": self.cached_prompts[emotion]}
                ]
            })
        
        # Add user audio if available (for mirroring tone)
        if user_audio:
            conversation.append({
                "role": "0",  # User speaker
                "content": [
                    {"type": "text", "text": "[user speech]"},
                    {"type": "audio", "data": user_audio}
                ]
            })
        
        # Add recent conversation context (last 2-3 turns max for memory)
        if previous_context and len(previous_context) > 0:
            # Limit to last 3 turns to prevent context overflow
            for turn in previous_context[-3:]:
                conversation.append(turn)
        
        # Add the actual response to generate
        conversation.append({
            "role": "1",
            "content": [
                {"type": "text", "text": text}
            ]
        })
        
        return {
            "conversation": conversation,
            "temperature": self.temperature_map.get(emotion, 0.7),
            "do_sample": True,
            "cache_implementation": "static"  # For faster inference
        }

class CSMGenerationPipeline:
    """Handles CSM generation with proper parameters"""
    
    def __init__(self):
        self.generator = None  # Loaded in initialize()
        self.audio_context_manager = AudioContextManager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Load CSM model"""
        try:
            logger.info("Loading CSM model...")
            
            # Import CSM generator
            import sys
            sys.path.append('/workspace/csm')
            from generator import load_csm_1b
            
            self.generator = load_csm_1b(device=self.device)
            logger.info(f"CSM model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading CSM model: {e}")
            raise Exception(f"Failed to load CSM model: {e}")
    
    async def generate_streaming(
        self,
        text: str,
        emotion: str = "empathetic",
        user_audio: Optional[bytes] = None,
        session_context: Optional[List] = None
    ) -> AsyncGenerator[TTSResponse, None]:
        """Stream audio generation with emotion through audio context"""
        
        # Prepare context with emotion
        context = await self.audio_context_manager.prepare_conversation_context(
            text=text,
            emotion=emotion,
            user_audio=user_audio,
            previous_context=session_context
        )
        
        # Chunk text for streaming (5-10 words)
        chunks = self._smart_chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            start_time = time.time()
            
            try:
                # Generate audio with CSM
                audio = self.generator.generate(
                    text=chunk,
                    speaker=1,
                    context=context["conversation"],
                    max_audio_length_ms=10000,
                    temperature=context["temperature"],
                    do_sample=context["do_sample"]
                )
                
                # Convert to PCM bytes
                audio_chunk = self._convert_to_pcm(audio)
                
                latency = (time.time() - start_time) * 1000
                
                yield TTSResponse(
                    audio_chunk=audio_chunk,
                    text_chunk=chunk,
                    emotion=emotion,
                    chunk_latency_ms=latency,
                    is_final=(i == len(chunks) - 1)
                )
                
            except Exception as e:
                logger.error(f"Error generating chunk {i}: {e}")
                # Return empty chunk on error
                yield TTSResponse(
                    audio_chunk=b"",
                    text_chunk=chunk,
                    emotion=emotion,
                    chunk_latency_ms=0,
                    is_final=(i == len(chunks) - 1)
                )
    
    def _smart_chunk_text(self, text: str, words_per_chunk: int = 7) -> List[str]:
        """Split text into speakable chunks"""
        import re
        
        # Split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) <= words_per_chunk:
                chunks.append(sentence)
            else:
                # Split long sentences
                for i in range(0, len(words), words_per_chunk):
                    chunk = ' '.join(words[i:i + words_per_chunk])
                    if i + words_per_chunk < len(words):
                        chunk += ','
                    chunks.append(chunk)
        
        return chunks
    
    def _convert_to_pcm(self, audio: torch.Tensor) -> bytes:
        """Convert tensor to PCM bytes"""
        try:
            # Ensure audio is on CPU and in correct format
            audio_np = audio.cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return audio_int16.tobytes()
        except Exception as e:
            logger.error(f"Error converting audio to PCM: {e}")
            return b""


class QueueManager:
    """Manages request queue with priority handling"""
    
    def __init__(self, max_queue_size: int = 20):
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.priority_queue = asyncio.Queue(maxsize=5)  # Priority lane
        self.active_requests = {}
        
    async def add_request(self, request: TTSRequest) -> bool:
        """Add request to queue"""
        try:
            if request.priority == "high":
                await self.priority_queue.put(request)
            else:
                await self.queue.put(request)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full, rejecting request: {request.session_id}")
            return False
    
    async def get_next_request(self) -> Optional[TTSRequest]:
        """Get next request from queue (priority first)"""
        try:
            # Check priority queue first
            if not self.priority_queue.empty():
                return await self.priority_queue.get()
            
            # Then regular queue
            if not self.queue.empty():
                return await self.queue.get()
            
            return None
        except Exception as e:
            logger.error(f"Error getting next request: {e}")
            return None

class ModelPool:
    """Manages warm model instances"""
    
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self.pipeline_pool = []
        self.available_pipelines = asyncio.Queue()
        
    async def initialize(self):
        """Initialize model pool"""
        logger.info(f"Initializing model pool with {self.pool_size} instances...")
        
        for i in range(self.pool_size):
            pipeline = CSMGenerationPipeline()
            await pipeline.initialize()
            self.pipeline_pool.append(pipeline)
            await self.available_pipelines.put(pipeline)
        
        logger.info(f"Model pool initialized with {len(self.pipeline_pool)} instances")
    
    async def get_pipeline(self) -> CSMGenerationPipeline:
        """Get available pipeline from pool"""
        return await self.available_pipelines.get()
    
    async def return_pipeline(self, pipeline: CSMGenerationPipeline):
        """Return pipeline to pool"""
        await self.available_pipelines.put(pipeline)

# FastAPI Application
app = FastAPI(title="Oviya CSM Streaming Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_pool = ModelPool(pool_size=2)
queue_manager = QueueManager(max_queue_size=20)

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting CSM Streaming Service...")
    await model_pool.initialize()
    logger.info("CSM Streaming Service ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down CSM Streaming Service...")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "csm-streaming",
        "timestamp": time.time(),
        "model_pool_size": len(model_pool.pipeline_pool),
        "queue_size": queue_manager.queue.qsize(),
        "priority_queue_size": queue_manager.priority_queue.qsize()
    }

@app.post("/tts")
async def tts_endpoint(request: Dict):
    """Single TTS request endpoint"""
    try:
        tts_request = TTSRequest(
            text=request.get("text", ""),
            emotion=request.get("emotion", "empathetic"),
            session_id=request.get("session_id", ""),
            priority=request.get("priority", "normal")
        )
        
        if not tts_request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Get pipeline from pool
        pipeline = await model_pool.get_pipeline()
        
        try:
            # Generate audio
            audio_chunks = []
            total_latency = 0
            first_chunk_latency = None
            
            async for response in pipeline.generate_streaming(
                text=tts_request.text,
                emotion=tts_request.emotion
            ):
                audio_chunks.append(response.audio_chunk)
                total_latency += response.chunk_latency_ms
                
                if first_chunk_latency is None:
                    first_chunk_latency = response.chunk_latency_ms
            
            # Combine all chunks
            combined_audio = b"".join(audio_chunks)
            
            return {
                "audio": combined_audio.hex(),  # Return as hex string
                "text": tts_request.text,
                "emotion": tts_request.emotion,
                "first_chunk_latency": first_chunk_latency or 0,
                "total_latency": total_latency,
                "chunks": len(audio_chunks)
            }
        
        finally:
            # Return pipeline to pool
            await model_pool.return_pipeline(pipeline)
    
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/tts/stream")
async def tts_stream_endpoint(websocket: WebSocket):
    """WebSocket streaming TTS endpoint"""
    await websocket.accept()
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            tts_request = TTSRequest(
                text=data.get("text", ""),
                emotion=data.get("emotion", "empathetic"),
                session_id=data.get("session_id", ""),
                priority=data.get("priority", "normal")
            )
            
            if not tts_request.text:
                await websocket.send_json({"error": "Text is required"})
                continue
            
            # Get pipeline from pool
            pipeline = await model_pool.get_pipeline()
            
            try:
                # Stream audio chunks
                async for response in pipeline.generate_streaming(
                    text=tts_request.text,
                    emotion=tts_request.emotion
                ):
                    await websocket.send_json({
                        "audio_chunk": response.audio_chunk.hex(),
                        "text_chunk": response.text_chunk,
                        "emotion": response.emotion,
                        "chunk_latency_ms": response.chunk_latency_ms,
                        "is_final": response.is_final
                    })
                
                # Send completion signal
                await websocket.send_json({"status": "complete"})
            
            finally:
                # Return pipeline to pool
                await model_pool.return_pipeline(pipeline)
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket stream: {e}")
        await websocket.send_json({"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
