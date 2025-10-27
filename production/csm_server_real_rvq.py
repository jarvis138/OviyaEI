#!/usr/bin/env python3
"""
CSM-1B Server with TRUE RVQ-Level Streaming
Based on Sesame's "Crossing the uncanny valley of conversational voice" paper

Key Features:
- RVQ-level streaming (12.5 Hz, 80ms per frame)
- Flush in 2-4 frame windows (160-320ms chunks)
- Conversational context awareness
- Emotion/prosody control
- True low-latency generation

Paper: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice

Architecture:
- Backbone (1B): Models zeroth codebook (semantic + prosody)
- Decoder (100M): Models remaining 31 codebooks (acoustic)
- Mimi: Decodes RVQ → PCM at 24kHz
"""

import sys
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
import base64
import io
import asyncio
from typing import Optional, List, Dict
import time
import torchaudio

# Import our RVQ streamer
try:
    from voice.csm_1b_stream import CSMRVQStreamer
except ImportError:
    # Fallback for direct execution
    from csm_1b_stream import CSMRVQStreamer

app = FastAPI(title="Oviya CSM-1B Server (RVQ Streaming)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0
    reference_emotion: str = "neutral"
    normalize_audio: bool = True
    conversation_context: Optional[List[Dict]] = None
    stream: bool = True
    flush_frames: int = 2  # RVQ frames per chunk (2-4 recommended by paper)

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int = 24000
    duration_ms: float

class HealthResponse(BaseModel):
    status: str
    service: str
    mode: str
    architecture: Dict
    performance: Dict

# Global streamer
csm_streamer = None

@app.on_event("startup")
async def startup():
    """Initialize CSM-1B RVQ streamer"""
    global csm_streamer
    
    print()
    print("=" * 70)
    print("🚀 INITIALIZING OVIYA CSM-1B RVQ STREAMING SERVER")
    print("=" * 70)
    print()
    print("   Based on Sesame's paper:")
    print("   'Crossing the uncanny valley of conversational voice'")
    print()
    
    try:
        csm_streamer = CSMRVQStreamer(
            model_id="sesame/csm-1b",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
            flush_rvq_frames=2,  # 160ms chunks (paper optimal)
            cache_dir="/workspace/.cache/huggingface"
        )
        
        print()
        print("=" * 70)
        print("✅ SERVER READY - TRUE RVQ STREAMING ENABLED")
        print("=" * 70)
        print()
        print("   Latency: ~160ms first audio (2 RVQ frames)")
        print("   Quality: 24kHz, 32 codebooks (1 semantic + 31 acoustic)")
        print("   Context: Last 3 conversation turns (~90s)")
        print()
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize CSM streamer: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.post("/generate/stream")
async def generate_audio_streaming(request: TTSRequest):
    """
    Stream audio with RVQ-level granularity
    
    Paper: "The decoder is significantly smaller than the backbone,
    enabling low-latency generation while keeping the model end-to-end."
    
    Streaming Strategy:
    - Generate RVQ tokens incrementally
    - Decode in 2-4 frame windows (160-320ms)
    - Yield PCM chunks immediately
    """
    if csm_streamer is None:
        raise HTTPException(status_code=503, detail="CSM streamer not initialized")
    
    print(f"🎵 Streaming request: '{request.text[:50]}...'")
    print(f"   Emotion: {request.reference_emotion}")
    print(f"   Context: {len(request.conversation_context) if request.conversation_context else 0} turns")
    print(f"   Flush: {request.flush_frames} RVQ frames")
    
    async def generate_chunks():
        chunk_index = 0
        total_audio_ms = 0
        start_time = time.time()
        
        try:
            async for pcm_chunk in csm_streamer.generate_streaming(
                text=request.text,
                emotion=request.reference_emotion,
                conversation_context=request.conversation_context
            ):
                # Convert PCM to WAV bytes
                audio_tensor = torch.from_numpy(pcm_chunk).unsqueeze(0)  # [1, samples]
                buffer = io.BytesIO()
                torchaudio.save(buffer, audio_tensor, 24000, format="wav")
                buffer.seek(0)
                audio_bytes = buffer.read()
                
                chunk_duration_ms = len(pcm_chunk) / 24000 * 1000
                total_audio_ms += chunk_duration_ms
                
                # Create response chunk
                chunk_data = {
                    "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
                    "chunk_index": chunk_index,
                    "chunk_duration_ms": chunk_duration_ms,
                    "sample_rate": 24000,
                    "quality": "csm_1b_rvq_streaming",
                    "architecture": "backbone_decoder_split"
                }
                
                yield f"data: {base64.b64encode(str(chunk_data).encode()).decode()}\n\n"
                
                chunk_index += 1
                await asyncio.sleep(0.001)
            
            # Send completion marker
            total_time = time.time() - start_time
            completion = {
                "type": "complete",
                "total_chunks": chunk_index,
                "total_audio_ms": total_audio_ms,
                "total_time_s": total_time,
                "real_time_factor": total_audio_ms / (total_time * 1000)
            }
            yield f"data: {base64.b64encode(str(completion).encode()).decode()}\n\n"
            
            print(f"   ✅ Streamed {chunk_index} chunks, {total_audio_ms:.0f}ms audio in {total_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Streaming error: {e}")
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {base64.b64encode(str(error_data).encode()).decode()}\n\n"
    
    return StreamingResponse(generate_chunks(), media_type="text/event-stream")

@app.post("/generate", response_model=TTSResponse)
async def generate_audio(request: TTSRequest):
    """
    Generate complete audio (non-streaming, for testing/comparison)
    
    Collects all RVQ chunks and returns as single response
    """
    if csm_streamer is None:
        raise HTTPException(status_code=503, detail="CSM streamer not initialized")
    
    try:
        start_time = time.time()
        print(f"🎵 Generating: '{request.text[:50]}...'")
        
        # Collect all chunks
        audio_chunks = []
        async for pcm_chunk in csm_streamer.generate_streaming(
            text=request.text,
            emotion=request.reference_emotion,
            conversation_context=request.conversation_context
        ):
            audio_chunks.append(pcm_chunk)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)
        audio_tensor = torch.from_numpy(full_audio).unsqueeze(0)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, 24000, format="wav")
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        generation_time = time.time() - start_time
        duration_ms = len(full_audio) / 24000 * 1000
        
        print(f"   ✅ Generated {duration_ms:.0f}ms in {generation_time:.2f}s")
        print(f"   Real-time factor: {duration_ms / (generation_time * 1000):.2f}x")
        
        return TTSResponse(
            audio_base64=base64.b64encode(audio_bytes).decode('utf-8'),
            sample_rate=24000,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check with architecture details
    
    Returns information about Sesame's CSM architecture
    """
    return HealthResponse(
        status="healthy" if csm_streamer else "initializing",
        service="Oviya CSM-1B (RVQ Streaming)",
        mode="rvq_streaming",
        architecture={
            "paper": "Crossing the uncanny valley of conversational voice",
            "model": "sesame/csm-1b",
            "backbone_size": "1B parameters",
            "decoder_size": "100M parameters",
            "num_codebooks": 32,
            "semantic_codebooks": 1,
            "acoustic_codebooks": 31,
            "rvq_frame_rate_hz": 12.5,
            "rvq_frame_duration_ms": 80,
            "codec": "Mimi (split-RVQ)",
            "sample_rate": 24000
        },
        performance={
            "flush_frames": 2,
            "chunk_duration_ms": 160,
            "latency_target_ms": "~160 (first audio)",
            "streaming": True,
            "real_time_capable": True
        }
    )

@app.get("/")
async def root():
    """API documentation"""
    return {
        "service": "Oviya CSM-1B Server (RVQ Streaming)",
        "version": "1.0.0",
        "paper": "https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice",
        "architecture": {
            "type": "Two-transformer design (Backbone + Decoder)",
            "streaming": "RVQ-level (12.5 Hz, 80ms frames)",
            "latency": "~160ms first audio"
        },
        "endpoints": {
            "/generate": {
                "method": "POST",
                "description": "Generate complete audio (non-streaming)",
                "parameters": {
                    "text": "Text to synthesize",
                    "reference_emotion": "Emotion tag (e.g., 'joyful', 'calm', 'neutral')",
                    "conversation_context": "List of previous turns for context"
                }
            },
            "/generate/stream": {
                "method": "POST",
                "description": "Stream audio with RVQ-level granularity (recommended)",
                "parameters": "Same as /generate",
                "streaming": "Server-Sent Events (SSE)"
            },
            "/health": {
                "method": "GET",
                "description": "Health check and architecture details"
            }
        },
        "paper_findings": {
            "without_context": "CSM matches human naturalness",
            "with_context": "Gap remains in prosodic appropriateness",
            "objective_metrics": "Near-human WER and speaker similarity",
            "novel_tests": "Homograph disambiguation, pronunciation consistency"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print()
    print("=" * 70)
    print("🎤 OVIYA CSM-1B RVQ STREAMING SERVER")
    print("=" * 70)
    print()
    print("   Starting on http://0.0.0.0:19517")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=19517,
        log_level="info"
    )

