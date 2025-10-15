#!/bin/bash
# Fix CSM audio quality by using remote API with real decoding
# Paste this entire script on Vast.ai

cd /workspace/oviya-production

cat > csm_server_real.py << 'EOFHYBRID'
#!/usr/bin/env python3
"""
Hybrid CSM-1B Server: Use remote API for real audio quality
"""

import sys
import os

os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
import base64
import io
import asyncio
import aiohttp
from typing import Optional, List, Dict
import time
import re

app = FastAPI(title="Oviya CSM-1B Server (Hybrid)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remote CSM API for high-quality audio
REMOTE_CSM_URL = "https://astronomy-initiative-paso-cream.trycloudflare.com/generate"

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0
    reference_emotion: str = "neutral"
    normalize_audio: bool = True
    conversation_context: Optional[List[Dict]] = None
    stream: bool = True

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int = 24000
    duration_ms: float
    emotions: int = 49

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'([.!?]+\s+)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)
    if not result:
        result = [text]
    return result

async def generate_audio_remote(text: str, emotion: str) -> bytes:
    """Use remote CSM API for REAL high-quality audio"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "text": text,
            "speaker": 0,
            "reference_emotion": emotion,
            "normalize_audio": True
        }
        
        try:
            async with session.post(REMOTE_CSM_URL, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    audio_bytes = base64.b64decode(data['audio_base64'])
                    return audio_bytes
                else:
                    print(f"Remote API error: {response.status}")
                    return None
        except Exception as e:
            print(f"Remote API failed: {e}")
            return None

async def stream_audio_chunks(text: str, emotion: str):
    """Stream using remote CSM API (real quality)"""
    sentences = split_into_sentences(text)
    
    print(f"   Streaming {len(sentences)} sentences via REMOTE API...")
    
    for i, sentence in enumerate(sentences):
        start = time.time()
        
        # Get REAL audio from remote API
        audio_bytes = await generate_audio_remote(sentence, emotion)
        
        if not audio_bytes:
            print(f"   ❌ Failed to generate chunk {i+1}")
            continue
        
        gen_time = time.time() - start
        
        print(f"   Chunk {i+1}/{len(sentences)}: Generated in {gen_time:.3f}s (REMOTE)")
        
        chunk_data = {
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
            "chunk_index": i,
            "total_chunks": len(sentences),
            "sample_rate": 24000,
            "quality": "remote_csm_api"
        }
        
        yield f"data: {base64.b64encode(str(chunk_data).encode()).decode()}\n\n"
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup():
    print("=" * 70)
    print("🎤 OVIYA CSM-1B SERVER (HYBRID MODE)")
    print("=" * 70)
    print("   Mode: Remote CSM API (real audio quality)")
    print(f"   Remote: {REMOTE_CSM_URL}")
    print("=" * 70)

@app.post("/generate/stream")
async def generate_audio_streaming(request: TTSRequest):
    print(f"🎵 Streaming via REMOTE: '{request.text[:50]}...'")
    
    return StreamingResponse(
        stream_audio_chunks(request.text, request.reference_emotion),
        media_type="text/event-stream"
    )

@app.post("/generate", response_model=TTSResponse)
async def generate_audio(request: TTSRequest):
    try:
        start_time = time.time()
        
        print(f"🎵 Generating via REMOTE: '{request.text[:50]}...'")
        
        # Use remote API for real quality
        audio_bytes = await generate_audio_remote(request.text, request.reference_emotion)
        
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Remote API failed")
        
        generation_time = time.time() - start_time
        
        # Decode to get duration
        import torchaudio
        buffer = io.BytesIO(audio_bytes)
        audio_tensor, sample_rate = torchaudio.load(buffer)
        duration_ms = (audio_tensor.shape[1] / sample_rate) * 1000
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"   ✅ {duration_ms:.0f}ms in {generation_time:.3f}s (REMOTE)")
        
        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=24000,
            duration_ms=duration_ms,
            emotions=49
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Oviya CSM-1B (Hybrid - Remote API)",
        "mode": "remote",
        "remote_url": REMOTE_CSM_URL,
        "quality": "high",
        "streaming": True
    }

@app.get("/")
async def root():
    return {
        "service": "Oviya CSM-1B Server (Hybrid)",
        "mode": "remote_api",
        "quality": "high_fidelity",
        "endpoints": {
            "/generate": "POST - Generate audio",
            "/generate/stream": "POST - Stream audio (recommended)",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=19517, log_level="info")
EOFHYBRID

# Install aiohttp for async requests
pip3 install aiohttp --quiet

# Restart server
./stop_csm.sh
./start_csm_1b.sh

echo ""
echo "✅ Server updated to use REMOTE CSM API (real audio quality)"
echo "🧪 Test: curl http://localhost:19517/health"

