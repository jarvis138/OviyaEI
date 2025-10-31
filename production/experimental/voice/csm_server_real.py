#!/usr/bin/env python3
"""
REAL CSM-1B Server with Mimi Decoder
Generates actual high-quality speech using Sesame's CSM-1B model
"""

import sys
import os

os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

# Model paths
model_path = "/root/.cache/huggingface/hub/models--sesame--csm-1b/snapshots/7a545da5caab051c5f5e6d9be161a5197e756a02"
model_path_mimi = "/root/.cache/huggingface/hub/models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, MimiModel
import numpy as np
import base64
import io
import asyncio
from typing import Optional, List, Dict
import time
import re
import torchaudio

@asynccontextmanager
async def lifespan(app: FastAPI):
    global csm_model, processor, mimi_decoder

    print("=" * 70)
    print("🎤 LOADING CSM-1B + MIMI DECODER")
    print("=" * 70)
    print("🔄 Lifespan function started")

    try:
        print("🔄 STEP 1: Entering try block")
        # Load processor
        print("📥 STEP 2: Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        print("✅ STEP 3: Processor loaded")

        # Load CSM model
        print("📥 STEP 4: Loading CSM-1B model...")
        csm_model = CsmForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        csm_model.eval()
        print("✅ STEP 5: CSM-1B model loaded")

        # Load Mimi decoder
        print("📥 STEP 6: Loading Mimi decoder...")
        mimi_decoder = MimiModel.from_pretrained(
            model_path_mimi,
            local_files_only=True
        ).to("cuda").eval()
        print("✅ STEP 7: Mimi decoder loaded")

        print("=" * 70)
        print("✅ STEP 8: OVIYA CSM-1B SERVER READY (REAL AUDIO)")
        print("=" * 70)
        print("   CSM-1B: Loaded")
        print("   Mimi Decoder: Loaded")
        print("   Device: CUDA")
        print("   Mode: Real high-quality speech")
        print("=" * 70)

    except Exception as e:
        print(f"❌ LIFESPAN ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Don't raise - let server start with degraded functionality
        print("⚠️ Server starting with limited functionality")

    print("🔄 About to yield - lifespan ready")
    yield  # Application runs here

    print("🔄 Lifespan cleanup")

app = FastAPI(title="Oviya CSM-1B Server (Real)", lifespan=lifespan)

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

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int = 24000
    duration_ms: float
    emotions: int = 49

# Global models
csm_model = None
processor = None
mimi_decoder = None

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for streaming"""
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

def generate_audio_real(text: str, emotion: str = "neutral") -> bytes:
    """Generate real audio using CSM-1B + Mimi"""
    global csm_model, processor, mimi_decoder
    
    try:
        # Format prompt with emotion
        prompt = f"[{emotion}] {text}"
        
        # Prepare inputs
        inputs = processor(prompt, return_tensors="pt")
        inputs = {k: v.to(csm_model.device) for k, v in inputs.items()}
        
        # Generate RVQ codes
        with torch.no_grad():
            outputs = csm_model.generate(**inputs, max_new_tokens=512)
            
            # Decode with Mimi
            codes = outputs.transpose(1, 2)  # [batch, frames, codebooks] -> [batch, codebooks, frames]
            decoder_output = mimi_decoder.decode(codes)
            
            # Extract audio tensor
            if hasattr(decoder_output, 'audio_values'):
                audio = decoder_output.audio_values
            elif hasattr(decoder_output, 'audio'):
                audio = decoder_output.audio
            elif isinstance(decoder_output, tuple):
                audio = decoder_output[0]
            else:
                audio = decoder_output
            
            # Ensure correct shape for saving: [channels, samples]
            if audio.dim() == 3:  # [batch, channels, samples]
                audio = audio.squeeze(0)  # [channels, samples]
            elif audio.dim() == 1:  # [samples]
                audio = audio.unsqueeze(0)  # [1, samples]
            
            # Save to bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio.cpu(), 24000, format="wav")
            buffer.seek(0)
            
            return buffer.read()
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise

async def stream_audio_chunks(text: str, emotion: str):
    """Stream audio in sentence chunks"""
    sentences = split_into_sentences(text)
    
    print(f"   Streaming {len(sentences)} sentences via REAL CSM-1B...")
    
    for i, sentence in enumerate(sentences):
        start = time.time()
        
        # Generate REAL audio
        audio_bytes = generate_audio_real(sentence, emotion)
        
        gen_time = time.time() - start
        
        print(f"   Chunk {i+1}/{len(sentences)}: Generated in {gen_time:.3f}s (REAL CSM-1B)")
        
        chunk_data = {
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
            "chunk_index": i,
            "total_chunks": len(sentences),
            "sample_rate": 24000,
            "quality": "csm_1b_mimi"
        }
        
        yield f"data: {base64.b64encode(str(chunk_data).encode()).decode()}\n\n"
        await asyncio.sleep(0.01)


@app.post("/generate/stream")
async def generate_audio_streaming(request: TTSRequest):
    print(f"🎵 Streaming REAL audio: '{request.text[:50]}...'")
    
    return StreamingResponse(
        stream_audio_chunks(request.text, request.reference_emotion),
        media_type="text/event-stream"
    )

@app.post("/generate", response_model=TTSResponse)
async def generate_audio(request: TTSRequest):
    try:
        start_time = time.time()
        
        print(f"🎵 Generating REAL audio: '{request.text[:50]}...'")
        
        # Generate with real CSM-1B + Mimi
        audio_bytes = generate_audio_real(request.text, request.reference_emotion)
        
        generation_time = time.time() - start_time
        
        # Decode to get duration
        buffer = io.BytesIO(audio_bytes)
        audio_tensor, sample_rate = torchaudio.load(buffer)
        duration_ms = (audio_tensor.shape[1] / sample_rate) * 1000
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"   ✅ {duration_ms:.0f}ms in {generation_time:.3f}s (REAL CSM-1B)")
        
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
        "service": "Oviya CSM-1B (Real)",
        "mode": "csm_1b_mimi",
        "quality": "high_fidelity",
        "streaming": True,
        "models": {
            "csm": "sesame/csm-1b",
            "decoder": "kyutai/mimi"
        }
    }

@app.get("/")
async def root():
    return {
        "service": "Oviya CSM-1B Server (Real Audio)",
        "mode": "csm_1b_mimi",
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
