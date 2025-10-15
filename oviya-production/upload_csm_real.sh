#!/bin/bash
# This script creates the real CSM server on Vast.ai

cat > /workspace/oviya-production/csm_server_real.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
REAL CSM-1B Server with Mimi Decoder
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
from transformers import CsmForConditionalGeneration, AutoProcessor, MimiModel
import numpy as np
import base64
import io
import asyncio
from typing import Optional, List, Dict
import time
import re
import torchaudio

app = FastAPI(title="Oviya CSM-1B Server (Real)")

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

csm_model = None
processor = None
mimi_decoder = None

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

def generate_audio_real(text: str, emotion: str = "neutral") -> bytes:
    global csm_model, processor, mimi_decoder
    try:
        prompt = f"[{emotion}] {text}"
        inputs = processor(prompt, return_tensors="pt")
        inputs = {k: v.to(csm_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = csm_model.generate(**inputs, max_new_tokens=512)
            codes = outputs.transpose(1, 2)
            decoder_output = mimi_decoder.decode(codes)
            
            if hasattr(decoder_output, 'audio_values'):
                audio = decoder_output.audio_values
            elif hasattr(decoder_output, 'audio'):
                audio = decoder_output.audio
            elif isinstance(decoder_output, tuple):
                audio = decoder_output[0]
            else:
                audio = decoder_output
            
            if audio.dim() == 3:
                audio = audio.squeeze(0)
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio.cpu(), 24000, format="wav")
            buffer.seek(0)
            return buffer.read()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise

async def stream_audio_chunks(text: str, emotion: str):
    sentences = split_into_sentences(text)
    print(f"Streaming {len(sentences)} sentences...")
    
    for i, sentence in enumerate(sentences):
        start = time.time()
        audio_bytes = generate_audio_real(sentence, emotion)
        gen_time = time.time() - start
        print(f"Chunk {i+1}/{len(sentences)}: {gen_time:.3f}s")
        
        chunk_data = {
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
            "chunk_index": i,
            "total_chunks": len(sentences),
            "sample_rate": 24000,
            "quality": "csm_1b_mimi"
        }
        yield f"data: {base64.b64encode(str(chunk_data).encode()).decode()}\n\n"
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup():
    global csm_model, processor, mimi_decoder
    print("="*70)
    print("LOADING CSM-1B + MIMI")
    print("="*70)
    
    processor = AutoProcessor.from_pretrained("sesame/csm-1b", cache_dir="/workspace/.cache/huggingface")
    csm_model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", cache_dir="/workspace/.cache/huggingface", device_map="auto", torch_dtype=torch.float16)
    csm_model.eval()
    mimi_decoder = MimiModel.from_pretrained("kyutai/mimi", cache_dir="/workspace/.cache/huggingface").to("cuda").eval()
    
    print("READY!")
    print("="*70)

@app.post("/generate/stream")
async def generate_audio_streaming(request: TTSRequest):
    print(f"Streaming: {request.text[:50]}...")
    return StreamingResponse(stream_audio_chunks(request.text, request.reference_emotion), media_type="text/event-stream")

@app.post("/generate", response_model=TTSResponse)
async def generate_audio(request: TTSRequest):
    try:
        start_time = time.time()
        print(f"Generating: {request.text[:50]}...")
        
        audio_bytes = generate_audio_real(request.text, request.reference_emotion)
        generation_time = time.time() - start_time
        
        buffer = io.BytesIO(audio_bytes)
        audio_tensor, sample_rate = torchaudio.load(buffer)
        duration_ms = (audio_tensor.shape[1] / sample_rate) * 1000
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"Done: {duration_ms:.0f}ms in {generation_time:.3f}s")
        return TTSResponse(audio_base64=audio_base64, sample_rate=24000, duration_ms=duration_ms, emotions=49)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Oviya CSM-1B Real", "mode": "csm_1b_mimi"}

@app.get("/")
async def root():
    return {"service": "Oviya CSM-1B Server", "endpoints": {"/generate": "POST", "/generate/stream": "POST", "/health": "GET"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=19517, log_level="info")
EOFPYTHON

chmod +x /workspace/oviya-production/csm_server_real.py

echo "File created!"
ls -lh /workspace/oviya-production/csm_server_real.py
