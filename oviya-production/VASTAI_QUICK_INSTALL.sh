#!/bin/bash
###############################################################################
# Quick Install CSM-1B RVQ Streaming on Vast.ai
# Copy-paste this entire script while SSH'd into Vast.ai
###############################################################################

echo "Starting installation..."

# Stop old servers
pkill -f csm_server
sleep 2

# Create directories
mkdir -p /workspace/oviya-production/voice

# Create simplified streaming server
cat > /workspace/oviya-production/csm_stream_server.py << 'EOFSERVER'
#!/usr/bin/env python3
"""CSM-1B RVQ Streaming Server - Simplified"""
import sys, os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch, numpy as np, base64, io, asyncio, time, torchaudio, re
from typing import Optional, List, Dict
from transformers import AutoProcessor, CsmForConditionalGeneration, MimiModel

app = FastAPI(title="Oviya CSM-1B RVQ")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0
    reference_emotion: str = "neutral"
    normalize_audio: bool = True
    conversation_context: Optional[List[Dict]] = None
    stream: bool = True

csm_model, processor, mimi = None, None, None

def split_sentences(text):
    sentences = re.split(r'([.!?]+\s+)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        s = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        if s.strip(): result.append(s.strip())
    return result if result else [text]

@app.on_event("startup")
async def startup():
    global csm_model, processor, mimi
    print("="*70)
    print("Loading CSM-1B + Mimi (RVQ Streaming)...")
    print("="*70)
    processor = AutoProcessor.from_pretrained("sesame/csm-1b", cache_dir="/workspace/.cache/huggingface")
    csm_model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", cache_dir="/workspace/.cache/huggingface", device_map="auto", torch_dtype=torch.float16).eval()
    mimi = MimiModel.from_pretrained("kyutai/mimi", cache_dir="/workspace/.cache/huggingface", torch_dtype=torch.float16).to("cuda").eval()
    print("="*70)
    print("✅ CSM-1B RVQ Streaming Ready!")
    print("="*70)

@app.post("/generate/stream")
async def generate_streaming(request: TTSRequest):
    print(f"Streaming: '{request.text[:50]}...'")
    
    async def gen_chunks():
        sentences = split_sentences(request.text)
        for i, sentence in enumerate(sentences):
            prompt = f"[{request.reference_emotion}] {sentence}"
            inputs = processor(prompt, return_tensors="pt")
            inputs = {k: v.to(csm_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = csm_model.generate(**inputs, max_new_tokens=512)
                codes = outputs.transpose(1, 2)
                audio = mimi.decode(codes)
                
                if hasattr(audio, "audio_values"): audio = audio.audio_values
                elif hasattr(audio, "audio"): audio = audio.audio
                elif isinstance(audio, tuple): audio = audio[0]
                
                if audio.dim() == 3: audio = audio.squeeze(0).squeeze(0)
                elif audio.dim() == 2: audio = audio.squeeze(0)
                elif audio.dim() == 1: audio = audio.unsqueeze(0)
                
                buffer = io.BytesIO()
                torchaudio.save(buffer, audio.cpu() if audio.dim() == 1 else audio.cpu().unsqueeze(0), 24000, format="wav")
                buffer.seek(0)
                
                chunk_data = {
                    "audio_base64": base64.b64encode(buffer.read()).decode(),
                    "chunk_index": i,
                    "total_chunks": len(sentences),
                    "sample_rate": 24000
                }
                yield f"data: {base64.b64encode(str(chunk_data).encode()).decode()}\n\n"
                await asyncio.sleep(0.01)
    
    return StreamingResponse(gen_chunks(), media_type="text/event-stream")

@app.post("/generate")
async def generate(request: TTSRequest):
    try:
        print(f"Generating: '{request.text[:50]}...'")
        prompt = f"[{request.reference_emotion}] {request.text}"
        inputs = processor(prompt, return_tensors="pt")
        inputs = {k: v.to(csm_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = csm_model.generate(**inputs, max_new_tokens=512)
            codes = outputs.transpose(1, 2)
            audio = mimi.decode(codes)
            
            if hasattr(audio, "audio_values"): audio = audio.audio_values
            elif hasattr(audio, "audio"): audio = audio.audio
            elif isinstance(audio, tuple): audio = audio[0]
            
            if audio.dim() == 3: audio = audio.squeeze(0)
            elif audio.dim() == 1: audio = audio.unsqueeze(0)
            
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio.cpu(), 24000, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            duration_ms = audio.shape[-1] / 24000 * 1000
            print(f"   Generated {duration_ms:.0f}ms")
            
            return {
                "audio_base64": base64.b64encode(audio_bytes).decode(),
                "sample_rate": 24000,
                "duration_ms": duration_ms
            }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "CSM-1B RVQ Streaming",
        "architecture": "Backbone + Decoder + Mimi",
        "rvq_frame_rate_hz": 12.5,
        "streaming": True
    }

@app.get("/")
async def root():
    return {
        "service": "Oviya CSM-1B RVQ Streaming",
        "paper": "https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice",
        "endpoints": {
            "/generate": "POST - Generate complete audio",
            "/generate/stream": "POST - Stream audio chunks",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\nStarting server on port 19517...")
    uvicorn.run(app, host="0.0.0.0", port=19517, log_level="info")
EOFSERVER

chmod +x /workspace/oviya-production/csm_stream_server.py

# Start server
echo ""
echo "Starting CSM-1B RVQ Streaming server..."
cd /workspace/oviya-production
nohup python3 csm_stream_server.py > /tmp/csm_stream.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for initialization (takes ~15 seconds)..."

# Wait for startup
for i in {1..20}; do
    sleep 1
    if grep -q "CSM-1B RVQ Streaming Ready" /tmp/csm_stream.log 2>/dev/null; then
        echo "✅ Server ready!"
        break
    fi
    echo -n "."
done
echo ""

# Test health
sleep 3
echo ""
echo "Testing health endpoint..."
curl -s http://localhost:19517/health | python3 -m json.tool

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "✅ INSTALLATION COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Server: Running on port 19517"
echo "Logs: tail -f /tmp/csm_stream.log"
echo ""
echo "Next: Expose via Cloudflare tunnel"
echo "  pkill cloudflared"
echo "  cloudflared tunnel --url http://localhost:19517 > /tmp/cf.log 2>&1 &"
echo "  sleep 3"
echo "  grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf.log | head -1"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"

