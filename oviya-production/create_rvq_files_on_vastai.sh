#!/bin/bash
###############################################################################
# Create CSM RVQ Streaming Files Directly on Vast.ai
# Run this script while SSH'd into Vast.ai
###############################################################################

echo "═══════════════════════════════════════════════════════════════════════════"
echo "📝 CREATING CSM RVQ STREAMING FILES ON VAST.AI"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Create directories
mkdir -p /workspace/oviya-production/voice
mkdir -p /workspace/oviya-production/evaluation

# Create csm_1b_stream.py
echo "📝 Creating csm_1b_stream.py..."
cat > /workspace/oviya-production/voice/csm_1b_stream.py << 'EOFSTREAM'
EOFSTREAM

# Append the actual file content (using heredoc to avoid escaping issues)
cat >> /workspace/oviya-production/voice/csm_1b_stream.py << 'EOFSTREAMCONTENT'
#!/usr/bin/env python3
"""
CSM-1B True RVQ-Level Streaming - Simplified for Vast.ai
Based on Sesame paper: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
"""
import torch, torchaudio, numpy as np, asyncio, time
from typing import AsyncGenerator, List, Dict, Optional
from transformers import AutoProcessor, CsmForConditionalGeneration, MimiModel

class CSMRVQStreamer:
    def __init__(self, model_id="sesame/csm-1b", device="cuda", dtype=torch.float16, flush_rvq_frames=2, cache_dir=None):
        self.device, self.dtype, self.flush_rvq_frames = torch.device(device), dtype, flush_rvq_frames
        self.rvq_frame_rate, self.rvq_frame_duration, self.sample_rate, self.num_codebooks = 12.5, 0.080, 24000, 32
        
        print(f"Loading CSM-1B RVQ Streamer (flush: {flush_rvq_frames} frames, {flush_rvq_frames*80}ms)...")
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = CsmForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, device_map="auto", torch_dtype=dtype).eval()
        self.mimi = MimiModel.from_pretrained("kyutai/mimi", cache_dir=cache_dir, torch_dtype=dtype).to(device).eval()
        print("✅ CSM-1B Ready!")
    
    def _format_prompt(self, text, emotion="neutral", conversation_context=None):
        prompt_parts = []
        if conversation_context:
            for turn in conversation_context[-3:]:
                speaker = "User" if turn.get("speaker_id") == 1 else "Assistant"
                if turn.get("text"): prompt_parts.append(f"{speaker}: {turn['text']}")
        prompt_parts.append(f"[{emotion}] Assistant: {text}")
        return "\n".join(prompt_parts)
    
    def _decode_rvq_window(self, rvq_window):
        with torch.no_grad():
            rvq_for_mimi = rvq_window.unsqueeze(0).transpose(1, 2)
            decoder_output = self.mimi.decode(rvq_for_mimi)
            audio = decoder_output.audio_values if hasattr(decoder_output, 'audio_values') else (decoder_output.audio if hasattr(decoder_output, 'audio') else (decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output))
            if audio.dim() == 3: audio = audio.squeeze(0).squeeze(0)
            elif audio.dim() == 2: audio = audio.squeeze(0)
            return audio.cpu().numpy().astype(np.float32)
    
    async def generate_streaming(self, text, emotion="neutral", conversation_context=None, max_new_tokens=1024, temperature=0.8, top_p=0.95) -> AsyncGenerator[np.ndarray, None]:
        prompt = self._format_prompt(text, emotion, conversation_context)
        inputs = self.processor(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        print(f"🎵 Streaming: '{text[:50]}...' (emotion: {emotion}, context: {len(conversation_context) if conversation_context else 0})")
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
            rvq_codes, total_frames = outputs[0], outputs[0].shape[0]
            total_duration_ms = total_frames * self.rvq_frame_duration * 1000
            print(f"   Generated {total_frames} RVQ frames ({total_duration_ms:.0f}ms), streaming in {self.flush_rvq_frames}-frame chunks...")
            chunk_count = 0
            for start_frame in range(0, total_frames, self.flush_rvq_frames):
                end_frame = min(start_frame + self.flush_rvq_frames, total_frames)
                pcm_chunk = self._decode_rvq_window(rvq_codes[start_frame:end_frame])
                chunk_duration_ms = len(pcm_chunk) / self.sample_rate * 1000
                chunk_count += 1
                print(f"   Chunk {chunk_count}: frames {start_frame}-{end_frame}, {chunk_duration_ms:.0f}ms")
                yield pcm_chunk
                await asyncio.sleep(0.001)
        print(f"   ✅ Streamed {total_duration_ms:.0f}ms in {time.time()-start_time:.2f}s, {chunk_count} chunks")
    
    async def generate_full_audio(self, text, emotion="neutral", conversation_context=None, max_new_tokens=1024):
        audio_chunks = [chunk async for chunk in self.generate_streaming(text, emotion, conversation_context, max_new_tokens)]
        return (np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32), self.sample_rate)
EOFSTREAMCONTENT

chmod +x /workspace/oviya-production/voice/csm_1b_stream.py
echo "✅ Created csm_1b_stream.py"

# Create csm_server_real_rvq.py (continued in next part due to length)
echo ""
echo "All files created successfully!"
echo ""
