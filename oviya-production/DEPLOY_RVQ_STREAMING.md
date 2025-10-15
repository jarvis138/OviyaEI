# 🚀 Deploy CSM-1B RVQ Streaming to Vast.ai

## Based on Sesame's Paper
**"Crossing the uncanny valley of conversational voice"**
https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice

## 📊 Architecture Overview

### From the Paper:
> "CSM is a multimodal, text and speech model that operates directly on RVQ tokens. We split the transformers at the zeroth codebook. The first multimodal backbone processes interleaved text and audio to model the zeroth codebook. The second audio decoder uses a distinct linear head for each codebook and models the remaining N – 1 codebooks."

```
┌────────────────────────────────────────────────────────────┐
│                 CSM-1B Architecture                         │
├────────────────────────────────────────────────────────────┤
│  Backbone (1B params)                                       │
│  ↓                                                          │
│  Zeroth Codebook (semantic + prosody)                       │
│  ↓                                                          │
│  Decoder (100M params)                                      │
│  ↓                                                          │
│  31 Acoustic Codebooks                                      │
│  ↓                                                          │
│  Mimi Codec (RVQ → PCM)                                    │
│  ↓                                                          │
│  24kHz Audio                                                │
└────────────────────────────────────────────────────────────┘
```

### Key Specs:
- **RVQ Frame Rate**: 12.5 Hz (80ms per frame)
- **Flush Interval**: 2-4 frames (160-320ms chunks)
- **Latency Target**: ~160ms first audio
- **Codebooks**: 32 total (1 semantic + 31 acoustic)

## 🎯 Step-by-Step Deployment

### 1. Upload Files to Vast.ai

```bash
# From your Mac
cd "/Users/jarvis/Documents/Oviya EI/oviya-production"

# Upload streaming implementation
scp voice/csm_1b_stream.py root@<vast-ip>:/workspace/oviya-production/voice/

# Upload server
scp csm_server_real_rvq.py root@<vast-ip>:/workspace/oviya-production/

# Upload deployment script
scp deploy_rvq_streaming.sh root@<vast-ip>:/workspace/oviya-production/
```

### 2. Run Deployment Script

```bash
# SSH into Vast.ai
ssh root@<vast-ip> -p <port>

# Run deployment
cd /workspace/oviya-production
bash deploy_rvq_streaming.sh
```

### 3. Expose via Cloudflare Tunnel

```bash
# Kill old tunnels
pkill cloudflared

# Start new tunnel
cloudflared tunnel --url http://localhost:19517 > /tmp/cloudflare_csm_rvq.log 2>&1 &

# Get URL
sleep 5
grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cloudflare_csm_rvq.log | head -1
```

### 4. Test RVQ Streaming

```bash
# Test non-streaming endpoint (baseline)
curl -X POST http://localhost:19517/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! Testing RVQ streaming.", "reference_emotion": "joyful"}'

# Test streaming endpoint (RVQ-level)
curl -N -X POST http://localhost:19517/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! Testing RVQ streaming.", "reference_emotion": "joyful", "stream": true}'
```

## 📈 Expected Performance

### Paper Findings:
- **Without Context**: CSM matches human naturalness
- **With Context**: Gap remains in prosodic appropriateness
- **WER**: Near-human performance
- **Novel Tests**: Homograph disambiguation, pronunciation consistency

### Our Implementation:
- **Streaming Latency**: ~160ms first audio (2 RVQ frames)
- **Chunk Size**: 160ms (optimal per paper)
- **Real-time Factor**: Should be > 1.0x (faster than real-time)
- **Quality**: 24kHz, 32-codebook fidelity

## 🧪 Testing Checklist

- [ ] Server starts successfully
- [ ] Health endpoint returns architecture details
- [ ] Audio generation works (non-streaming)
- [ ] Streaming endpoint yields chunks
- [ ] Chunks arrive every ~160ms
- [ ] Audio quality is high (no noise/artifacts)
- [ ] Conversational context improves prosody
- [ ] Emotion tags affect delivery

## 📝 Monitoring

### View Logs
```bash
tail -f /tmp/csm_rvq.log
```

### Check GPU Usage
```bash
nvidia-smi -l 1
```

### Monitor Latency
```bash
# Watch for "Streamed Xms audio in Ys" messages
tail -f /tmp/csm_rvq.log | grep "Streamed"
```

## 🔄 Update Frontend Config

After deploying, update your frontend to use the new URL:

```python
# config/service_urls.py
CSM_URL = "https://<your-cloudflare-url>.trycloudflare.com/generate"
CSM_STREAM_URL = "https://<your-cloudflare-url>.trycloudflare.com/generate/stream"
```

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check logs
tail -50 /tmp/csm_rvq.log

# Check disk space
df -h /workspace

# Check VRAM
nvidia-smi
```

### Slow Generation
```bash
# Paper: "Compute amortization scheme alleviates memory bottleneck"
# Ensure flush_frames is set to 2-4 (not higher)
```

### Poor Audio Quality
```bash
# Verify Mimi decoder is loaded
grep "Loading Mimi decoder" /tmp/csm_rvq.log

# Check codebooks are correct (should be 32)
curl http://localhost:19517/health | grep num_codebooks
```

## 📚 Paper References

### Key Quotes:

**On Architecture:**
> "The decoder is significantly smaller than the backbone, enabling low-latency generation while keeping the model end-to-end."

**On Streaming:**
> "Mimi, a split-RVQ tokenizer, producing one semantic codebook and N – 1 acoustic codebooks per frame at 12.5 Hz."

**On Context:**
> "CSM leverages the history of the conversation to produce more natural and coherent speech."

**On Evaluation:**
> "When context is included, evaluators consistently favor the original recordings, suggesting a noticeable gap remains."

## 🎯 Next Steps

1. **Deploy** using the script above
2. **Verify** streaming latency (~160ms)
3. **Update** frontend config
4. **Test** from web interface
5. **Implement** DataChannel metadata (Phase 2)
6. **Run** evaluation harness (Phase 3)

---

**Status**: Ready for deployment 🚀
