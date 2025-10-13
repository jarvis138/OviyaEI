# Cloudflare Tunnel Setup Complete

## ✅ Current Status

### Ollama Tunnel (WORKING)
```
URL: https://prime-show-visit-lock.trycloudflare.com
Port: 11434
Status: ✅ ACTIVE
Test: curl https://prime-show-visit-lock.trycloudflare.com/api/tags
```

### CSM Tunnel (NEEDS UPDATE)
```
Current Port Tunnel: 8000 (WRONG)
Actual CSM Port: 19517
Status: ⏳ PENDING
```

## 🔧 Action Required

### On Vast.ai Instance

1. **Stop the wrong tunnel** (if still running):
   ```bash
   # Find and kill the process on port 8000
   ps aux | grep cloudflared
   kill <PID>  # Kill the one tunneling port 8000
   ```

2. **Create tunnel for correct CSM port** (19517):
   ```bash
   cloudflared tunnel --url http://localhost:19517
   ```

3. **Copy the generated URL** - you'll see output like:
   ```
   Your quick Tunnel has been created! Visit it at:
   https://something-random.trycloudflare.com
   ```

4. **Test the CSM tunnel**:
   ```bash
   curl https://something-random.trycloudflare.com/health
   # Should return: {"status": "healthy"}
   ```

### On Local Machine (This Machine)

5. **Update the CSM URL** in these files:

   Replace `YOUR-CSM-TUNNEL` with your actual tunnel URL in:
   
   - `config/service_urls.py`
   - `realtime_conversation.py`
   - `test_realtime_system.py`
   
   Example:
   ```python
   # Before
   csm_url = "https://YOUR-CSM-TUNNEL.trycloudflare.com/generate"
   
   # After (with your actual URL)
   csm_url = "https://something-random.trycloudflare.com/generate"
   ```

6. **Run tests**:
   ```bash
   python3 test_realtime_system.py
   ```

## 📋 Files Updated

### ✅ Already Updated with Ollama URL
- `realtime_conversation.py` - Updated to Cloudflare tunnel
- `test_realtime_system.py` - Updated to Cloudflare tunnel
- `config/service_urls.py` - New centralized config file

### ⏳ Pending CSM URL Update
These files still have placeholder `YOUR-CSM-TUNNEL`:
- `realtime_conversation.py` (line 31)
- `test_realtime_system.py` (line 25)
- `config/service_urls.py` (line 8)

### 📝 Other Files to Update Later
Once you have the CSM tunnel URL, also update:
- `pipeline.py`
- `test_diverse_scenarios.py`
- `test_llm_prosody_5.py`
- `test_5_scenarios.py`
- `test_all_enhancements.py`
- `production_sanity_tests.py`

## 🎯 Benefits of Cloudflare Tunnels

### vs localhost.run
- ✅ More stable (doesn't expire as frequently)
- ✅ Better performance (Cloudflare CDN)
- ✅ Free and unlimited bandwidth
- ✅ DDoS protection included

### vs ngrok
- ✅ No account required for testing
- ✅ Completely free (no bandwidth limits)
- ✅ Better global performance
- ✅ No time limits

## 🔄 Keep Tunnels Running

### Run in Background
```bash
# On Vast.ai instance
nohup cloudflared tunnel --url http://localhost:11434 > /tmp/ollama-tunnel.log 2>&1 &
nohup cloudflared tunnel --url http://localhost:19517 > /tmp/csm-tunnel.log 2>&1 &

# Check tunnel URLs anytime
grep "trycloudflare.com" /tmp/ollama-tunnel.log
grep "trycloudflare.com" /tmp/csm-tunnel.log

# Check if tunnels are running
ps aux | grep cloudflared
```

### View Logs
```bash
tail -f /tmp/ollama-tunnel.log
tail -f /tmp/csm-tunnel.log
```

## 🧪 Testing

### Test Ollama (Already Working)
```bash
# List models
curl https://prime-show-visit-lock.trycloudflare.com/api/tags

# Generate response
curl -X POST https://prime-show-visit-lock.trycloudflare.com/api/generate \
  -d '{"model":"qwen2.5:7b","prompt":"Hello","stream":false}'
```

### Test CSM (After Creating Tunnel)
```bash
# Health check
curl https://YOUR-TUNNEL.trycloudflare.com/health

# Generate voice
curl -X POST https://YOUR-TUNNEL.trycloudflare.com/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test",
    "emotion_params": {
      "style_token": "neutral",
      "pitch_scale": 1.0,
      "rate_scale": 1.0,
      "energy_scale": 1.0
    }
  }' --output test.wav
```

## 📊 Tunnel Status

| Service | Port | Tunnel URL | Status |
|---------|------|------------|--------|
| Ollama | 11434 | `prime-show-visit-lock.trycloudflare.com` | ✅ Active |
| CSM | 19517 | `YOUR-CSM-TUNNEL.trycloudflare.com` | ⏳ Pending |

## 🚀 Next Steps

1. ✅ Ollama tunnel is working
2. ⏳ Create CSM tunnel on port 19517
3. ⏳ Update CSM URL in all files
4. ⏳ Test complete system
5. ⏳ Run background tunnels for stability

---

**Last Updated**: October 13, 2025  
**Ollama Tunnel**: ✅ WORKING  
**CSM Tunnel**: ⏳ WAITING FOR CREATION


