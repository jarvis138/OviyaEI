#!/bin/bash
###############################################################################
# Deploy CSM-1B RVQ Streaming to Vast.ai
# Based on Sesame's "Crossing the uncanny valley" paper
###############################################################################

echo "═══════════════════════════════════════════════════════════════════════════"
echo "🚀 DEPLOYING CSM-1B RVQ STREAMING SERVER"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "   Based on: Sesame's Conversational Voice paper"
echo "   Architecture: Backbone (1B) + Decoder (100M) + Mimi"
echo "   Streaming: RVQ-level (12.5 Hz, 80ms frames)"
echo "   Latency: ~160ms first audio"
echo ""

# Stop old servers
echo "⏹️  Stopping old CSM servers..."
pkill -f "csm_server"
sleep 2

# Create voice directory if needed
mkdir -p /workspace/oviya-production/voice

# Copy new files (assumes they're already uploaded)
echo "📦 Checking files..."
if [ ! -f "/workspace/oviya-production/voice/csm_1b_stream.py" ]; then
    echo "❌ csm_1b_stream.py not found!"
    echo "   Please upload it first:"
    echo "   scp voice/csm_1b_stream.py root@vast.ai:/workspace/oviya-production/voice/"
    exit 1
fi

if [ ! -f "/workspace/oviya-production/csm_server_real_rvq.py" ]; then
    echo "❌ csm_server_real_rvq.py not found!"
    echo "   Please upload it first:"
    echo "   scp csm_server_real_rvq.py root@vast.ai:/workspace/oviya-production/"
    exit 1
fi

echo "✅ Files found"

# Make executable
chmod +x /workspace/oviya-production/csm_server_real_rvq.py

# Check disk space
echo ""
echo "💾 Checking disk space..."
df -h /workspace | tail -1

# Install any missing dependencies
echo ""
echo "📦 Checking dependencies..."
pip3 show torchaudio > /dev/null 2>&1 || pip3 install torchaudio
pip3 show transformers > /dev/null 2>&1 || pip3 install transformers --upgrade

# Start server
echo ""
echo "🚀 Starting RVQ streaming server..."
cd /workspace/oviya-production

nohup python3 csm_server_real_rvq.py > /tmp/csm_rvq.log 2>&1 &
SERVER_PID=$!

echo "   PID: $SERVER_PID"
echo "   Log: /tmp/csm_rvq.log"

# Wait for startup
echo ""
echo "⏳ Waiting for server to initialize..."
echo "   (Loading CSM-1B + Mimi takes ~10-15 seconds)"

for i in {1..30}; do
    sleep 1
    if grep -q "SERVER READY" /tmp/csm_rvq.log 2>/dev/null; then
        echo "   ✅ Server ready!"
        break
    fi
    echo -n "."
done
echo ""

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "❌ Server failed to start!"
    echo ""
    echo "Logs:"
    tail -50 /tmp/csm_rvq.log
    exit 1
fi

# Test health endpoint
echo ""
echo "🔍 Testing health endpoint..."
sleep 5

HEALTH_RESPONSE=$(curl -s http://localhost:19517/health)
if [ $? -eq 0 ]; then
    echo "✅ Health check passed"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    echo "❌ Health check failed"
    echo "   Server may still be initializing..."
fi

# Test audio generation
echo ""
echo "🧪 Testing audio generation..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:19517/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! This is a test of RVQ streaming.", "reference_emotion": "joyful"}' \
  --max-time 30)

if echo "$TEST_RESPONSE" | grep -q "audio_base64"; then
    DURATION=$(echo "$TEST_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['duration_ms'])" 2>/dev/null)
    echo "✅ Audio generation working!"
    echo "   Duration: ${DURATION}ms"
else
    echo "⚠️  Audio test failed or timed out"
    echo "   Response: $TEST_RESPONSE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Server Status:"
echo "   PID: $SERVER_PID"
echo "   Port: 19517"
echo "   Log: /tmp/csm_rvq.log"
echo ""
echo "📋 Next Steps:"
echo "   1. Expose via Cloudflare tunnel:"
echo "      cloudflared tunnel --url http://localhost:19517 > /tmp/cloudflare_csm_rvq.log 2>&1 &"
echo "      grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cloudflare_csm_rvq.log | head -1"
echo ""
echo "   2. Monitor logs:"
echo "      tail -f /tmp/csm_rvq.log"
echo ""
echo "   3. Test streaming endpoint:"
echo "      curl -N -X POST http://localhost:19517/generate/stream \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"text\": \"Hello world\", \"stream\": true}'"
echo ""
echo "   4. Update frontend config with new Cloudflare URL"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "🎯 Performance Metrics:"
echo "   - RVQ frame rate: 12.5 Hz (80ms per frame)"
echo "   - Flush interval: 2 frames (160ms chunks)"
echo "   - Expected first audio: ~160ms"
echo "   - Quality: 24kHz, 32 codebooks"
echo ""
echo "📚 Architecture (from paper):"
echo "   - Backbone: 1B params (zeroth codebook)"
echo "   - Decoder: 100M params (31 acoustic codebooks)"
echo "   - Codec: Mimi (split-RVQ)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"

