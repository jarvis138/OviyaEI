#!/bin/bash

echo "═══════════════════════════════════════════════════════════════════════════"
echo "🚀 DEPLOYING REAL CSM-1B SERVER"
echo "═══════════════════════════════════════════════════════════════════════════"

# Stop old server
echo "⏹️  Stopping old server..."
pkill -f "csm_server"

# Install torchaudio if needed
echo "📦 Checking dependencies..."
pip3 show torchaudio > /dev/null 2>&1 || pip3 install torchaudio

# Start new server
echo "🚀 Starting REAL CSM-1B server..."
cd /workspace/oviya-production
nohup python3 csm_server_real.py > /tmp/csm_real.log 2>&1 &

echo "   PID: $!"
sleep 5

# Test health
echo ""
echo "🔍 Testing server..."
curl -s http://localhost:19517/health | python3 -m json.tool

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "✅ REAL CSM-1B SERVER DEPLOYED!"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "📋 Next steps:"
echo "   1. Expose via ngrok: ngrok http 19517"
echo "   2. Update frontend CSM_URL to ngrok URL"
echo "   3. Test: curl http://localhost:19517/health"
echo "   4. View logs: tail -f /tmp/csm_real.log"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
