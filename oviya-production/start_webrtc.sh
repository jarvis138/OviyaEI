#!/bin/bash
# Quick Start Script for Oviya WebRTC Voice Mode

echo "════════════════════════════════════════════════════════════════════"
echo "🎤 OVIYA WEBRTC VOICE MODE - QUICK START"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check if we're in the right directory
if [ ! -f "voice_server_webrtc.py" ]; then
    echo "❌ Error: voice_server_webrtc.py not found"
    echo "   Please run this script from /oviya-production directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "   Please install Python 3.9+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check dependencies
echo "🔍 Checking dependencies..."
python3 -c "import aiortc" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  aiortc not found. Installing dependencies..."
    pip3 install -r requirements_webrtc.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ Dependencies installed"
fi

echo ""
echo "🔗 Checking Vast.ai services..."
echo ""

# Check WhisperX
echo -n "   WhisperX... "
curl -s --max-time 5 https://msgid-enquiries-williams-lands.trycloudflare.com/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅"
else
    echo "⚠️  (may be down)"
fi

# Check Ollama
echo -n "   Ollama...   "
curl -s --max-time 5 https://prime-show-visit-lock.trycloudflare.com/api/tags > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅"
else
    echo "⚠️  (may be down)"
fi

# Check CSM
echo -n "   CSM TTS...  "
curl -s --max-time 5 https://astronomy-initiative-paso-cream.trycloudflare.com/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅"
else
    echo "⚠️  (may be down)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "🚀 STARTING WEBRTC SERVER..."
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Server will be available at:"
echo "   🌐 http://localhost:8000"
echo ""
echo "WebRTC client interface:"
echo "   🎤 http://localhost:8000/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Start server
python3 voice_server_webrtc.py

