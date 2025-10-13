#!/bin/bash

# Script to test both Cloudflare tunnels
# Usage: ./test_tunnels.sh [csm-tunnel-url]

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          CLOUDFLARE TUNNEL TEST                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Test Ollama Tunnel
echo "🧪 Testing Ollama Tunnel..."
echo "URL: https://prime-show-visit-lock.trycloudflare.com"
echo ""

OLLAMA_RESPONSE=$(curl -s https://prime-show-visit-lock.trycloudflare.com/api/tags 2>&1)

if echo "$OLLAMA_RESPONSE" | grep -q "qwen2.5:7b"; then
    echo "✅ Ollama Tunnel: WORKING"
    echo "   Model: qwen2.5:7b available"
else
    echo "❌ Ollama Tunnel: FAILED"
    echo "   Response: $OLLAMA_RESPONSE"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test CSM Tunnel
if [ -z "$1" ]; then
    echo "⏳ CSM Tunnel: NOT CONFIGURED"
    echo "   Run: ./test_tunnels.sh https://your-csm-tunnel.trycloudflare.com"
else
    CSM_URL="$1"
    echo "🧪 Testing CSM Tunnel..."
    echo "URL: $CSM_URL"
    echo ""
    
    CSM_RESPONSE=$(curl -s "$CSM_URL/health" 2>&1)
    
    if echo "$CSM_RESPONSE" | grep -q "healthy"; then
        echo "✅ CSM Tunnel: WORKING"
        echo "   Status: healthy"
    else
        echo "❌ CSM Tunnel: FAILED"
        echo "   Response: $CSM_RESPONSE"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Test complete!                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"


