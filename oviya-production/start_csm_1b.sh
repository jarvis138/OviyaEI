#!/bin/bash
# Start the real CSM-1B server

echo "=" * 70)
echo "🚀 Starting Oviya CSM-1B Server (Real)"
echo "="================================================================

# Set environment variables for HuggingFace cache
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Change to workspace directory
cd /workspace/oviya-production

# Check if model exists
if [ ! -d "/workspace/.cache/huggingface" ]; then
    echo "❌ CSM-1B model not found in /workspace/.cache/huggingface"
    echo "   Please download the model first"
    exit 1
fi

echo "✅ Model cache found"
echo "📍 Working directory: $(pwd)"
echo "🎯 Starting server on port 19517..."
echo ""

# Start server in background with logging
nohup python3 csm_server_real.py > /workspace/csm_1b.log 2>&1 &

SERVER_PID=$!
echo "   PID: $SERVER_PID"
echo "   Log: /workspace/csm_1b.log"

# Wait for server to start
echo ""
echo "⏳ Waiting for server to initialize..."
sleep 5

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server is running!"
    echo ""
    
    # Test health endpoint
    echo "🧪 Testing health endpoint..."
    sleep 2
    
    HEALTH=$(curl -s http://localhost:19517/health)
    if [ $? -eq 0 ]; then
        echo "✅ Health check passed:"
        echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
    else
        echo "⚠️  Health check failed (server may still be loading)"
    fi
    
    echo ""
    echo "=" * 70)
    echo "🎉 CSM-1B Server Started Successfully!"
    echo "=" * 70)
    echo "Server URL: http://localhost:19517"
    echo "Health: http://localhost:19517/health"
    echo "Logs: tail -f /workspace/csm_1b.log"
    echo "Stop: ./stop_csm.sh"
    echo "=" * 70)
else
    echo "❌ Server failed to start"
    echo "Check logs: tail -f /workspace/csm_1b.log"
    exit 1
fi

