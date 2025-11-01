#!/bin/bash
# 🧪 Test Oviya EI Deployment Locally

echo "🧪 Testing Oviya EI Deployment"
echo "=============================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test functions
test_health() {
    echo "Testing $1..."
    if curl -f --max-time 10 "$2" &>/dev/null; then
        echo -e "${GREEN}✅ $1: OK${NC}"
        return 0
    else
        echo -e "${RED}❌ $1: FAILED${NC}"
        return 1
    fi
}

echo "🔍 Checking services..."

# Test Ollama
test_health "Ollama" "http://localhost:11434/api/tags"

# Test CSM Voice
test_health "CSM Voice" "http://localhost:19517/health"

# Test Oviya Backend
test_health "Oviya Backend" "http://localhost:8000/health"

# Test Oviya API
test_health "Oviya API" "http://localhost:8080/healthz"

echo ""
echo "📊 System Resources:"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "No GPU detected"

echo ""
echo "Memory:"
free -h | grep "^Mem:"

echo ""
echo "Disk:"
df -h / | tail -1

echo ""
echo "🧵 Running Processes:"
ps aux | grep -E "(ollama|csm|oviya|uvicorn)" | grep -v grep | head -10
