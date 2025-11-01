#!/bin/bash
echo "🚀 OVIYA EI COMPLETE VAST.AI DEPLOYMENT"
echo "========================================"

# Exit on any error
set -e

# System setup
echo "📦 Installing system dependencies..."
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git curl wget htop python3-dev build-essential

# Install Ollama
echo "🤖 Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Clone repository
echo "📥 Cloning Oviya EI repository..."
git clone https://github.com/jarvis138/OviyaEI.git
cd OviyaEI/production

# Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Environment variables
echo "🔐 Configuring environment..."
export HUGGINGFACE_TOKEN="[REDACTED_TOKEN]"
export OVIYA_SECRET="oviya_secure_secret_2024"
export OVIYA_ENV="production"
export CLOUD_GPU_AVAILABLE="true"

# Start Ollama service
echo "🧠 Starting Ollama service..."
ollama serve &
sleep 15

# Pull model
echo "📥 Downloading Llama model..."
ollama pull llama3.2:3b

# Test systems
echo "🧪 Testing brain system..."
python3 -c "
try:
    from brain.llm_brain import OviyaBrain
    brain = OviyaBrain()
    response = brain.think('Hello, deployment test')
    print('✅ Brain system working')
    print(f'Response keys: {list(response.keys())}')
except Exception as e:
    print(f'❌ Brain test failed: {e}')
"

echo "🎵 Testing voice system..."
python3 -c "
try:
    from voice.csm_1b_generator_optimized import get_optimized_streamer
    streamer = get_optimized_streamer()
    audio = streamer.generate_voice('Voice deployment test', speaker_id=42)
    print(f'✅ Voice system working: {len(audio)} bytes')
except Exception as e:
    print(f'❌ Voice test failed: {e}')
"

# Start services
echo "🚀 Starting Oviya services..."
nohup python3 websocket_server.py > websocket.log 2>&1 &

# Health check
sleep 10
if curl -f http://localhost:8000/health; then
    echo "✅ WebSocket server is healthy!"
else
    echo "⚠️ WebSocket server may not be ready yet"
fi

echo ""
echo "�� DEPLOYMENT COMPLETED!"
echo "========================"
echo "🌐 WebSocket Server: http://localhost:8000"
echo "🔍 Health Check: curl http://localhost:8000/health"
echo "📋 Logs: tail -f websocket.log"
echo "🧪 Test: python3 test_complete_system_integration.py"
echo ""
echo "To connect from your local machine:"
echo "ssh -p 19199 root@175.155.64.172 -L 8080:localhost:8080"
echo "Then visit: http://localhost:8080"
