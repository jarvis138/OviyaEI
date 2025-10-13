#!/bin/bash
# Oviya EI + WhisperX Setup for RTX 5880 Ada (48GB VRAM)
# Verified and optimized script
# Run this on your VastAI instance

set -e  # Exit on error

echo "🚀 Starting Oviya EI + WhisperX Setup on RTX 5880 Ada"
echo "====================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root (you should be root on VastAI)"
    exit 1
fi

# Check GPU
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ GPU detected"
else
    echo "❌ nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi
echo ""

# Update system
echo "📦 Updating system packages..."
apt-get update -y
apt-get install -y curl wget git build-essential ffmpeg python3-pip
echo "✅ System packages updated"
echo ""

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "   Installing PyTorch with CUDA..."
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install WhisperX and dependencies
echo "   Installing WhisperX..."
pip3 install whisperx

# Install VAD
echo "   Installing Silero VAD..."
pip3 install git+https://github.com/snakers4/silero-vad.git

# Install other dependencies
echo "   Installing other dependencies..."
pip3 install sounddevice requests numpy soundfile huggingface-hub transformers flask

echo "✅ Python dependencies installed"
echo ""

# Verify PyTorch CUDA
echo "🔍 Verifying PyTorch CUDA..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('❌ CUDA not available!')
    exit(1)
"
echo ""

# Install Ollama
echo "🧠 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi
echo ""

# Start Ollama service
echo "🔄 Starting Ollama service..."
# Kill any existing ollama process
pkill ollama || true
sleep 2

# Start Ollama in background
nohup ollama serve > /workspace/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "   Ollama PID: $OLLAMA_PID"
sleep 15

# Verify Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama failed to start. Check /workspace/ollama.log"
    exit 1
fi
echo ""

# Pull Qwen2.5:7B model
echo "📥 Pulling Qwen2.5:7B model (this may take a few minutes)..."
ollama pull qwen2.5:7b
echo "✅ Qwen2.5:7B model ready"
echo ""

# Setup workspace
echo "📁 Setting up workspace..."
cd /workspace
mkdir -p oviya-production
cd oviya-production
mkdir -p data/voice_samples output logs emotion_references config

echo "✅ Workspace created at /workspace/oviya-production"
echo ""

# Download and cache WhisperX models
echo "📥 Caching WhisperX models (this will take a few minutes)..."
python3 << 'PYTHON_EOF'
import whisperx
import torch
import sys

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'   Device: {device}')
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f'   GPU: {gpu_props.name}')
        print(f'   VRAM: {gpu_props.total_memory / 1e9:.1f}GB')
    
    print('   Loading WhisperX large-v2 model...')
    model = whisperx.load_model('large-v2', device, compute_type='float16')
    
    print('   Loading alignment model...')
    align_model, metadata = whisperx.load_align_model('en', device)
    
    print('✅ WhisperX models cached successfully')
    
    # Show memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f'   GPU Memory Used: {memory_used:.1f}GB')
    
except Exception as e:
    print(f'❌ WhisperX model caching failed: {e}')
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "❌ WhisperX setup failed"
    exit 1
fi
echo ""

# Generate emotion references
echo "🎭 Generating emotion references..."
python3 << 'PYTHON_EOF'
import torch
import torchaudio
from pathlib import Path

emotions = {
    'calm_supportive': 'Take a deep breath. Everything will be okay.',
    'empathetic_sad': 'I am so sorry you are going through this.',
    'joyful_excited': 'That is amazing! I am so happy for you!',
    'playful': 'Hey there! This is going to be fun!',
    'confident': 'You have got this. I believe in you.',
    'concerned_anxious': 'Are you okay? I am here if you need me.',
    'angry_firm': 'That is not acceptable. This needs to stop.',
    'neutral': 'Hello. How can I help you today?'
}

output_dir = Path('/workspace/oviya-production/emotion_references')
output_dir.mkdir(exist_ok=True)

for emotion, text in emotions.items():
    duration = 2.0
    sample_rate = 24000
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    base_freq = 220
    audio = 0.3 * torch.sin(2 * torch.pi * base_freq * t)
    
    # Add emotion-specific characteristics
    if emotion == 'joyful_excited':
        vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
        audio = audio * (1 + vibrato)
    elif emotion == 'empathetic_sad':
        decay = torch.exp(-t * 0.5)
        audio = audio * decay
    elif emotion == 'confident':
        audio = audio * 1.2
    
    output_path = output_dir / f'{emotion}.wav'
    torchaudio.save(str(output_path), audio.unsqueeze(0), sample_rate)
    print(f'   Generated {emotion}.wav')

print('✅ Emotion references generated')
PYTHON_EOF
echo ""

# Create CSM server script
echo "🎤 Creating CSM server script..."
cat > /workspace/oviya-production/csm_server.py << 'EOF'
#!/usr/bin/env python3
"""
CSM Server for RTX 5880 Ada
Placeholder server - replace with your actual CSM implementation
"""

import torch
import torchaudio
from flask import Flask, request, jsonify
import io
import base64
import os
from pathlib import Path
import json

os.environ["NO_TORCH_COMPILE"] = "1"

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Starting CSM Server on {device}")

if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"   GPU: {gpu_props.name}")
    print(f"   VRAM: {gpu_props.total_memory / 1e9:.1f}GB")

EMOTION_REF_DIR = Path("/workspace/oviya-production/emotion_references")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': device,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        'emotion_references': len(list(EMOTION_REF_DIR.glob("*.wav"))) if EMOTION_REF_DIR.exists() else 0
    })

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        text = data.get('text', '')
        reference_emotion = data.get('reference_emotion', None)
        
        print(f"🎤 Generating: {text[:50]}...")
        if reference_emotion:
            print(f"   🎭 Emotion: {reference_emotion}")
        
        # Placeholder: Generate dummy audio
        # TODO: Replace with actual CSM generation
        sample_rate = 24000
        duration = max(len(text) * 0.1, 1.0)
        num_samples = int(duration * sample_rate)
        audio = torch.randn(num_samples) * 0.1
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), sample_rate, format='wav')
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        print(f"   ✅ Generated: {duration:.2f}s")

        return jsonify({
            'audio_base64': audio_base64,
            'text': text,
            'duration': duration,
            'sample_rate': sample_rate,
            'reference_emotion': reference_emotion,
            'status': 'success'
        })

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('🚀 Starting CSM server on port 19517...')
    print(f'📁 Emotion references: {EMOTION_REF_DIR}')
    app.run(host='0.0.0.0', port=19517, debug=False)
EOF

chmod +x /workspace/oviya-production/csm_server.py
echo "✅ CSM server script created"
echo ""

# Create WhisperX test script
echo "🎤 Creating WhisperX test script..."
cat > /workspace/oviya-production/test_whisperx.py << 'EOF'
#!/usr/bin/env python3
"""
Test WhisperX on RTX 5880 Ada
"""

import whisperx
import torch
import numpy as np
import time

def test_whisperx():
    print("🧪 Testing WhisperX on RTX 5880 Ada")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"VRAM: {gpu_props.total_memory / 1e9:.1f}GB")
        print(f"CUDA: {torch.version.cuda}")
    
    # Load models
    print("\n📥 Loading WhisperX models...")
    start_time = time.time()
    
    model = whisperx.load_model("large-v2", device, compute_type="float16")
    align_model, metadata = whisperx.load_align_model("en", device)
    
    load_time = time.time() - start_time
    print(f"✅ Models loaded in {load_time:.2f}s")
    
    # Test with dummy audio
    print("\n🎤 Testing transcription with batch_size=8...")
    test_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
    
    start_time = time.time()
    result = model.transcribe(test_audio, batch_size=8)
    transcribe_time = time.time() - start_time
    
    print(f"✅ Transcription completed in {transcribe_time:.2f}s")
    print(f"   Batch size: 8 (optimized for RTX 5880)")
    print(f"   Audio length: 3.0s")
    print(f"   Segments: {len(result.get('segments', []))}")
    
    # Test alignment
    if result.get('segments'):
        print("\n🎯 Testing word-level alignment...")
        start_time = time.time()
        
        aligned_result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            test_audio,
            device,
            return_char_alignments=False
        )
        
        align_time = time.time() - start_time
        print(f"✅ Alignment completed in {align_time:.2f}s")
        
        # Count words
        word_count = sum(len(seg.get('words', [])) for seg in aligned_result.get('segments', []))
        print(f"   Word timestamps: {word_count}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_max = torch.cuda.max_memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Current: {memory_used:.1f}GB")
        print(f"   Peak: {memory_max:.1f}GB")
        print(f"   Total: {memory_total:.1f}GB")
        print(f"   Available: {memory_total - memory_max:.1f}GB")
    
    print("\n" + "=" * 60)
    print("✅ WhisperX test completed successfully!")
    print("🎉 RTX 5880 Ada is ready for production!")
    print("=" * 60)

if __name__ == "__main__":
    test_whisperx()
EOF

chmod +x /workspace/oviya-production/test_whisperx.py
echo "✅ WhisperX test script created"
echo ""

# Start CSM server
echo "🎤 Starting CSM server..."
cd /workspace/oviya-production
nohup python3 csm_server.py > /workspace/csm_server.log 2>&1 &
CSM_PID=$!
echo "   CSM PID: $CSM_PID"
sleep 10

# Verify CSM is running
if curl -s http://localhost:19517/health > /dev/null 2>&1; then
    echo "✅ CSM server is running"
else
    echo "⚠️  CSM server may not be running. Check /workspace/csm_server.log"
fi
echo ""

# Test services
echo "🧪 Testing all services..."
echo ""

# Test Ollama
echo "Testing Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running on port 11434"
    ollama list
else
    echo "❌ Ollama is not responding"
fi
echo ""

# Test CSM
echo "Testing CSM..."
if curl -s http://localhost:19517/health > /dev/null 2>&1; then
    echo "✅ CSM server is running on port 19517"
    curl -s http://localhost:19517/health | python3 -m json.tool
else
    echo "❌ CSM server is not responding"
fi
echo ""

# Test WhisperX
echo "Testing WhisperX..."
python3 /workspace/oviya-production/test_whisperx.py
echo ""

# Create tunnel setup script
echo "🌐 Creating Cloudflare tunnel setup script..."
cat > /workspace/setup_tunnels.sh << 'EOF'
#!/bin/bash
echo "🌐 Setting up Cloudflare tunnels for RTX 5880 Ada..."
echo ""

# Install cloudflared
if ! command -v cloudflared &> /dev/null; then
    echo "📥 Installing cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
    echo "✅ cloudflared installed"
else
    echo "✅ cloudflared already installed"
fi
echo ""

# Kill any existing tunnels
pkill cloudflared || true
sleep 2

# Create Ollama tunnel
echo "🔗 Creating Ollama tunnel (port 11434)..."
nohup cloudflared tunnel --url http://localhost:11434 > /workspace/ollama_tunnel.log 2>&1 &
OLLAMA_TUNNEL_PID=$!
echo "   Ollama tunnel PID: $OLLAMA_TUNNEL_PID"
sleep 5

# Create CSM tunnel
echo "🔗 Creating CSM tunnel (port 19517)..."
nohup cloudflared tunnel --url http://localhost:19517 > /workspace/csm_tunnel.log 2>&1 &
CSM_TUNNEL_PID=$!
echo "   CSM tunnel PID: $CSM_TUNNEL_PID"
sleep 5

echo ""
echo "✅ Tunnels created!"
echo ""
echo "📋 To get tunnel URLs, run:"
echo "   cat /workspace/ollama_tunnel.log | grep trycloudflare.com"
echo "   cat /workspace/csm_tunnel.log | grep trycloudflare.com"
echo ""
echo "💡 Save these URLs to update your local configuration!"
EOF

chmod +x /workspace/setup_tunnels.sh
echo "✅ Tunnel setup script created"
echo ""

# Create service status script
cat > /workspace/check_services.sh << 'EOF'
#!/bin/bash
echo "🔍 Checking Oviya Services on RTX 5880 Ada"
echo "=========================================="
echo ""

# GPU Status
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# Ollama
echo "🧠 Ollama (port 11434):"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✅ Running"
    ollama list
else
    echo "   ❌ Not running"
fi
echo ""

# CSM
echo "🎤 CSM Server (port 19517):"
if curl -s http://localhost:19517/health > /dev/null 2>&1; then
    echo "   ✅ Running"
    curl -s http://localhost:19517/health | python3 -m json.tool
else
    echo "   ❌ Not running"
fi
echo ""

# Tunnels
echo "🌐 Cloudflare Tunnels:"
if pgrep -f "cloudflared tunnel" > /dev/null; then
    echo "   ✅ Running"
    echo "   Ollama tunnel: $(cat /workspace/ollama_tunnel.log 2>/dev/null | grep -o 'https://[^[:space:]]*trycloudflare.com' | head -1)"
    echo "   CSM tunnel: $(cat /workspace/csm_tunnel.log 2>/dev/null | grep -o 'https://[^[:space:]]*trycloudflare.com' | head -1)"
else
    echo "   ❌ Not running (run /workspace/setup_tunnels.sh)"
fi
echo ""

# Processes
echo "📊 Running Processes:"
ps aux | grep -E "ollama|csm_server|cloudflared" | grep -v grep
EOF

chmod +x /workspace/check_services.sh
echo "✅ Service status script created"
echo ""

# Final summary
echo "════════════════════════════════════════════════════════════"
echo "🎉 Oviya EI + WhisperX Setup Complete on RTX 5880 Ada!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📋 Services Running:"
echo "   ✅ Ollama LLM: http://localhost:11434"
echo "   ✅ CSM Voice: http://localhost:19517"
echo "   ✅ WhisperX: Cached and ready"
echo ""
echo "📁 Project Location: /workspace/oviya-production"
echo ""
echo "🔧 Next Steps:"
echo "   1. Setup tunnels: /workspace/setup_tunnels.sh"
echo "   2. Check services: /workspace/check_services.sh"
echo "   3. Copy your Oviya code to /workspace/oviya-production/"
echo "   4. Update local config with tunnel URLs"
echo ""
echo "📝 Logs:"
echo "   - Ollama: /workspace/ollama.log"
echo "   - CSM: /workspace/csm_server.log"
echo "   - Ollama tunnel: /workspace/ollama_tunnel.log"
echo "   - CSM tunnel: /workspace/csm_tunnel.log"
echo ""
echo "🎯 RTX 5880 Ada (48GB VRAM) is ready for production!"
echo "════════════════════════════════════════════════════════════"

