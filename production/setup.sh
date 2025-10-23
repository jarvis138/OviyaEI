#!/bin/bash
# Oviya Production Setup Script

echo "🚀 Setting up Oviya Production with OpenVoiceV2"
echo "================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p external
mkdir -p data/voice_samples
mkdir -p voice/adapters
mkdir -p output
mkdir -p logs

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama not found. Installing Ollama..."
    
    # Install Ollama (Linux/macOS)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - install via Homebrew or direct download
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Please install Ollama manually from https://ollama.ai"
        fi
    else
        echo "Please install Ollama manually from https://ollama.ai"
    fi
else
    echo "✅ Ollama found"
fi

# Start Ollama service
echo "🔄 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
sleep 5

# Pull Qwen2.5:7B model
echo "📥 Pulling Qwen2.5:7B model..."
ollama pull qwen2.5:7b

# Setup OpenVoiceV2
echo "🎤 Setting up OpenVoiceV2..."

if [ ! -d "external/OpenVoice" ]; then
    echo "📥 Cloning OpenVoice repository..."
    git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice
else
    echo "✅ OpenVoice already cloned"
fi

cd external/OpenVoice

# Install OpenVoice
echo "📦 Installing OpenVoice..."
pip install -e .

# Download OpenVoiceV2 model
echo "📥 Downloading OpenVoiceV2 model..."
mkdir -p models
huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ./models/OpenVoiceV2

cd ../..

# Test CSM connection
echo "🧪 Testing CSM connection..."
python3 test_csm_connection.py

# Test the setup
echo "🧪 Testing setup..."
# Test Ollama
echo "Testing Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not responding"
fi

# Test Python imports
echo "Testing Python imports..."
python3 -c "
try:
    from emotion_detector.detector import EmotionDetector
    from brain.llm_brain import OviyaBrain
    from emotion_controller.controller import EmotionController
    from voice.openvoice_tts import HybridVoiceEngine
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Create example voice sample
echo "🎵 Creating example voice sample..."
cat > data/voice_samples/README.md << 'EOF'
# Voice Samples

Place Oviya's voice samples here for voice cloning.

## Required Format
- WAV format
- 24kHz sample rate
- Clear, single speaker
- 3-10 seconds duration
- Natural speech (not too fast/slow)

## Example Files
- oviya_calm.wav
- oviya_joyful.wav
- oviya_empathetic.wav
EOF

# Create run script
echo "📝 Creating run script..."
cat > run_oviya.sh << 'EOF'
#!/bin/bash
# Run Oviya Production Pipeline

echo "🚀 Starting Oviya Production Pipeline"

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "🔄 Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Run the pipeline
python3 pipeline.py
EOF

chmod +x run_oviya.sh

# Create voice cloning script
echo "📝 Creating voice cloning script..."
cat > clone_voice.py << 'EOF'
#!/usr/bin/env python3
"""
Voice Cloning Script for Oviya

Usage:
    python3 clone_voice.py path/to/reference_audio.wav
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice.openvoice_tts import OpenVoiceV2TTS

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 clone_voice.py path/to/reference_audio.wav")
        sys.exit(1)
    
    reference_path = sys.argv[1]
    
    if not Path(reference_path).exists():
        print(f"❌ Reference audio not found: {reference_path}")
        sys.exit(1)
    
    print(f"🎤 Cloning voice from: {reference_path}")
    
    tts = OpenVoiceV2TTS()
    success = tts.clone_voice(reference_path, "oviya")
    
    if success:
        print("✅ Voice cloning completed!")
        print("You can now use 'oviya' as speaker_id in the pipeline")
    else:
        print("❌ Voice cloning failed")

if __name__ == "__main__":
    main()
EOF

chmod +x clone_voice.py

echo ""
echo "✅ Setup completed!"
echo ""
echo "Next steps:"
echo "1. Add Oviya's voice samples to data/voice_samples/"
echo "2. Run: python3 clone_voice.py data/voice_samples/oviya_sample.wav"
echo "3. Run: ./run_oviya.sh"
echo ""
echo "For manual testing:"
echo "- Test emotion detector: python3 emotion_detector/detector.py"
echo "- Test brain: python3 brain/llm_brain.py"
echo "- Test emotion controller: python3 emotion_controller/controller.py"
echo "- Test voice: python3 voice/openvoice_tts.py"
echo ""
echo "🎉 Oviya is ready to go!"
