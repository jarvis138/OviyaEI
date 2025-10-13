#!/bin/bash
# OpenVoiceV2 Installation Script for Vast.ai
# Run this on your Vast.ai server

echo "🚀 Installing OpenVoiceV2 on Vast.ai..."
echo "=========================================="

# Navigate to workspace
cd /workspace

# Clone OpenVoiceV2 repository
echo "📥 Cloning OpenVoiceV2 repository..."
git clone https://github.com/myshell-ai/OpenVoice.git

# Navigate to OpenVoice directory
cd OpenVoice

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install additional dependencies that might be missing
pip install torch torchaudio librosa soundfile numpy scipy

# Test installation
echo "🧪 Testing OpenVoiceV2 installation..."
python3 -c "
try:
    from openvoice import se_extractor
    print('✅ OpenVoiceV2 installation successful!')
except ImportError as e:
    print(f'❌ OpenVoiceV2 installation failed: {e}')
    exit(1)
"

echo "✅ OpenVoiceV2 installation complete!"
echo "📁 Location: /workspace/OpenVoice"