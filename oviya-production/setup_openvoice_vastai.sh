#!/bin/bash
# Setup OpenVoiceV2 on Vast.ai Server
# Run this on your Vast.ai instance

echo "🚀 Setting up OpenVoiceV2 on Vast.ai..."

# Navigate to workspace
cd /workspace

# Clone OpenVoiceV2 repository
echo "📦 Cloning OpenVoiceV2..."
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice

# Install dependencies
echo "📦 Installing OpenVoiceV2 dependencies..."
pip install -r requirements.txt

# Create emotion references directory
echo "📁 Creating emotion references directory..."
mkdir -p /workspace/emotion_references

echo "✅ OpenVoiceV2 setup complete!"
echo ""
echo "Next steps:"
echo "1. Test OpenVoiceV2: python -c 'import openvoice; print(openvoice.__version__)'"
echo "2. Run emotion reference extraction"


