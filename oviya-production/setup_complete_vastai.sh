#!/bin/bash
# Complete OpenVoiceV2 + Emotion Reference Setup for Vast.ai
# Run this script on your Vast.ai server

echo "🚀 Complete OpenVoiceV2 + Emotion Reference Setup"
echo "=================================================="

# Step 1: Install OpenVoiceV2
echo ""
echo "📥 Step 1: Installing OpenVoiceV2..."
cd /workspace

if [ -d "OpenVoice" ]; then
    echo "   ✅ OpenVoice already exists, skipping installation"
else
    echo "   📥 Cloning OpenVoiceV2 repository..."
    git clone https://github.com/myshell-ai/OpenVoice.git
    
    echo "   📦 Installing dependencies..."
    cd OpenVoice
    pip install -r requirements.txt
    pip install torch torchaudio librosa soundfile numpy scipy
    
    echo "   🧪 Testing installation..."
    python3 -c "
try:
    from openvoice import se_extractor
    print('   ✅ OpenVoiceV2 installation successful!')
except ImportError as e:
    print(f'   ❌ OpenVoiceV2 installation failed: {e}')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ OpenVoiceV2 installation failed!"
        exit 1
    fi
fi

# Step 2: Generate emotion references
echo ""
echo "🎭 Step 2: Generating emotion reference files..."
cd /workspace

# Create emotion references directory
mkdir -p emotion_references

# Generate emotion references
python3 generate_emotion_references.py

if [ $? -ne 0 ]; then
    echo "❌ Emotion reference generation failed!"
    exit 1
fi

# Step 3: Check generated files
echo ""
echo "📋 Step 3: Checking generated files..."
ls -la emotion_references/

echo ""
echo "🎉 Setup complete! Ready to start CSM server with emotion references."
echo ""
echo "To start the updated CSM server, run:"
echo "   python3 csm_server_with_emotions.py"
echo ""
echo "The server will run on port 19517 with emotion reference support!"

