#!/bin/bash
# Complete OpenVoiceV2 Model Download Script
# Downloads OpenVoiceV2 repository + models from Hugging Face

echo "📥 Downloading OpenVoiceV2 Models"
echo "=================================="

cd /workspace

# Step 1: Clone OpenVoiceV2 repository
echo ""
echo "📦 Step 1: Cloning OpenVoiceV2 repository..."
if [ -d "OpenVoice" ]; then
    echo "   ✅ OpenVoice repository already exists"
else
    git clone https://github.com/myshell-ai/OpenVoice.git
    if [ $? -eq 0 ]; then
        echo "   ✅ Repository cloned successfully"
    else
        echo "   ❌ Failed to clone repository"
        exit 1
    fi
fi

# Step 2: Install dependencies
echo ""
echo "📦 Step 2: Installing dependencies..."
cd OpenVoice
pip install -r requirements.txt
pip install torch torchaudio librosa soundfile numpy scipy

# Step 3: Download OpenVoiceV2 models from Hugging Face
echo ""
echo "📥 Step 3: Downloading OpenVoiceV2 models from Hugging Face..."

# Create checkpoints directory
mkdir -p checkpoints_v2

# Download models using huggingface-cli
echo "   Downloading base speaker models..."
huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir checkpoints_v2 --repo-type model

if [ $? -eq 0 ]; then
    echo "   ✅ Models downloaded successfully"
else
    echo "   ⚠️  Huggingface-cli failed, trying git-lfs..."
    cd checkpoints_v2
    git lfs install
    git clone https://huggingface.co/myshell-ai/OpenVoiceV2 .
    cd ..
fi

# Step 4: Verify models
echo ""
echo "🔍 Step 4: Verifying downloaded models..."
if [ -d "checkpoints_v2" ]; then
    echo "   📁 Models directory: $(du -sh checkpoints_v2)"
    echo "   📋 Contents:"
    ls -lh checkpoints_v2/ | head -10
    echo "   ✅ Models verified"
else
    echo "   ❌ Models not found"
    exit 1
fi

# Step 5: Test installation
echo ""
echo "🧪 Step 5: Testing OpenVoiceV2 installation..."
cd /workspace/OpenVoice
python3 -c "
import sys
sys.path.insert(0, '/workspace/OpenVoice')
try:
    import torch
    import torchaudio
    print('   ✅ PyTorch and TorchAudio working')
    
    # Try to import OpenVoice modules
    from openvoice import se_extractor
    print('   ✅ OpenVoice modules loaded')
    
    print('   ✅ OpenVoiceV2 installation complete!')
except Exception as e:
    print(f'   ❌ Installation test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 OpenVoiceV2 models downloaded and verified!"
    echo "📁 Location: /workspace/OpenVoice/checkpoints_v2"
else
    echo "❌ Installation test failed"
    exit 1
fi


