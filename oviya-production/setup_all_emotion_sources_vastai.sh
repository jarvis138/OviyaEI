#!/bin/bash
# Complete Multi-Source Emotion System Setup for Vast.ai
# Downloads and sets up: OpenVoiceV2, EmotiVoice, StyleTTS2, VITS-emotion
# Then extracts 49+ real emotion references

set -e  # Exit on error

echo "🎭 Multi-Source Emotion Reference System Setup"
echo "=============================================="
echo ""

# Check if we're on Vast.ai
if [ ! -d "/workspace" ]; then
    echo "❌ Error: /workspace not found. Are you on Vast.ai?"
    exit 1
fi

cd /workspace

# Activate conda environment if it exists
if [ -d "/venv/csm" ]; then
    source /opt/miniforge3/etc/profile.d/conda.sh
    conda activate csm
    echo "✅ Activated conda environment: csm"
fi

# ============================================
# 1. Setup OpenVoiceV2
# ============================================
echo ""
echo "📥 [1/4] Setting up OpenVoiceV2..."
echo "=============================================="

if [ ! -d "/workspace/OpenVoice" ]; then
    git clone https://github.com/myshell-ai/OpenVoice.git
    cd OpenVoice
    pip install -e . --no-deps
    pip install torch torchaudio librosa pydub soundfile
    cd /workspace
    echo "✅ OpenVoiceV2 cloned and installed"
else
    echo "✅ OpenVoiceV2 already exists"
fi

# Download OpenVoice V2 checkpoints
if [ ! -d "/workspace/OpenVoice/checkpoints_v2" ]; then
    cd /workspace/OpenVoice
    echo "📥 Downloading OpenVoiceV2 checkpoints..."
    mkdir -p checkpoints_v2
    huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir checkpoints_v2
    echo "✅ OpenVoiceV2 checkpoints downloaded"
    cd /workspace
else
    echo "✅ OpenVoiceV2 checkpoints already exist"
fi

# ============================================
# 2. Setup EmotiVoice
# ============================================
echo ""
echo "📥 [2/4] Setting up EmotiVoice..."
echo "=============================================="

if [ ! -d "/workspace/EmotiVoice" ]; then
    git clone https://github.com/netease-youdao/EmotiVoice.git
    cd EmotiVoice
    pip install -e . --no-deps
    pip install torch torchaudio numpy scipy
    cd /workspace
    echo "✅ EmotiVoice cloned and installed"
else
    echo "✅ EmotiVoice already exists"
fi

# Download EmotiVoice pretrained models
if [ ! -d "/workspace/EmotiVoice/outputs" ]; then
    cd /workspace/EmotiVoice
    echo "📥 Downloading EmotiVoice checkpoints..."
    mkdir -p outputs
    # Download from Hugging Face
    huggingface-cli download netease-youdao/EmotiVoice --local-dir outputs || echo "⚠️ EmotiVoice models download failed, will use fallback"
    cd /workspace
else
    echo "✅ EmotiVoice checkpoints already exist"
fi

# ============================================
# 3. Setup StyleTTS2
# ============================================
echo ""
echo "📥 [3/4] Setting up StyleTTS2..."
echo "=============================================="

if [ ! -d "/workspace/StyleTTS2" ]; then
    git clone https://github.com/yl4579/StyleTTS2.git
    cd StyleTTS2
    pip install -e . --no-deps
    pip install phonemizer transformers accelerate
    cd /workspace
    echo "✅ StyleTTS2 cloned and installed"
else
    echo "✅ StyleTTS2 already exists"
fi

# Download StyleTTS2 LJSpeech model
if [ ! -d "/workspace/StyleTTS2/Models" ]; then
    cd /workspace/StyleTTS2
    mkdir -p Models
    echo "📥 Downloading StyleTTS2 checkpoints..."
    huggingface-cli download yl4579/StyleTTS2-LJSpeech --local-dir Models || echo "⚠️ StyleTTS2 models download failed, will use fallback"
    cd /workspace
else
    echo "✅ StyleTTS2 checkpoints already exist"
fi

# ============================================
# 4. Setup Emotion Datasets
# ============================================
echo ""
echo "📦 [4/4] Setting up emotion dataset directories..."
echo "=============================================="

mkdir -p /workspace/emotion_datasets/{IEMOCAP,RAVDESS,CREMA-D,MELD,EmoDB}
echo "✅ Dataset directories created"
echo "⚠️  Note: Actual datasets require manual download due to licensing"
echo "   - IEMOCAP: Requires research license"
echo "   - RAVDESS: Available at zenodo.org/record/1188976"
echo "   - CREMA-D: Available at github.com/CheyneyComputerScience/CREMA-D"
echo "   For now, we'll use the TTS models for emotion references"

# ============================================
# 5. Install additional dependencies
# ============================================
echo ""
echo "📦 Installing additional dependencies..."
echo "=============================================="

pip install librosa soundfile pydub phonemizer unidecode inflect --quiet
echo "✅ Dependencies installed"

# ============================================
# Summary
# ============================================
echo ""
echo "🎉 Setup Complete!"
echo "=============================================="
echo ""
echo "📊 Installed Components:"
echo "   ✅ OpenVoiceV2:  /workspace/OpenVoice"
echo "   ✅ EmotiVoice:   /workspace/EmotiVoice"
echo "   ✅ StyleTTS2:    /workspace/StyleTTS2"
echo "   ✅ Datasets:     /workspace/emotion_datasets"
echo ""
echo "🚀 Next Steps:"
echo "   1. Run: python3 extract_real_emotion_references.py"
echo "   2. This will generate 49+ real emotion WAV files"
echo "   3. Then restart CSM server with: python3 csm_server_expanded_emotions.py"
echo ""

