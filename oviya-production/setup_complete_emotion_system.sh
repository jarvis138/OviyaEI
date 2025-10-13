#!/bin/bash
# Complete Multi-Source Emotion System Setup
# Downloads models, extracts emotions, generates references, and starts server

echo "🚀 Complete Multi-Source Emotion System Setup"
echo "=============================================="
echo ""
echo "This script will:"
echo "  1. Download OpenVoiceV2 models"
echo "  2. Set up emotion datasets"
echo "  3. Extract and blend emotions"
echo "  4. Generate 49+ emotion references"
echo "  5. Start CSM server with expanded emotions"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled"
    exit 1
fi

cd /workspace

# Step 1: Download OpenVoiceV2 models
echo ""
echo "================================"
echo "Step 1: Downloading OpenVoiceV2"
echo "================================"
./download_openvoice_models.sh
if [ $? -ne 0 ]; then
    echo "❌ OpenVoiceV2 download failed"
    exit 1
fi

# Step 2: Set up emotion datasets directory
echo ""
echo "======================================"
echo "Step 2: Setting up emotion datasets"
echo "======================================"
./download_emotion_datasets.sh

# Step 3: Extract all emotions
echo ""
echo "======================================"
echo "Step 3: Extracting & blending emotions"
echo "======================================"
python3 extract_all_emotions.py
if [ $? -ne 0 ]; then
    echo "❌ Emotion extraction failed"
    exit 1
fi

# Step 4: Verify emotion library
echo ""
echo "======================================"
echo "Step 4: Verifying emotion library"
echo "======================================"
if [ -d "/workspace/emotion_references" ]; then
    emotion_count=$(ls -1 /workspace/emotion_references/*.wav 2>/dev/null | wc -l)
    echo "✅ Found $emotion_count emotion references"
    echo ""
    echo "📋 Available emotions:"
    ls -1 /workspace/emotion_references/*.wav | head -10
    if [ $emotion_count -gt 10 ]; then
        echo "   ... and $(($emotion_count - 10)) more"
    fi
else
    echo "❌ Emotion references directory not found"
    exit 1
fi

# Step 5: Test CSM server (don't start yet)
echo ""
echo "======================================"
echo "Step 5: Checking CSM server"
echo "======================================"
python3 -c "
import sys
sys.path.insert(0, '/workspace/csm')
try:
    from generator import load_csm_1b
    print('✅ CSM modules available')
except Exception as e:
    print(f'❌ CSM not available: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ CSM check failed"
    echo "   Please ensure CSM is installed in /workspace/csm/"
    exit 1
fi

# Summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📊 Emotion System Summary:"
echo "   Total emotions: $emotion_count"
echo "   Location: /workspace/emotion_references/"
echo ""
echo "🚀 To start the CSM server with expanded emotions:"
echo "   python3 csm_server_expanded_emotions.py"
echo ""
echo "📝 The server will:"
echo "   - Run on port 19517"
echo "   - Support 49+ emotions"
echo "   - Use OpenVoiceV2 + blended references"
echo ""
echo "✅ Ready to use!"


