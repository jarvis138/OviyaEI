#!/bin/bash
# Download CSM-1B and Mimi models for Oviya

echo "📥 Downloading CSM-1B Models"
echo "============================"

# Set cache directory
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface"

echo "📁 Cache directory: $HF_HOME"

# Create cache directory
mkdir -p "$HF_HOME"

# Step 1: Download CSM-1B model
echo ""
echo "📦 Step 1: Downloading CSM-1B model..."
huggingface-cli download sesame/csm-1b --local-dir "$HF_HOME/hub/models--sesame--csm-1b" --repo-type model

if [ $? -eq 0 ]; then
    echo "   ✅ CSM-1B model downloaded successfully"
else
    echo "   ❌ CSM-1B download failed"
    exit 1
fi

# Step 2: Download Mimi decoder model
echo ""
echo "📦 Step 2: Downloading Mimi decoder..."
huggingface-cli download kyutai/mimi --local-dir "$HF_HOME/hub/models--kyutai--mimi" --repo-type model

if [ $? -eq 0 ]; then
    echo "   ✅ Mimi decoder downloaded successfully"
else
    echo "   ❌ Mimi download failed"
    exit 1
fi

# Step 3: Verify models
echo ""
echo "🔍 Step 3: Verifying downloaded models..."
if [ -d "$HF_HOME/hub/models--sesame--csm-1b" ]; then
    echo "   📁 CSM-1B: $(du -sh "$HF_HOME/hub/models--sesame--csm-1b")"
    echo "   📋 Contents:"
    ls -la "$HF_HOME/hub/models--sesame--csm-1b" | head -5
else
    echo "   ❌ CSM-1B model not found"
    exit 1
fi

if [ -d "$HF_HOME/hub/models--kyutai--mimi" ]; then
    echo "   📁 Mimi: $(du -sh "$HF_HOME/hub/models--kyutai--mimi")"
    echo "   📋 Contents:"
    ls -la "$HF_HOME/hub/models--kyutai--mimi" | head -5
else
    echo "   ❌ Mimi decoder not found"
    exit 1
fi

# Step 4: Test imports
echo ""
echo "🧪 Step 4: Testing model imports..."
python3 -c "
import os
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, MimiModel

# Set paths to match server expectations
os.environ['HF_HOME'] = '$PWD/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '$PWD/.cache/huggingface'

model_path = '$PWD/.cache/huggingface/hub/models--sesame--csm-1b'
mimi_path = '$PWD/.cache/huggingface/hub/models--kyutai--mimi'

try:
    print('   🔄 Loading processor...')
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    print('   ✅ Processor loaded')

    print('   🔄 Loading CSM model...')
    csm_model = CsmForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    print('   ✅ CSM model loaded')

    print('   🔄 Loading Mimi decoder...')
    mimi_decoder = MimiModel.from_pretrained(mimi_path, local_files_only=True)
    print('   ✅ Mimi decoder loaded')

    print('   ✅ All models loaded successfully!')

except Exception as e:
    print(f'   ❌ Model loading failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 CSM-1B models downloaded and verified!"
    echo "📁 Cache location: $HF_HOME"
    echo ""
    echo "🚀 Ready to start CSM-1B server!"
else
    echo "❌ Model verification failed"
    exit 1
fi
