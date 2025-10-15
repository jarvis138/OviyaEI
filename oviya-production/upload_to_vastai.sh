#!/bin/bash
###############################################################################
# Upload CSM RVQ Streaming Files to Vast.ai
###############################################################################

# Configuration (update these)
VAST_HOST="${1:-ssh5.vast.ai}"
VAST_PORT="${2:-12345}"
VAST_USER="root"

if [ "$1" == "" ] || [ "$2" == "" ]; then
    echo "Usage: $0 <vast-host> <port>"
    echo "Example: $0 ssh5.vast.ai 12345"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════════════"
echo "📤 UPLOADING CSM RVQ STREAMING FILES TO VAST.AI"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "   Target: ${VAST_USER}@${VAST_HOST}:${VAST_PORT}"
echo ""

# Create remote directories
echo "📁 Creating remote directories..."
ssh -p $VAST_PORT $VAST_USER@$VAST_HOST "mkdir -p /workspace/oviya-production/voice"

# Upload streaming implementation
echo ""
echo "📤 Uploading csm_1b_stream.py..."
scp -P $VAST_PORT voice/csm_1b_stream.py $VAST_USER@$VAST_HOST:/workspace/oviya-production/voice/
if [ $? -eq 0 ]; then
    echo "   ✅ Uploaded"
else
    echo "   ❌ Failed"
    exit 1
fi

# Upload server
echo ""
echo "📤 Uploading csm_server_real_rvq.py..."
scp -P $VAST_PORT csm_server_real_rvq.py $VAST_USER@$VAST_HOST:/workspace/oviya-production/
if [ $? -eq 0 ]; then
    echo "   ✅ Uploaded"
else
    echo "   ❌ Failed"
    exit 1
fi

# Upload deployment script
echo ""
echo "📤 Uploading deploy_rvq_streaming.sh..."
scp -P $VAST_PORT deploy_rvq_streaming.sh $VAST_USER@$VAST_HOST:/workspace/oviya-production/
if [ $? -eq 0 ]; then
    echo "   ✅ Uploaded"
else
    echo "   ❌ Failed"
    exit 1
fi

# Make deployment script executable
echo ""
echo "🔧 Making deployment script executable..."
ssh -p $VAST_PORT $VAST_USER@$VAST_HOST "chmod +x /workspace/oviya-production/deploy_rvq_streaming.sh"

# Verify upload
echo ""
echo "🔍 Verifying upload..."
ssh -p $VAST_PORT $VAST_USER@$VAST_HOST "ls -lh /workspace/oviya-production/{csm_server_real_rvq.py,deploy_rvq_streaming.sh} /workspace/oviya-production/voice/csm_1b_stream.py"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "✅ UPLOAD COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "📋 Next steps on Vast.ai:"
echo ""
echo "   ssh -p $VAST_PORT $VAST_USER@$VAST_HOST"
echo "   cd /workspace/oviya-production"
echo "   bash deploy_rvq_streaming.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
