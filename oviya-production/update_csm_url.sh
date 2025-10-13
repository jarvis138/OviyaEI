#!/bin/bash

# Script to update CSM URL across all files
# Usage: ./update_csm_url.sh https://your-tunnel.trycloudflare.com

if [ -z "$1" ]; then
    echo "❌ Error: Please provide the CSM tunnel URL"
    echo "Usage: ./update_csm_url.sh https://your-tunnel.trycloudflare.com"
    exit 1
fi

CSM_TUNNEL_URL="$1"
CSM_URL_WITH_ENDPOINT="${CSM_TUNNEL_URL}/generate"

echo "🔄 Updating CSM URL to: $CSM_URL_WITH_ENDPOINT"
echo ""

# Files to update
FILES=(
    "config/service_urls.py"
    "realtime_conversation.py"
    "test_realtime_system.py"
    "pipeline.py"
    "test_diverse_scenarios.py"
    "test_llm_prosody_5.py"
    "test_5_scenarios.py"
    "test_all_enhancements.py"
    "production_sanity_tests.py"
)

# Update each file
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "📝 Updating $file..."
        # macOS compatible sed
        sed -i '' "s|https://YOUR-CSM-TUNNEL.trycloudflare.com/generate|$CSM_URL_WITH_ENDPOINT|g" "$file"
        sed -i '' "s|https://tanja-flockier-jayleen.ngrok-free.dev/generate|$CSM_URL_WITH_ENDPOINT|g" "$file"
        echo "   ✅ Updated"
    else
        echo "   ⚠️  File not found: $file"
    fi
done

echo ""
echo "✅ All files updated!"
echo ""
echo "🧪 Test the changes:"
echo "   python3 test_realtime_system.py"


