#!/bin/bash

# Setup script for expanded emotion library on Vast.ai
# This script will:
# 1. Generate 28 emotion references using CSM
# 2. Create emotion library config
# 3. Start CSM server with expanded emotions

set -e  # Exit on error

echo "======================================================================="
echo "🎨 EXPANDED EMOTION LIBRARY SETUP"
echo "======================================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Create directories
echo -e "\n${YELLOW}[1/5] Creating directories...${NC}"
mkdir -p /workspace/emotion_references
mkdir -p /workspace/scripts
cd /workspace

# Step 2: Copy scripts to Vast.ai (user will do this manually or via scp)
echo -e "\n${YELLOW}[2/5] Please ensure these files are uploaded to /workspace/scripts/:${NC}"
echo "   - generate_expanded_emotions_vastai.py"
echo "   - vastai_csm_server_expanded.py"
echo ""
read -p "Press Enter when files are ready..."

# Step 3: Generate emotion references
echo -e "\n${YELLOW}[3/5] Generating 28 emotion references (this will take ~15-20 minutes)...${NC}"
cd /workspace/csm/csm
python3 /workspace/scripts/generate_expanded_emotions_vastai.py

# Check if generation was successful
REF_COUNT=$(ls /workspace/emotion_references/*.wav 2>/dev/null | wc -l)
echo -e "${GREEN}✅ Generated $REF_COUNT emotion reference files${NC}"

# Step 4: Create emotion library config
echo -e "\n${YELLOW}[4/5] Creating emotion library config...${NC}"
cat > /workspace/emotion_references/emotion_library.json << 'EOF'
{
  "version": "1.0",
  "total_emotions": 28,
  "tiers": {
    "tier1_core": [
      "calm_supportive", "empathetic_sad", "joyful_excited",
      "confident", "neutral", "comforting", "encouraging",
      "thoughtful", "affectionate", "reassuring"
    ],
    "tier2_contextual": [
      "playful", "concerned_anxious", "melancholy", "wistful",
      "tired", "curious", "dreamy", "relieved", "proud"
    ],
    "tier3_expressive": [
      "angry_firm", "sarcastic", "mischievous", "tender",
      "amused", "sympathetic", "reflective", "grateful", "apologetic"
    ]
  },
  "emotion_texts": {
    "calm_supportive": "Take a deep breath. Everything will be okay.",
    "empathetic_sad": "I'm so sorry you're going through this.",
    "joyful_excited": "Wow! That is wonderful! I am so excited!",
    "playful": "Hey there! This is going to be fun!",
    "confident": "You've got this. I believe in you.",
    "concerned_anxious": "Are you okay? I'm here if you need me.",
    "angry_firm": "That's not acceptable. This needs to stop.",
    "neutral": "Hello. How can I help you today.",
    "comforting": "It's okay. I'm here for you, everything will be alright.",
    "encouraging": "You can do this! I believe in you completely.",
    "thoughtful": "Let me think about that for a moment. That's interesting.",
    "affectionate": "I care about you so much. You mean a lot to me.",
    "reassuring": "Don't worry. You're safe, and everything is going to be fine.",
    "melancholy": "Sometimes things are hard, and that's okay to feel.",
    "wistful": "I remember those days. It feels like a distant dream now.",
    "tired": "It's been a long day. I'm feeling a bit worn out.",
    "curious": "Really? Tell me more! I want to know everything about it.",
    "dreamy": "Imagine a peaceful place where everything is calm and beautiful.",
    "relieved": "Oh thank goodness! I'm so glad that worked out.",
    "proud": "Look at what you've accomplished! That's truly impressive.",
    "sarcastic": "Oh yeah, that's exactly what I meant. Totally.",
    "mischievous": "I have an idea, and you're going to love this.",
    "tender": "You're so precious to me. I want you to know that.",
    "amused": "Ha! That's actually pretty funny when you think about it.",
    "sympathetic": "I understand how you feel. That must be really difficult.",
    "reflective": "Looking back, I can see how all of this connects.",
    "grateful": "Thank you so much. I really appreciate everything you've done.",
    "apologetic": "I'm truly sorry. I didn't mean for things to turn out this way."
  }
}
EOF

echo -e "${GREEN}✅ Created emotion library config${NC}"

# Step 5: Start CSM server
echo -e "\n${YELLOW}[5/5] Starting CSM server with expanded emotions...${NC}"
echo -e "${GREEN}Server will run on port 6006${NC}"
echo ""
cd /workspace/csm/csm
python3 /workspace/scripts/vastai_csm_server_expanded.py

# Note: Server runs in foreground. To run in background, add & at the end
# or use: nohup python3 /workspace/scripts/vastai_csm_server_expanded.py > csm_server.log 2>&1 &

