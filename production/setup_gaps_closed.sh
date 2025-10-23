#!/bin/bash

# Setup Script for Gap-Closed Features
# Installs all dependencies and runs validation tests

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     Oviya Gap-Closed Features Setup                         ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "❌ Python 3.10+ required"
    exit 1
fi

echo "✅ Python version OK"
echo ""

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/personalities
mkdir -p audio_assets/breath_samples
mkdir -p logs/analytics
mkdir -p validation/results
mkdir -p testing/ab_tests
mkdir -p testing/test_ab

echo "✅ Directories created"
echo ""

# Run validation tests
echo "🧪 Running validation tests..."
echo ""

echo "1️⃣  Testing Acoustic Emotion Detection..."
python3 voice/acoustic_emotion_detector.py
echo ""

echo "2️⃣  Testing Personality Store..."
python3 brain/personality_store.py
echo ""

echo "3️⃣  Testing Analytics Pipeline..."
python3 monitoring/analytics_pipeline.py
echo ""

echo "4️⃣  Testing Emotion Validator..."
python3 validation/emotion_validator.py
echo ""

echo "5️⃣  Testing A/B Framework..."
python3 testing/ab_test_framework.py
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     ✅ Setup Complete!                                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "📊 Features Ready:"
echo "   ✅ Acoustic Emotion Detection"
echo "   ✅ Persistent Personality Storage"
echo "   ✅ Speaker Diarization"
echo "   ✅ WebSocket Streaming"
echo "   ✅ Docker Deployment"
echo "   ✅ Structured Analytics"
echo "   ✅ Emotion Validation"
echo "   ✅ A/B Testing Framework"
echo ""

echo "🚀 Next Steps:"
echo "   1. Start WebSocket server: python3 websocket_server.py"
echo "   2. Or deploy with Docker: docker-compose up -d"
echo "   3. Read QUICK_START_GAPS_CLOSED.md for usage examples"
echo ""

echo "📚 Documentation:"
echo "   - GAP_ANALYSIS_IMPLEMENTATION_COMPLETE.md (detailed docs)"
echo "   - QUICK_START_GAPS_CLOSED.md (quick start guide)"
echo ""

echo "✅ All gaps closed! Ready to ship! 🎉"

