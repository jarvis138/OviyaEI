#!/bin/bash
# Final Deep Cleanup Script
# Removes ALL temporary fix files, setup guides, and old documentation

echo "🧹 FINAL DEEP CLEANUP - Removing all temporary files..."
echo ""

# Create archive directories
mkdir -p archive/setup_guides
mkdir -p archive/fix_files
mkdir -p archive/old_complete_docs

echo "📦 Archiving setup guides and quickstarts..."
# Move all setup/quickstart/guide files
mv COMPLETE_EMOTION_SYSTEM_GUIDE.md archive/setup_guides/ 2>/dev/null
mv COMPLETE_EMOTION_SYSTEM_SETUP.md archive/setup_guides/ 2>/dev/null
mv EXPANDED_EMOTIONS_GUIDE.md archive/setup_guides/ 2>/dev/null
mv QUICKSTART_EMOTION_REFERENCES.md archive/setup_guides/ 2>/dev/null
mv SKIP_OPENVOICE_QUICKSTART.md archive/setup_guides/ 2>/dev/null
mv STAGE0_GUIDE.md archive/setup_guides/ 2>/dev/null
mv VASTAI_SETUP_INSTRUCTIONS.md archive/setup_guides/ 2>/dev/null
mv QUICK_START_EXPANDED_EMOTIONS.md archive/setup_guides/ 2>/dev/null

echo "📦 Archiving ALL fix files..."
# Move ALL FIX files
mv AUDIO_QUALITY_FIXES.md archive/fix_files/ 2>/dev/null
mv CSM_SERVER_FIXED.txt archive/fix_files/ 2>/dev/null
mv FINAL_CSM_FIX.txt archive/fix_files/ 2>/dev/null
mv FINAL_DEPENDENCY_FIX.txt archive/fix_files/ 2>/dev/null
mv FINAL_FIX_PYTORCH.txt archive/fix_files/ 2>/dev/null
mv FIX_BOTH_SERVICES.txt archive/fix_files/ 2>/dev/null
mv FIX_CSM_WATERMARKER.txt archive/fix_files/ 2>/dev/null
mv FIX_HUGGINGFACE_HUB.txt archive/fix_files/ 2>/dev/null
mv FIX_NUMPY.txt archive/fix_files/ 2>/dev/null
mv FIX_PYTORCH_CUDA.txt archive/fix_files/ 2>/dev/null
mv FIX_TORCHTUNE_V2.txt archive/fix_files/ 2>/dev/null
mv FIX_TORCHTUNE.txt archive/fix_files/ 2>/dev/null
mv FIX_WATERMARKER.txt archive/fix_files/ 2>/dev/null
mv PROPER_CSM_SETUP.txt archive/fix_files/ 2>/dev/null
mv SETUP_NGROK.txt archive/fix_files/ 2>/dev/null

echo "📦 Archiving old SUCCESS/COMPLETE docs..."
# Move success/complete documentation (keep only latest)
mv CSM_SERVER_SUCCESS.md archive/old_complete_docs/ 2>/dev/null
mv STAGE0_COMPLETE.md archive/old_complete_docs/ 2>/dev/null
mv IMPLEMENTATION_COMPLETE.md archive/old_complete_docs/ 2>/dev/null
mv PRODUCTION_IMPLEMENTATION_COMPLETE.md archive/old_complete_docs/ 2>/dev/null
mv EXPANDED_EMOTIONS_SUMMARY.md archive/old_complete_docs/ 2>/dev/null

# Keep these essential docs:
# - ENHANCEMENTS_COMPLETE.md (current feature list)
# - FIX_CSM_EMOTION_REFERENCES.md (troubleshooting guide)
# - PRODUCTION_READINESS.md
# - IMPLEMENTATION_STATUS.md
# - PROJECT_STRUCTURE.md
# - PROJECT_SUMMARY.md
# - README.md

echo "📦 Archiving other old files..."
# Move other misc files
mv VASTAI_DEPLOYMENT.md archive/setup_guides/ 2>/dev/null
mv stage0_emotion_test.py archive/old_tests/ 2>/dev/null
mv generate_emotion_references.py archive/old_tests/ 2>/dev/null
mv generate_speech_emotion_refs_vastai.py archive/old_tests/ 2>/dev/null
mv extract_ravdess_emotions.py archive/old_tests/ 2>/dev/null

# Clean up scripts directory
echo "📦 Archiving scripts directory..."
if [ -d "scripts" ]; then
    mv scripts archive/ 2>/dev/null
fi

# Clean up emotion_reference directory (empty)
if [ -d "emotion_reference" ]; then
    if [ -z "$(ls -A emotion_reference)" ]; then
        echo "🗑️  Removing empty emotion_reference directory..."
        rmdir emotion_reference 2>/dev/null
    fi
fi

# Clean up any remaining .txt files (except important ones)
echo "📦 Archiving misc .txt files..."
find . -maxdepth 1 -name "*.txt" ! -name "requirements.txt" -exec mv {} archive/fix_files/ \; 2>/dev/null

# Clean up any remaining test WAV files
echo "🗑️  Removing test WAV files..."
find . -maxdepth 1 -name "*.wav" -delete 2>/dev/null

# Remove .DS_Store again
find . -name ".DS_Store" -delete 2>/dev/null

# Remove Python cache again
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

echo ""
echo "✅ FINAL CLEANUP COMPLETE!"
echo ""
echo "📊 Current project structure:"
echo ""

# Count files
CORE_FILES=$(find . -maxdepth 1 -name "*.py" ! -name "test_*" ! -name "cleanup_*" | wc -l | xargs)
TEST_FILES=$(find . -maxdepth 1 -name "test_*.py" | wc -l | xargs)
DOC_FILES=$(find . -maxdepth 1 -name "*.md" | wc -l | xargs)
ARCHIVED_FILES=$(find archive -type f | wc -l | xargs)

echo "   ✅ Core Python files: $CORE_FILES"
echo "   ✅ Test files: $TEST_FILES"
echo "   ✅ Documentation files: $DOC_FILES"
echo "   ✅ Archived files: $ARCHIVED_FILES"
echo ""
echo "📁 Essential files kept:"
echo "   - pipeline.py, monitoring.py, optimizations.py"
echo "   - production_sanity_tests.py"
echo "   - brain/, voice/, emotion_controller/, emotion_detector/"
echo "   - test_beyond_maya.py, test_5_scenarios.py, test_diverse_scenarios.py"
echo "   - test_llm_prosody_5.py, test_all_enhancements.py"
echo "   - PRODUCTION_READINESS.md, IMPLEMENTATION_STATUS.md"
echo "   - ENHANCEMENTS_COMPLETE.md, PROJECT_STRUCTURE.md"
echo "   - PROJECT_SUMMARY.md, README.md, QUICK_START.md"
echo "   - FIX_CSM_EMOTION_REFERENCES.md (troubleshooting)"
echo ""
echo "🗄️  Archived to:"
echo "   - archive/setup_guides/ (setup & quickstart docs)"
echo "   - archive/fix_files/ (all FIX_*.txt files)"
echo "   - archive/old_complete_docs/ (old SUCCESS/COMPLETE docs)"
echo "   - archive/old_tests/ (deprecated test scripts)"
echo "   - archive/old_docs/ (outdated documentation)"
echo ""


