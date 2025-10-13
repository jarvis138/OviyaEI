#!/bin/bash
# Project Cleanup Script
# Removes temporary files, old tests, and organizes the project

echo "🧹 Cleaning up Oviya Production project..."
echo ""

# Create archive directory for old files
mkdir -p archive/old_tests
mkdir -p archive/old_docs
mkdir -p archive/logs

# Move old test files to archive
echo "📦 Archiving old test files..."
mv test_audio_quality.py archive/old_tests/ 2>/dev/null
mv test_both_services.py archive/old_tests/ 2>/dev/null
mv test_complete_system.py archive/old_tests/ 2>/dev/null
mv test_csm_connection.py archive/old_tests/ 2>/dev/null
mv test_emotion_library.py archive/old_tests/ 2>/dev/null
mv test_maya_enhancements.py archive/old_tests/ 2>/dev/null
mv test_maya_pipeline.py archive/old_tests/ 2>/dev/null
mv test_production_readiness.py archive/old_tests/ 2>/dev/null
mv test_services.py archive/old_tests/ 2>/dev/null
mv test_single_scenario.py archive/old_tests/ 2>/dev/null
mv test_voice_only.py archive/old_tests/ 2>/dev/null

# Move old documentation to archive
echo "📦 Archiving old documentation..."
mv BEYOND_MAYA_IMPLEMENTATION.md archive/old_docs/ 2>/dev/null
mv BEYOND_MAYA_SUMMARY.md archive/old_docs/ 2>/dev/null
mv EMOTION_REFERENCE_GUIDE.md archive/old_docs/ 2>/dev/null
mv EMOTION_REFERENCE_SUCCESS.md archive/old_docs/ 2>/dev/null
mv IMPLEMENTATION_SUMMARY.md archive/old_docs/ 2>/dev/null
mv MAYA_LEVEL_IMPLEMENTATION.md archive/old_docs/ 2>/dev/null
mv README_MAYA.md archive/old_docs/ 2>/dev/null
mv REAL_EMOTION_SUCCESS.md archive/old_docs/ 2>/dev/null

# Move server setup scripts to archive
echo "📦 Archiving server setup scripts..."
mv csm_server_with_emotions.py archive/old_tests/ 2>/dev/null
mv csm_server_expanded_emotions.py archive/old_tests/ 2>/dev/null
mv fixed_csm_server_vastai.py archive/old_tests/ 2>/dev/null
mv update_csm_server_vastai.py archive/old_tests/ 2>/dev/null
mv extract_emotion_references_vastai.py archive/old_tests/ 2>/dev/null
mv extract_all_emotions.py archive/old_tests/ 2>/dev/null
mv extract_real_emotion_references.py archive/old_tests/ 2>/dev/null

# Move paste files to archive
echo "📦 Archiving paste/instruction files..."
mv PASTE_*.txt archive/old_docs/ 2>/dev/null
mv TEST_AND_START_CSM.txt archive/old_docs/ 2>/dev/null

# Move log files
echo "📦 Archiving log files..."
mv *.log archive/logs/ 2>/dev/null
mv /tmp/diverse_test.log archive/logs/ 2>/dev/null

# Remove DS_Store files
echo "🗑️  Removing .DS_Store files..."
find . -name ".DS_Store" -delete

# Remove Python cache
echo "🗑️  Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# Remove old WAV test files from root
echo "🗑️  Removing old test WAV files..."
rm -f test_*.wav 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Project structure:"
echo "   ✅ Current tests: test_beyond_maya.py, test_5_scenarios.py, test_diverse_scenarios.py, test_llm_prosody_5.py"
echo "   ✅ Production files: pipeline.py, brain/, voice/, emotion_controller/"
echo "   ✅ Documentation: PRODUCTION_READINESS.md, IMPLEMENTATION_STATUS.md, ENHANCEMENTS_COMPLETE.md"
echo "   ✅ Archives: archive/old_tests/, archive/old_docs/, archive/logs/"
echo ""


