#!/usr/bin/env python3
"""
Stage 0: Emotion Reference Evaluation

Tests CSM's ability to reproduce emotions from OpenVoice V2 references
BEFORE any fine-tuning.

This is the baseline evaluation that determines if CSM can respond to
emotional conditioning from OpenVoice V2's built-in emotion library.

Usage:
    python stage0_emotion_test.py
"""

import sys
from pathlib import Path
import json

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from voice.emotion_teacher import OpenVoiceEmotionTeacher
from voice.openvoice_tts import HybridVoiceEngine
from emotion_reference.emotion_evaluator import EmotionTransferEvaluator


def print_header():
    """Print header."""
    print("\n" + "="*70)
    print(" "*15 + "🧪 STAGE 0: EMOTION REFERENCE EVALUATION")
    print("="*70)
    print("\nPurpose:")
    print("  Test if CSM can reproduce emotions from OpenVoice V2 references")
    print("  WITHOUT any fine-tuning or training.")
    print("\nWhat we're testing:")
    print("  • OpenVoice V2 (Teacher) provides emotional reference audio")
    print("  • CSM (Student) tries to reproduce that emotion")
    print("  • We measure how well CSM matches the emotional tone")
    print("\nWhy this matters:")
    print("  If CSM responds to OpenVoice V2 emotions, we can proceed to")
    print("  fine-tune it with Oviya's voice. If not, we need a different approach.")
    print("\n" + "="*70)


def check_prerequisites():
    """Check if required components are available."""
    print("\n🔍 Checking prerequisites...")
    
    issues = []
    
    # Check CSM service
    try:
        import requests
        response = requests.get("http://localhost:6006/health", timeout=5)
        if response.status_code == 200:
            print("✅ CSM service running")
        else:
            issues.append("CSM service not responding properly")
    except:
        issues.append("CSM service not accessible at localhost:6006")
    
    # Check OpenVoice V2
    openvoice_path = Path("external/OpenVoice")
    if openvoice_path.exists():
        print("✅ OpenVoice V2 repository found")
    else:
        print("⚠️ OpenVoice V2 not found (will use mock references)")
    
    # Check required directories
    Path("output/emotion_transfer").mkdir(parents=True, exist_ok=True)
    Path("data/emotion_embeddings").mkdir(parents=True, exist_ok=True)
    Path("temp/reference_audio").mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")
    
    return issues


def main():
    """Main function."""
    # Print header
    print_header()
    
    # Check prerequisites
    issues = check_prerequisites()
    
    if issues:
        print("\n⚠️ Issues detected:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nContinuing with available components...")
    
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70)
    
    # Step 1: Initialize OpenVoice V2 (Teacher)
    print("\n[1/3] Loading OpenVoice V2 (Teacher)...")
    teacher = OpenVoiceEmotionTeacher()
    
    # Generate reference samples
    print("\n📦 Generating emotion reference samples...")
    try:
        teacher.generate_reference_samples("output/emotion_transfer/references")
        print("✅ Reference samples generated")
    except Exception as e:
        print(f"⚠️ Reference generation warning: {e}")
    
    # Step 2: Initialize CSM (Student)
    print("\n[2/3] Loading CSM (Student)...")
    student = HybridVoiceEngine(
        csm_url="http://45.78.17.160:6006/generate",
        default_engine="csm"  # Force CSM for testing
    )
    
    # Step 3: Initialize Evaluator
    print("\n[3/3] Initializing Evaluator...")
    evaluator = EmotionTransferEvaluator(
        teacher=teacher,
        student=student,
        config_path="config/emotion_reference.json"
    )
    
    print("\n✅ All components initialized")
    
    # Run evaluation
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    results = evaluator.run_full_evaluation()
    
    # Save results
    output_path = "output/emotion_transfer/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Final recommendations
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if results["successful"] > 0:
        print("\n✅ Evaluation completed!")
        print("\n📁 Check the following files:")
        print(f"   • Audio outputs: output/emotion_transfer/")
        print(f"   • JSON results: {output_path}")
        print("\n🎧 Listen to the audio files:")
        print("   • teacher_*.wav = OpenVoice V2 emotional references")
        print("   • csm_*.wav = CSM's attempt to reproduce those emotions")
        print("\n💡 Compare them to determine if CSM captured the emotion!")
    else:
        print("\n❌ Evaluation failed - check error messages above")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


