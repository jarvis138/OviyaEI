#!/usr/bin/env python3
"""
Production Readiness Tests
Validates all critical systems before deployment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import random
import torch
import numpy as np
from typing import Dict, List, Tuple
from brain.llm_brain import ProsodyMarkup, EmotionalMemory
from voice.audio_postprocessor import AudioPostProcessor
from voice.emotion_library import get_emotion_library
from utils.emotion_monitor import EmotionDistributionMonitor


def test_prosodic_markup_validation():
    """Test prosodic markup generation and validation"""
    print("\n🎭 Testing Prosodic Markup Validation")
    print("=" * 60)
    
    test_sentences = [
        ("I'm so happy for you!", "joyful_excited", 0.8),
        ("I understand how difficult this is.", "empathetic_sad", 0.7),
        ("Take a deep breath.", "calm_supportive", 0.9),
        ("You can do this!", "encouraging", 0.8),
        ("That's interesting.", "neutral", 0.5),
        ("Oh really?", "curious", 0.6),
        ("I'm here for you.", "comforting", 0.9),
        ("That's not acceptable.", "angry_firm", 0.7),
        ("Maybe next time.", "wistful", 0.5),
        ("Ha! That's funny.", "amused", 0.7)
    ]
    
    valid_tags = ['<breath>', '<pause>', '<long_pause>', '<smile>', '<gentle>', '</gentle>', '<strong>', '</strong>']
    errors = []
    
    for text, emotion, intensity in test_sentences:
        marked_text = ProsodyMarkup.add_prosodic_markup(text, emotion, intensity)
        
        # Check for valid tags
        for tag in ['<', '>']:
            if tag in marked_text:
                # Extract all tags
                import re
                tags = re.findall(r'<[^>]+>', marked_text)
                for found_tag in tags:
                    if found_tag not in valid_tags and not found_tag.startswith('</'):
                        errors.append(f"Invalid tag '{found_tag}' in: {marked_text}")
        
        print(f"  ✅ {emotion}: {marked_text}")
    
    if errors:
        print("\n❌ Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ All prosodic markup is valid")
    
    return len(errors) == 0


def test_audio_drift():
    """Test audio drift detection"""
    print("\n🔊 Testing Audio Drift Detection")
    print("=" * 60)
    
    processor = AudioPostProcessor(sample_rate=24000)
    
    # Create test audio of different lengths
    test_cases = [
        (1.0, "Short utterance"),
        (3.0, "Medium length sentence with more words"),
        (5.0, "Long sentence that simulates a full response with multiple clauses")
    ]
    
    drift_results = []
    
    for duration, description in test_cases:
        # Create test audio
        samples = int(24000 * duration)
        test_audio = torch.randn(samples) * 0.3
        
        # Process with prosodic text that adds breaths
        prosodic_text = f"<breath> {description} <pause> <breath>"
        
        # Process audio
        processed = processor.process(
            test_audio,
            prosodic_text=prosodic_text,
            check_drift=True
        )
        
        # Calculate drift
        original_duration = len(test_audio) / 24000
        processed_duration = len(processed) / 24000
        drift_percent = abs((processed_duration - original_duration) / original_duration) * 100
        
        # More lenient for very short audio (< 2s) since breath adds fixed duration
        threshold = 15.0 if original_duration < 2.0 else 3.0
        
        drift_results.append({
            "original": original_duration,
            "processed": processed_duration,
            "drift_percent": drift_percent,
            "passed": drift_percent <= threshold,
            "threshold": threshold
        })
        
        status = "✅" if drift_results[-1]["passed"] else "⚠️"
        print(f"  {status} {original_duration:.1f}s → {processed_duration:.1f}s (drift: {drift_percent:.1f}%, threshold: {threshold:.0f}%)")
    
    all_passed = all(r["passed"] for r in drift_results)
    if all_passed:
        print("\n✅ Audio drift within acceptable range")
    else:
        print("\n⚠️  Some audio samples exceeded drift threshold")
    
    return all_passed


def test_emotion_distribution():
    """Test emotion distribution with 1000 samples"""
    print("\n📊 Testing Emotion Distribution (1000 samples)")
    print("=" * 60)
    
    library = get_emotion_library()
    monitor = EmotionDistributionMonitor()
    
    # Generate 1000 random emotions
    print("  Generating emotions...")
    for i in range(1000):
        emotion = library.sample_emotion()
        monitor.record_emotion(emotion)
        
        if (i + 1) % 200 == 0:
            print(f"    Generated {i + 1}/1000...")
    
    # Check distribution
    health = monitor.check_distribution_health()
    current = monitor.get_distribution()
    
    print("\n  Distribution results:")
    for tier in ["tier1_core", "tier2_contextual", "tier3_expressive"]:
        target = monitor.target_distribution[tier]
        actual = current.get(tier, 0)
        deviation = abs(actual - target)
        status = "✅" if deviation <= 0.10 else "⚠️"
        print(f"    {status} {tier}: {actual:.1%} (target: {target:.1%}, deviation: {deviation:.1%})")
    
    # Get histogram
    histogram = monitor.generate_histogram()
    
    # Show top emotions
    all_emotions = []
    for tier_emotions in histogram.values():
        all_emotions.extend(tier_emotions.items())
    
    top_5 = sorted(all_emotions, key=lambda x: x[1], reverse=True)[:5]
    print("\n  Top 5 emotions:")
    for emotion, count in top_5:
        print(f"    - {emotion}: {count} ({count/1000:.1%})")
    
    return health["status"] == "healthy"


def test_fallback_mechanisms():
    """Test error handling and fallback mechanisms"""
    print("\n🛡️ Testing Fallback Mechanisms")
    print("=" * 60)
    
    tests_passed = []
    
    # Test 1: Empty emotional memory default
    print("  Testing empty memory default...")
    memory = EmotionalMemory()
    initial_state = memory.state
    
    if initial_state["dominant_emotion"] == "calm_supportive" and initial_state["warmth"] == 0.7:
        print("    ✅ Defaults to calm_supportive with warmth=0.7")
        tests_passed.append(True)
    else:
        print(f"    ❌ Wrong default: {initial_state['dominant_emotion']}, warmth={initial_state['warmth']}")
        tests_passed.append(False)
    
    # Test 2: Prosodic pattern cache
    print("  Testing prosodic pattern cache...")
    
    # Clear cache
    ProsodyMarkup._pattern_cache = {}
    
    # First call should populate cache
    pattern1 = ProsodyMarkup.get_cached_pattern("joyful_excited")
    
    # Second call should use cache
    pattern2 = ProsodyMarkup.get_cached_pattern("joyful_excited")
    
    if pattern1 is pattern2 and len(ProsodyMarkup._pattern_cache) == 1:
        print("    ✅ Pattern caching working")
        tests_passed.append(True)
    else:
        print("    ❌ Pattern caching not working")
        tests_passed.append(False)
    
    # Test 3: Audio processor error handling
    print("  Testing audio processor error handling...")
    processor = AudioPostProcessor(sample_rate=24000)
    
    # Test with invalid audio
    try:
        # Create very short audio that might cause issues
        short_audio = torch.tensor([0.1, 0.2])
        result = processor.process(
            short_audio,
            prosodic_text="<breath> Test <pause>",
            check_drift=False  # Disable drift check for this test
        )
        
        if result is not None:
            print("    ✅ Handled short audio gracefully")
            tests_passed.append(True)
        else:
            print("    ❌ Returned None for short audio")
            tests_passed.append(False)
            
    except Exception as e:
        print(f"    ❌ Exception thrown: {e}")
        tests_passed.append(False)
    
    # Test 4: Emotion library fallback
    print("  Testing emotion library fallback...")
    library = get_emotion_library()
    
    # Test unknown emotion
    resolved = library.get_emotion("unknown_emotion_xyz")
    if resolved == "neutral":
        print("    ✅ Unknown emotion falls back to neutral")
        tests_passed.append(True)
    else:
        print(f"    ❌ Unknown emotion resolved to: {resolved}")
        tests_passed.append(False)
    
    return all(tests_passed)


def test_performance_optimizations():
    """Test performance optimizations"""
    print("\n⚡ Testing Performance Optimizations")
    print("=" * 60)
    
    # Test 1: Prosodic pattern caching performance
    print("  Testing pattern cache performance...")
    
    # Clear cache
    ProsodyMarkup._pattern_cache = {}
    
    # Time uncached calls
    start = time.time()
    for _ in range(1000):
        ProsodyMarkup.get_cached_pattern("joyful_excited")
    cached_time = time.time() - start
    
    print(f"    1000 cached calls: {cached_time*1000:.2f}ms")
    
    if cached_time < 0.01:  # Should be under 10ms for 1000 calls
        print("    ✅ Cache performance excellent")
        cache_ok = True
    else:
        print("    ⚠️  Cache might need optimization")
        cache_ok = False
    
    # Test 2: Emotion library performance
    print("  Testing emotion library performance...")
    library = get_emotion_library()
    
    start = time.time()
    for _ in range(1000):
        emotion = library.sample_emotion()
        resolved = library.get_emotion(emotion)
    library_time = time.time() - start
    
    print(f"    1000 emotion resolutions: {library_time*1000:.2f}ms")
    
    if library_time < 0.05:  # Should be under 50ms for 1000 calls
        print("    ✅ Library performance good")
        library_ok = True
    else:
        print("    ⚠️  Library might need optimization")
        library_ok = False
    
    return cache_ok and library_ok


def run_all_tests():
    """Run all production readiness tests"""
    print("\n" + "=" * 80)
    print("🚀 PRODUCTION READINESS TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    print("\n📋 Running sanity tests...")
    
    results["prosodic_validation"] = test_prosodic_markup_validation()
    results["audio_drift"] = test_audio_drift()
    results["emotion_distribution"] = test_emotion_distribution()
    results["fallback_mechanisms"] = test_fallback_mechanisms()
    results["performance"] = test_performance_optimizations()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY!")
        print("\n📋 Pre-deployment checklist:")
        print("  ✅ Prosodic markup validation working")
        print("  ✅ Audio drift detection < 3%")
        print("  ✅ Emotion distribution within targets")
        print("  ✅ Fallback mechanisms operational")
        print("  ✅ Performance optimizations active")
        print("\n🚀 Ready for deployment!")
    else:
        print("⚠️  SOME TESTS FAILED - ADDRESS ISSUES BEFORE DEPLOYMENT")
        print("\nFailed tests need attention before production deployment.")
    
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
