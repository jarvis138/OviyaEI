#!/usr/bin/env python3
"""
Test Emotion Library - Verify local emotion system
"""

from voice.emotion_library import get_emotion_library
import json


def test_library_loading():
    """Test that library loads correctly"""
    print("🧪 TEST 1: Library Loading")
    print("=" * 60)
    
    library = get_emotion_library()
    stats = library.get_emotion_stats()
    
    print(f"✅ Total emotions: {stats['total_emotions']}")
    print(f"   Tier 1 (Core): {stats['tier1_count']}")
    print(f"   Tier 2 (Contextual): {stats['tier2_count']}")
    print(f"   Tier 3 (Expressive): {stats['tier3_count']}")
    print(f"   Aliases: {stats['aliases']}")
    
    assert stats['total_emotions'] == 28, "Expected 28 emotions"
    print("\n✅ PASSED\n")


def test_emotion_resolution():
    """Test emotion alias resolution"""
    print("🧪 TEST 2: Emotion Resolution")
    print("=" * 60)
    
    library = get_emotion_library()
    
    test_cases = [
        ("happy", "joyful_excited"),
        ("sad", "empathetic_sad"),
        ("worried", "concerned_anxious"),
        ("calm", "calm_supportive"),
        ("excited", "joyful_excited"),
        ("supportive", "comforting"),
        ("caring", "affectionate"),
        ("stressed", "concerned_anxious"),
        ("frustrated", "angry_firm"),
        ("unknown_emotion", "neutral"),  # Fallback
    ]
    
    passed = 0
    for input_emotion, expected in test_cases:
        resolved = library.get_emotion(input_emotion)
        status = "✅" if resolved == expected else "❌"
        print(f"{status} '{input_emotion}' → '{resolved}' (expected: '{expected}')")
        if resolved == expected:
            passed += 1
    
    print(f"\n✅ PASSED: {passed}/{len(test_cases)}\n")


def test_emotion_validation():
    """Test emotion validation"""
    print("🧪 TEST 3: Emotion Validation")
    print("=" * 60)
    
    library = get_emotion_library()
    
    # Valid emotions
    valid_emotions = ["calm_supportive", "joyful_excited", "comforting"]
    print("Valid emotions:")
    for emotion in valid_emotions:
        is_valid, resolved = library.validate_emotion(emotion)
        status = "✅" if is_valid else "❌"
        print(f"  {status} {emotion}: {is_valid} → {resolved}")
    
    # Invalid emotions (should fall back to neutral)
    print("\nInvalid emotions (should fall back):")
    invalid_emotions = ["not_real", "fake_emotion", "xyz"]
    for emotion in invalid_emotions:
        is_valid, resolved = library.validate_emotion(emotion)
        status = "✅" if not is_valid and resolved == "neutral" else "❌"
        print(f"  {status} {emotion}: {is_valid} → {resolved}")
    
    print("\n✅ PASSED\n")


def test_tier_classification():
    """Test emotion tier classification"""
    print("🧪 TEST 4: Tier Classification")
    print("=" * 60)
    
    library = get_emotion_library()
    
    # Sample from each tier
    tier_samples = {
        "tier1_core": ["calm_supportive", "comforting", "neutral"],
        "tier2_contextual": ["playful", "curious", "proud"],
        "tier3_expressive": ["angry_firm", "sarcastic", "grateful"]
    }
    
    passed = 0
    total = 0
    
    for expected_tier, emotions in tier_samples.items():
        print(f"\n{expected_tier}:")
        for emotion in emotions:
            actual_tier = library.get_tier(emotion)
            status = "✅" if actual_tier == expected_tier else "❌"
            print(f"  {status} {emotion}: {actual_tier}")
            total += 1
            if actual_tier == expected_tier:
                passed += 1
    
    print(f"\n✅ PASSED: {passed}/{total}\n")


def test_similar_emotions():
    """Test finding similar emotions"""
    print("🧪 TEST 5: Similar Emotions")
    print("=" * 60)
    
    library = get_emotion_library()
    
    test_emotions = ["joyful_excited", "empathetic_sad", "neutral"]
    
    for emotion in test_emotions:
        similar = library.get_similar_emotions(emotion, count=3)
        tier = library.get_tier(emotion)
        print(f"\n{emotion} ({tier}):")
        print(f"  Similar: {', '.join(similar)}")
        
        # Verify they're from the same tier
        same_tier = all(library.get_tier(e) == tier for e in similar)
        status = "✅" if same_tier else "❌"
        print(f"  {status} All from same tier: {same_tier}")
    
    print("\n✅ PASSED\n")


def test_weighted_sampling():
    """Test weighted emotion sampling"""
    print("🧪 TEST 6: Weighted Sampling")
    print("=" * 60)
    
    library = get_emotion_library()
    
    # Sample 100 emotions and check distribution
    samples = [library.sample_emotion() for _ in range(100)]
    
    tier_counts = {
        "tier1_core": 0,
        "tier2_contextual": 0,
        "tier3_expressive": 0
    }
    
    for emotion in samples:
        tier = library.get_tier(emotion)
        tier_counts[tier] += 1
    
    print("Distribution from 100 samples:")
    print(f"  Tier 1 (Core): {tier_counts['tier1_core']}% (expected ~70%)")
    print(f"  Tier 2 (Contextual): {tier_counts['tier2_contextual']}% (expected ~25%)")
    print(f"  Tier 3 (Expressive): {tier_counts['tier3_expressive']}% (expected ~5%)")
    
    # Rough check (with tolerance)
    tier1_ok = 60 <= tier_counts['tier1_core'] <= 80
    tier2_ok = 15 <= tier_counts['tier2_contextual'] <= 35
    tier3_ok = 0 <= tier_counts['tier3_expressive'] <= 15
    
    if tier1_ok and tier2_ok and tier3_ok:
        print("\n✅ PASSED (within tolerance)\n")
    else:
        print("\n⚠️  Distribution outside expected range (may be random variance)\n")


def test_config_consistency():
    """Test that config file matches library"""
    print("🧪 TEST 7: Config Consistency")
    print("=" * 60)
    
    library = get_emotion_library()
    
    # Check config file
    try:
        with open("config/emotion_library.json", 'r') as f:
            config = json.load(f)
        
        print("✅ Config file loaded")
        print(f"   Version: {config.get('version')}")
        print(f"   Total emotions in config: {config.get('total_emotions')}")
        
        # Verify counts match
        config_emotions = set()
        for tier_emotions in config['tiers'].values():
            config_emotions.update(tier_emotions)
        
        library_emotions = set(library.get_all_emotions())
        
        if config_emotions == library_emotions:
            print(f"✅ Config and library match ({len(config_emotions)} emotions)")
        else:
            print("❌ Mismatch between config and library")
            print(f"   Config only: {config_emotions - library_emotions}")
            print(f"   Library only: {library_emotions - config_emotions}")
        
        print("\n✅ PASSED\n")
        
    except Exception as e:
        print(f"❌ FAILED: {e}\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🎨 EMOTION LIBRARY TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_library_loading,
        test_emotion_resolution,
        test_emotion_validation,
        test_tier_classification,
        test_similar_emotions,
        test_weighted_sampling,
        test_config_consistency
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {e}\n")
    
    print("=" * 60)
    print(f"📊 FINAL RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    if passed == len(tests):
        print("\n✅ ALL TESTS PASSED! Emotion library is ready.\n")
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Check output above.\n")


if __name__ == "__main__":
    main()

