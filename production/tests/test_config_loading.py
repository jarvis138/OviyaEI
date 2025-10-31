#!/usr/bin/env python3
"""
Test Configuration File Loading and Validation
==============================================

Tests that configuration files load correctly and contain required keys.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_voice_config_keys():
    """Test voice configuration has required keys"""
    try:
        from config.production_voice_config import VOICE_CONFIG

        required_keys = ["primary_engine", "fallback_engine", "sample_rate"]
        for key in required_keys:
            assert key in VOICE_CONFIG, f"Missing {key} in VOICE_CONFIG"
            assert VOICE_CONFIG[key] is not None, f"{key} is None in VOICE_CONFIG"

        # Validate sample rate
        assert VOICE_CONFIG["sample_rate"] == 24000, f"Expected sample_rate 24000, got {VOICE_CONFIG['sample_rate']}"

        print("✅ Voice config validation passed")
        return True

    except Exception as e:
        print(f"❌ Voice config validation failed: {e}")
        return False

def test_whisperx_config_keys():
    """Test WhisperX configuration has required keys"""
    try:
        from config.whisperx_config import WHISPERX_CONFIG

        required_keys = ["batch_size", "language", "compute_type"]
        for key in required_keys:
            assert key in WHISPERX_CONFIG, f"Missing {key} in WHISPERX_CONFIG"
            assert WHISPERX_CONFIG[key] is not None, f"{key} is None in WHISPERX_CONFIG"

        # Validate reasonable values
        assert isinstance(WHISPERX_CONFIG["batch_size"], int), "batch_size should be integer"
        assert WHISPERX_CONFIG["batch_size"] > 0, "batch_size should be positive"
        assert WHISPERX_CONFIG["language"] in ["en", "multi"], "language should be 'en' or 'multi'"

        print("✅ WhisperX config validation passed")
        return True

    except Exception as e:
        print(f"❌ WhisperX config validation failed: {e}")
        return False

def test_voice_config_fallback():
    """Test voice config fallback when import fails"""
    try:
        # Test that voice module loads its config
        import voice

        # Verify fallback config has required keys
        required_keys = ["primary_engine", "fallback_engine", "sample_rate"]
        for key in required_keys:
            assert key in voice.VOICE_CONFIG, f"Missing {key} in fallback VOICE_CONFIG"

        print("✅ Voice config fallback validation passed")
        return True

    except Exception as e:
        print(f"❌ Voice config fallback validation failed: {e}")
        return False

def run_config_tests():
    """Run all configuration tests"""
    print("🧪 RUNNING CONFIGURATION TESTS")
    print("=" * 50)

    tests = [
        test_voice_config_keys,
        test_whisperx_config_keys,
        test_voice_config_fallback
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    print(f"\n📊 RESULTS: {passed}/{total} config tests passed")

    if passed == total:
        print("🎉 ALL CONFIG TESTS PASSED!")
        return True
    else:
        print("❌ SOME CONFIG TESTS FAILED")
        return False

if __name__ == "__main__":
    import sys
    success = run_config_tests()
    sys.exit(0 if success else 1)
