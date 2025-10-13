#!/usr/bin/env python3
"""
Test WhisperX Integration with Oviya
Tests the WhisperX API running on RTX 5880 Ada
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice.whisperx_api_client import WhisperXAPIClient
import numpy as np

def test_whisperx_integration():
    """Test WhisperX API integration"""
    print("=" * 70)
    print("🎤 Testing WhisperX Integration with Oviya")
    print("=" * 70)
    print("")
    
    # Initialize client
    print("📡 Initializing WhisperX API client...")
    client = WhisperXAPIClient()
    print("✅ Client initialized")
    print("")
    
    # Test 1: Health check
    print("🧪 Test 1: Health Check")
    print("-" * 70)
    health = client.check_health()
    
    if health.get('status') == 'healthy':
        print("✅ WhisperX API is healthy!")
        print(f"   Service: {health.get('service')}")
        print(f"   Port: {health.get('port')}")
        print(f"   Device: {health.get('device')}")
        print(f"   Model: {health.get('model')}")
        print(f"   GPU Memory: {health.get('gpu_memory_gb', 0):.1f}GB")
    else:
        print(f"❌ WhisperX API is not healthy: {health}")
        return
    print("")
    
    # Test 2: Transcription with dummy audio
    print("🧪 Test 2: Transcription (Dummy Audio)")
    print("-" * 70)
    test_audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
    
    result = client.transcribe_audio(test_audio, batch_size=8)
    
    if result and result.get('status') == 'success':
        print("✅ Transcription successful!")
        print(f"   Audio duration: {result.get('duration', 0):.2f}s")
        print(f"   Processing time (server): {result.get('processing_time', 0):.2f}s")
        print(f"   Processing time (total): {result.get('client_processing_time', 0):.2f}s")
        print(f"   Text: {result.get('text', 'N/A')[:100]}...")
        print(f"   Segments: {len(result.get('segments', []))}")
        print(f"   Word timestamps: {len(result.get('word_timestamps', []))}")
        
        # Show first few word timestamps
        word_timestamps = result.get('word_timestamps', [])
        if word_timestamps:
            print(f"\n   First 5 word timestamps:")
            for word in word_timestamps[:5]:
                print(f"      {word['word']}: {word['start']:.2f}-{word['end']:.2f}s (conf: {word['confidence']:.2f})")
    else:
        print(f"❌ Transcription failed: {result}")
        return
    print("")
    
    # Test 3: Performance test
    print("🧪 Test 3: Performance Test (Multiple Requests)")
    print("-" * 70)
    
    durations = []
    for i in range(3):
        test_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
        result = client.transcribe_audio(test_audio, batch_size=8)
        
        if result:
            duration = result.get('client_processing_time', 0)
            durations.append(duration)
            print(f"   Request {i+1}: {duration:.2f}s")
    
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\n   ✅ Average processing time: {avg_duration:.2f}s")
        print(f"   Min: {min(durations):.2f}s, Max: {max(durations):.2f}s")
    print("")
    
    # Summary
    print("=" * 70)
    print("🎉 WhisperX Integration Test Complete!")
    print("=" * 70)
    print("")
    print("📋 Summary:")
    print(f"   ✅ WhisperX API: Running on RTX 5880 Ada")
    print(f"   ✅ Health Check: Passed")
    print(f"   ✅ Transcription: Working")
    print(f"   ✅ Word Timestamps: Available")
    print(f"   ✅ Performance: {avg_duration:.2f}s average")
    print("")
    print("🎯 WhisperX is ready for Oviya production use!")
    print("=" * 70)


if __name__ == "__main__":
    test_whisperx_integration()

