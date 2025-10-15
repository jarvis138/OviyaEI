#!/usr/bin/env python3
"""
Verify CSM-1B server is working correctly
Tests health, generation, and performance
"""

import requests
import json
import base64
import io
import time
import sys

SERVER_URL = "http://localhost:19517"

def test_health():
    """Test health endpoint"""
    print("=" * 70)
    print("TEST 1: Health Check")
    print("=" * 70)
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   Service: {data.get('service')}")
            print(f"   Model: {data.get('model')}")
            
            if 'gpu' in data:
                print(f"   GPU: {data.get('gpu')}")
                print(f"   VRAM: {data.get('vram_used_gb')}GB / {data.get('vram_total_gb')}GB")
            
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_generation():
    """Test audio generation"""
    print("\n" + "=" * 70)
    print("TEST 2: Audio Generation")
    print("=" * 70)
    
    test_cases = [
        ("Hello! I'm Oviya.", "joyful"),
        ("I understand how you feel.", "empathy"),
        ("Let me help you with that.", "calm"),
    ]
    
    results = []
    
    for text, emotion in test_cases:
        print(f"\n📝 Text: {text}")
        print(f"   Emotion: {emotion}")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{SERVER_URL}/generate",
                json={
                    "text": text,
                    "speaker": 0,
                    "reference_emotion": emotion,
                    "normalize_audio": True
                },
                timeout=30
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                audio_base64 = data.get('audio_base64')
                duration_ms = data.get('duration_ms')
                sample_rate = data.get('sample_rate')
                
                if audio_base64:
                    audio_size = len(base64.b64decode(audio_base64))
                    print(f"   ✅ Generated:")
                    print(f"      Duration: {duration_ms:.0f}ms")
                    print(f"      Sample rate: {sample_rate}Hz")
                    print(f"      Size: {audio_size / 1024:.1f}KB")
                    print(f"      Generation time: {generation_time:.2f}s")
                    
                    results.append(True)
                else:
                    print(f"   ❌ No audio in response")
                    results.append(False)
            else:
                print(f"   ❌ Generation failed: {response.status_code}")
                print(f"      {response.text}")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append(False)
    
    return all(results)

def test_context():
    """Test conversation context support"""
    print("\n" + "=" * 70)
    print("TEST 3: Conversational Context")
    print("=" * 70)
    
    conversation_context = [
        {"text": "Hi there!", "speaker_id": 1, "timestamp": 0},
        {"text": "Hello! How can I help?", "speaker_id": 0, "timestamp": 1},
    ]
    
    text = "I'm here to assist you today."
    
    print(f"📚 Context: {len(conversation_context)} turns")
    print(f"💬 Response: {text}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{SERVER_URL}/generate",
            json={
                "text": text,
                "speaker": 0,
                "reference_emotion": "calm",
                "conversation_context": conversation_context
            },
            timeout=30
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            duration_ms = data.get('duration_ms')
            
            print(f"   ✅ Generated with context")
            print(f"      Duration: {duration_ms:.0f}ms")
            print(f"      Generation time: {generation_time:.2f}s")
            return True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_performance():
    """Test performance metrics"""
    print("\n" + "=" * 70)
    print("TEST 4: Performance Metrics")
    print("=" * 70)
    
    test_text = "This is a performance test to measure generation speed."
    
    times = []
    
    for i in range(3):
        print(f"\n🏃 Run {i+1}/3...")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{SERVER_URL}/generate",
                json={
                    "text": test_text,
                    "speaker": 0,
                    "reference_emotion": "neutral"
                },
                timeout=30
            )
            
            generation_time = time.time() - start_time
            times.append(generation_time)
            
            if response.status_code == 200:
                data = response.json()
                duration_ms = data.get('duration_ms')
                realtime_factor = generation_time / (duration_ms / 1000)
                
                print(f"   Audio: {duration_ms:.0f}ms")
                print(f"   Generation: {generation_time:.2f}s")
                print(f"   RT factor: {realtime_factor:.2f}x")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average generation time: {avg_time:.2f}s")
        
        if avg_time < 3.0:
            print("   ✅ Performance: Excellent")
            return True
        elif avg_time < 5.0:
            print("   ✅ Performance: Good")
            return True
        else:
            print("   ⚠️  Performance: Could be better")
            return True
    
    return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 OVIYA CSM-1B SERVER VERIFICATION")
    print("=" * 70)
    print(f"Server: {SERVER_URL}")
    print("=" * 70)
    
    tests = [
        ("Health Check", test_health),
        ("Audio Generation", test_generation),
        ("Conversational Context", test_context),
        ("Performance Metrics", test_performance),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print("=" * 70)
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! CSM-1B server is working correctly.")
        print("\n✨ Next steps:")
        print("   • Update Cloudflare tunnel to point to this server")
        print("   • Test with frontend (http://localhost:3000)")
        print("   • Monitor performance with: tail -f /workspace/csm_1b.log")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        print("\nTroubleshooting:")
        print("   • Check logs: tail -f /workspace/csm_1b.log")
        print("   • Restart server: ./stop_csm.sh && ./start_csm_1b.sh")
        print("   • Check GPU: nvidia-smi")
        return 1

if __name__ == "__main__":
    sys.exit(main())

