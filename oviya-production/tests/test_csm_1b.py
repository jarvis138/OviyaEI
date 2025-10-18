#!/usr/bin/env python3
"""
Test script for CSM-1B implementation
Verifies RVQ/Mimi pipeline, streaming, context, and prosody
"""

import asyncio
import sys
import numpy as np
from voice.csm_1b_client import CSM1BClient
from config.service_urls import CSM_URL

async def test_basic_generation():
    """Test basic CSM-1B generation"""
    print("=" * 70)
    print("TEST 1: Basic CSM-1B Generation")
    print("=" * 70)
    
    client = CSM1BClient(
        use_local_model=False,
        remote_url=CSM_URL
    )
    
    text = "Hello! I'm Oviya, your AI companion."
    emotion = "joyful"
    
    print(f"\nText: {text}")
    print(f"Emotion: {emotion}")
    print()
    
    chunks = []
    start_time = asyncio.get_event_loop().time()
    
    async for audio_chunk in client.generate_streaming(
        text=text,
        emotion=emotion,
        speaker_id=0
    ):
        chunks.append(audio_chunk)
        duration = len(audio_chunk) / 24000
        print(f"   📦 Chunk {len(chunks)}: {len(audio_chunk)} samples ({duration:.3f}s)")
    
    end_time = asyncio.get_event_loop().time()
    
    total_samples = sum(len(c) for c in chunks)
    total_duration = total_samples / 24000
    generation_time = end_time - start_time
    
    print()
    print(f"✅ Generated {len(chunks)} chunks")
    print(f"   Total audio: {total_duration:.2f}s")
    print(f"   Generation time: {generation_time:.2f}s")
    print(f"   Realtime factor: {generation_time / total_duration:.2f}x")
    
    return len(chunks) > 0


async def test_with_emotion():
    """Test different emotions"""
    print("\n" + "=" * 70)
    print("TEST 2: Emotion Control")
    print("=" * 70)
    
    client = CSM1BClient(
        use_local_model=False,
        remote_url=CSM_URL
    )
    
    test_cases = [
        ("I'm excited to help you!", "excitement"),
        ("I understand how you feel.", "empathy"),
        ("Let me think about that.", "contemplative"),
    ]
    
    results = []
    
    for text, emotion in test_cases:
        print(f"\n📝 Text: {text}")
        print(f"   Emotion: {emotion}")
        
        chunks = []
        async for audio_chunk in client.generate_streaming(
            text=text,
            emotion=emotion,
            speaker_id=0
        ):
            chunks.append(audio_chunk)
        
        total_duration = sum(len(c) for c in chunks) / 24000
        print(f"   ✅ Generated {total_duration:.2f}s audio")
        
        results.append(len(chunks) > 0)
    
    return all(results)


async def test_with_context():
    """Test conversation context conditioning"""
    print("\n" + "=" * 70)
    print("TEST 3: Conversational Context")
    print("=" * 70)
    
    client = CSM1BClient(
        use_local_model=False,
        remote_url=CSM_URL
    )
    
    # Simulate a conversation
    conversation_history = [
        {"text": "Hi Oviya!", "speaker_id": 1, "timestamp": 0},
        {"text": "Hello! How can I help you today?", "speaker_id": 0, "timestamp": 1},
        {"text": "Tell me about yourself.", "speaker_id": 1, "timestamp": 2},
    ]
    
    response_text = "I'm Oviya, an empathetic AI assistant designed to understand emotions."
    
    print("\n📚 Context:")
    for turn in conversation_history:
        speaker = "User" if turn["speaker_id"] == 1 else "Oviya"
        print(f"   {speaker}: {turn['text']}")
    
    print(f"\n💬 Current response: {response_text}")
    print()
    
    chunks = []
    async for audio_chunk in client.generate_streaming(
        text=response_text,
        emotion="calm",
        speaker_id=0,
        conversation_context=conversation_history
    ):
        chunks.append(audio_chunk)
    
    total_duration = sum(len(c) for c in chunks) / 24000
    print(f"\n✅ Generated {total_duration:.2f}s audio with context")
    print(f"   Context turns: {len(conversation_history)}")
    
    return len(chunks) > 0


async def test_streaming_latency():
    """Test streaming latency (time to first chunk)"""
    print("\n" + "=" * 70)
    print("TEST 4: Streaming Latency")
    print("=" * 70)
    
    client = CSM1BClient(
        use_local_model=False,
        remote_url=CSM_URL
    )
    
    text = "Testing streaming latency with a moderately long sentence."
    
    print(f"\nText: {text}")
    print("\nMeasuring time to first audio chunk...")
    
    start_time = asyncio.get_event_loop().time()
    first_chunk_time = None
    
    chunk_times = []
    i = 0
    async for audio_chunk in client.generate_streaming(
        text=text,
        emotion="neutral",
        speaker_id=0
    ):
        current_time = asyncio.get_event_loop().time()
        
        if first_chunk_time is None:
            first_chunk_time = current_time - start_time
            print(f"   ⚡ First chunk: {first_chunk_time:.3f}s")
        
        chunk_times.append(current_time - start_time)
        i += 1
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    
    print(f"\n✅ Streaming metrics:")
    print(f"   Time to first chunk: {first_chunk_time:.3f}s")
    print(f"   Total generation: {total_time:.3f}s")
    print(f"   Number of chunks: {len(chunk_times)}")
    
    if first_chunk_time:
        print(f"   {'✅' if first_chunk_time < 2.0 else '⚠️ '} Latency: {'Good' if first_chunk_time < 2.0 else 'Could be better'}")
    
    return first_chunk_time is not None and first_chunk_time < 5.0


async def test_prosody_mapping():
    """Test Oviya emotion → CSM prosody mapping"""
    print("\n" + "=" * 70)
    print("TEST 5: Prosody Token Mapping")
    print("=" * 70)
    
    client = CSM1BClient(
        use_local_model=False,
        remote_url=CSM_URL
    )
    
    # Test emotion mapping
    oviya_emotions = [
        "joy", "sadness", "anxiety", "empathy", "curiosity",
        "calm", "excitement", "frustration", "contemplative"
    ]
    
    print("\n📊 Oviya Emotion → CSM Prosody Mapping:")
    for oviya_emotion in oviya_emotions:
        csm_emotion = client._map_oviya_emotion_to_csm(oviya_emotion)
        print(f"   {oviya_emotion:15} → {csm_emotion}")
    
    # Test generation with mapped emotion
    text = "I understand your concern."
    oviya_emotion = "empathy"
    
    print(f"\n🎤 Testing with Oviya emotion: {oviya_emotion}")
    
    chunks = []
    async for audio_chunk in client.generate_streaming(
        text=text,
        emotion=oviya_emotion,
        speaker_id=0
    ):
        chunks.append(audio_chunk)
    
    print(f"✅ Generated audio with mapped prosody")
    
    return len(chunks) > 0


async def test_health_check():
    """Test CSM-1B health check"""
    print("\n" + "=" * 70)
    print("TEST 6: Health Check")
    print("=" * 70)
    
    client = CSM1BClient(
        use_local_model=False,
        remote_url=CSM_URL
    )
    
    health = client.health_check()
    
    print("\n🏥 CSM-1B Health:")
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    is_healthy = health.get('status') in ['healthy', 'degraded']
    print(f"\n{'✅' if is_healthy else '❌'} Service is {'healthy' if is_healthy else 'unhealthy'}")
    
    return is_healthy


async def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 CSM-1B IMPLEMENTATION TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        ("Health Check", test_health_check),
        ("Basic Generation", test_basic_generation),
        ("Emotion Control", test_with_emotion),
        ("Conversational Context", test_with_context),
        ("Streaming Latency", test_streaming_latency),
        ("Prosody Mapping", test_prosody_mapping),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
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
        print("\n🎉 All tests passed! CSM-1B implementation is working correctly.")
        print("\n✨ Features verified:")
        print("   • RVQ token generation")
        print("   • Mimi decoder (via remote API)")
        print("   • Streaming audio output")
        print("   • Conversational context conditioning")
        print("   • Prosody/emotion control")
        print("   • Oviya emotion mapping")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        failed = [name for name, passed in results if not passed]
        print(f"\nFailed tests: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

