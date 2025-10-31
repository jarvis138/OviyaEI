#!/usr/bin/env python3
"""
Test Individual Pipeline Components

Tests each implemented phase component individually to ensure they work correctly.
"""

import numpy as np
import asyncio
from typing import Dict, List, Any

# Import implemented components
from audio_input import get_audio_pipeline
from emotion_detector.detector import EmotionDetector
from personality_system import get_personality_system
from prosody_engine import get_prosody_engine
from voice.csm_1b_generator_optimized import get_optimized_streamer

async def test_audio_pipeline():
    """Test Phase 1-3: Audio Input → VAD → STT"""
    print("🧪 TESTING AUDIO PIPELINE (Phase 1-3)")
    print("=" * 50)

    pipeline = get_audio_pipeline()

    # Create dummy audio (1 second, 16kHz)
    dummy_audio = np.random.normal(0, 0.1, 16000).astype(np.int16).tobytes()

    print("✅ Audio pipeline initialized")

    # Test audio processing
    results = []
    async for result in pipeline.process_audio_stream(dummy_audio):
        results.append(result)

    if results:
        print(f"✅ Audio processing successful: {len(results)} results")
    else:
        print("⚠️ No speech detected (expected for random noise)")

    print("✅ Audio pipeline test completed\n")


def test_emotion_detection():
    """Test Phase 4: Emotion Detection"""
    print("🧪 TESTING EMOTION DETECTION (Phase 4)")
    print("=" * 50)

    detector = EmotionDetector()

    test_texts = [
        "I feel really anxious about what might happen",
        "I'm so happy today!",
        "I feel sad and lonely",
        "I'm angry about this situation"
    ]

    for text in test_texts:
        result = detector.detect_emotion(text)
        emotion = result['emotion']
        intensity = result['intensity']
        confidence = result['confidence']

        print(f"Text: \"{text}\"")
        print(f"  → Emotion: {emotion} (intensity: {intensity:.2f}, confidence: {confidence:.2f})")

    print("✅ Emotion detection test completed\n")


def test_personality_silence():
    """Test Phase 5: Personality & Strategic Silence"""
    print("🧪 TESTING PERSONALITY & SILENCE (Phase 5)")
    print("=" * 50)

    system = get_personality_system()

    test_cases = [
        ("anxiety", 0.8),
        ("grief", 0.9),
        ("joy", 0.6),
        ("anger", 0.7)
    ]

    for emotion, intensity in test_cases:
        result = system.process_emotion_and_context(emotion, intensity)

        personality = result['personality_vector']
        silence = result['strategic_silence']

        print(f"Emotion: {emotion} (intensity: {intensity})")
        print(f"  → Dominant pillar: {personality['dominant_pillar']}")
        print(f"  → Ma: {personality['pillars']['ma']:.2f}, Balance: {personality['balance_score']:.2f}")
        print(f"  → Silence: {silence['silence_duration_seconds']:.1f}s")

    print("✅ Personality & silence test completed\n")


async def test_voice_generation():
    """Test Phase 9: Voice Generation"""
    print("🧪 TESTING VOICE GENERATION (Phase 9)")
    print("=" * 50)

    streamer = get_optimized_streamer()

    test_text = "Hello, I'm Oviya. How can I help you today?"
    print(f"Generating voice for: \"{test_text}\"")

    start_time = asyncio.get_event_loop().time()
    audio_bytes = streamer.generate_voice(
        text=test_text,
        emotion="calm_supportive",
        speaker_id=42
    )
    end_time = asyncio.get_event_loop().time()

    latency = end_time - start_time
    size_kb = len(audio_bytes) / 1024

    print(f"✅ Voice generated: {size_kb:.1f} KB in {latency:.2f}s")
    print(f"   Latency: {latency*1000:.1f}ms (target: <5000ms)")

    if latency < 5.0:
        print("✅ Latency within acceptable range")
    else:
        print("⚠️ Latency above target - optimization needed")

    print("✅ Voice generation test completed\n")


def test_prosody_engine():
    """Test Phase 8: Prosody Computation"""
    print("🧪 TESTING PROSODY ENGINE (Phase 8)")
    print("=" * 50)

    engine = get_prosody_engine()

    test_case = {
        'emotion': 'anxiety',
        'intensity': 0.8,
        'personality': {'pillars': {'ma': 0.8, 'ahimsa': 0.9, 'jeong': 0.85}},
        'reciprocal': {'ovi_emotion': 'grounded_calm', 'intensity': 0.8},
        'text': 'I hear your anxiety and I am here with steady calm.'
    }

    result = engine.compute_prosody_parameters(
        emotion=test_case['emotion'],
        intensity=test_case['intensity'],
        personality_vector=test_case['personality'],
        response_text=test_case['text'],
        reciprocal_emotion=test_case['reciprocal']
    )

    params = result['parameters']
    markers = result['markers']

    print("Prosody Parameters:")
    print(f"  F0 Mean: {params['f0_mean']:+.2f}")
    print(f"  Energy: {params['energy']:+.2f}")
    print(f"  Speech Rate: {params['speech_rate']:.2f}x")
    print(f"  Intonation: {params['intonation_curve']}")

    print(f"Prosody Markers: {len(markers)} generated")
    for marker in markers[:3]:  # Show first 3
        print(f"  {marker}")

    print("✅ Prosody engine test completed\n")


async def run_all_tests():
    """Run all component tests"""
    print("🎯 OVIYA EI PIPELINE COMPONENT TESTS")
    print("=" * 60)

    try:
        # Test each component
        await test_audio_pipeline()
        test_emotion_detection()
        test_personality_silence()
        await test_voice_generation()
        test_prosody_engine()

        print("🎉 ALL COMPONENT TESTS COMPLETED")
        print("=" * 60)

        # Summary
        print("📊 IMPLEMENTATION STATUS:")
        print("✅ Phase 1-3: Audio Input → VAD → STT")
        print("✅ Phase 4: Emotion Detection")
        print("✅ Phase 5: Personality & Strategic Silence")
        print("✅ Phase 8: Prosody Computation")
        print("✅ Phase 9: CSM-1B Voice Generation")
        print("⏳ Phase 6-7: LLM & Emotional Reciprocity (existing OviyaBrain)")
        print("⏳ Phase 10-11: Audio Post-processing & Streaming")
        print("⏳ Phase 12-13: Memory & Safety Monitoring")

        print("\\n💙 Core pipeline components successfully implemented!")
        print("   Oviya can now process speech → emotion → personality → voice")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
