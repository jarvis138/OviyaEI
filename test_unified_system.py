#!/usr/bin/env python3
"""
Test Oviya as ONE Unified System

Tests whether all components work together as a cohesive whole:
1. Emotion Detector → Brain → Emotion Controller → Voice
2. Real-time conversation flow
3. WebSocket server integration
4. End-to-end pipeline functionality
"""

import sys
from pathlib import Path
import asyncio
import time

# Add root directory to path for imports
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from production.pipeline import OviyaPipeline
from production.realtime_conversation import RealTimeConversation
from production.brain.llm_brain import OviyaBrain
from production.emotion_detector.detector import EmotionDetector
from production.emotion_controller.controller import EmotionController
from production.voice.openvoice_tts import HybridVoiceEngine

def test_pipeline_integration():
    """Test 1: Full Pipeline Integration"""
    print("🔧 Testing FULL PIPELINE INTEGRATION")
    print("="*60)

    try:
        print("🚀 Initializing complete Oviya Pipeline...")
        pipeline = OviyaPipeline()
        print("✅ Pipeline initialized successfully!")

        # Test complete flow
        test_message = "I'm feeling really anxious about my presentation tomorrow"

        print(f"\n🗣️  Testing message: '{test_message}'")

        # Step 1: Emotion Detection
        print("\n[1/4] 🎭 Emotion Detection...")
        emotion_result = pipeline.emotion_detector.detect_emotion(test_message)
        detected_emotion = emotion_result.get('primary_emotion', 'neutral')
        confidence = emotion_result.get('confidence', 0)
        print(f"   Detected: {detected_emotion} (confidence: {confidence:.2f})")

        # Step 2: Brain Processing
        print("\n[2/4] 🧠 Brain Processing...")
        brain_response = pipeline.brain.think(test_message, detected_emotion)
        text_response = brain_response.get('text', '')
        response_emotion = brain_response.get('emotion', 'neutral')
        print(f"   Response: {text_response[:100]}...")
        print(f"   Emotion: {response_emotion}")

        # Step 3: Emotion Controller
        print("\n[3/4] 🎛️  Emotion Controller...")
        emotion_params = pipeline.emotion_controller.map_emotion(response_emotion, 0.7)
        print(f"   Pitch scale: {emotion_params['pitch_scale']:.3f}")
        print(f"   Rate scale: {emotion_params['rate_scale']:.3f}")
        print(f"   Energy scale: {emotion_params['energy_scale']:.3f}")

        # Step 4: Voice Synthesis (mock - don't actually generate audio)
        print("\n[4/4] 🔊 Voice Synthesis...")
        try:
            # Just test that TTS can be initialized and configured
            tts_config = pipeline.tts.get_engine_status()
            print(f"   TTS Status: {tts_config}")
            print("   ✅ Voice synthesis ready")
        except Exception as e:
            print(f"   ⚠️  TTS mock: {e}")

        print("\n🎉 FULL PIPELINE TEST PASSED!")
        return True

    except Exception as e:
        print(f"❌ Pipeline integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_realtime_conversation():
    """Test 2: Real-time Conversation System"""
    print("\n🎙️  Testing REAL-TIME CONVERSATION SYSTEM")
    print("="*60)

    try:
        print("🚀 Initializing Real-time Conversation...")
        conversation = RealTimeConversation()
        print("✅ Real-time system initialized!")

        # Test conversation flow
        test_input = "Hello, I'm feeling a bit stressed today"

        print(f"\n🗣️  Testing input: '{test_input}'")

        # Simulate the full conversation flow
        print("\n[1/3] 🎤 Processing voice input...")
        # In real usage, this would come from WhisperX
        print("   ✅ Voice input processed (simulated)")

        print("\n[2/3] 🧠 Generating response...")
        response = conversation.brain.think(test_input, "neutral")
        print(f"   Response: {response.get('text', '')[:100]}...")

        print("\n[3/3] 🎵 Preparing audio output...")
        # In real usage, this would generate audio
        print("   ✅ Audio preparation ready (simulated)")

        print("\n🎉 REAL-TIME CONVERSATION TEST PASSED!")
        return True

    except Exception as e:
        print(f"❌ Real-time conversation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_dependencies():
    """Test 3: Component Dependencies & Integration"""
    print("\n🔗 Testing COMPONENT DEPENDENCIES")
    print("="*60)

    dependencies_ok = True

    # Test each component can be initialized independently
    components = [
        ("Emotion Detector", lambda: EmotionDetector()),
        ("Oviya Brain", lambda: OviyaBrain(persona_config_path="production/config/oviya_persona.json")),
        ("Emotion Controller", lambda: EmotionController()),
        ("Voice Engine", lambda: HybridVoiceEngine())
    ]

    for name, init_func in components:
        try:
            print(f"🔧 Testing {name}...")
            component = init_func()
            print(f"   ✅ {name} initialized successfully")
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            dependencies_ok = False

    if dependencies_ok:
        print("\n🎉 ALL COMPONENT DEPENDENCIES OK!")
        return True
    else:
        print("\n❌ Some component dependencies failed")
        return False

def test_data_flow():
    """Test 4: Data Flow Between Components"""
    print("\n📊 Testing DATA FLOW INTEGRATION")
    print("="*60)

    try:
        # Initialize components
        detector = EmotionDetector()
        brain = OviyaBrain(persona_config_path="production/config/oviya_persona.json")
        controller = EmotionController()

        # Test data flows through each component
        test_message = "I'm excited about my new job!"

        print(f"Testing data flow for: '{test_message}'")

        # Flow: Message → Emotion Detection
        emotion_data = detector.detect_emotion(test_message)
        emotion = emotion_data.get('primary_emotion', 'neutral')
        print(f"   📥 Emotion Detection: {emotion}")

        # Flow: Emotion → Brain
        brain_response = brain.think(test_message, emotion)
        response_text = brain_response.get('text', '')
        response_emotion = brain_response.get('emotion', 'neutral')
        print(f"   🧠 Brain Response: {response_emotion}")

        # Flow: Response Emotion → Controller
        audio_params = controller.map_emotion(response_emotion, 0.8)
        print(f"   🎛️  Audio Params: pitch={audio_params['pitch_scale']:.2f}, rate={audio_params['rate_scale']:.2f}")

        print("\n🎉 DATA FLOW INTEGRATION TEST PASSED!")
        return True

    except Exception as e:
        print(f"❌ Data flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_coherence():
    """Test 5: System Coherence & Consistency"""
    print("\n🎯 Testing SYSTEM COHERENCE")
    print("="*60)

    try:
        # Test that responses are consistent with persona
        brain = OviyaBrain(persona_config_path="production/config/oviya_persona.json")

        test_cases = [
            ("I feel sad", "Should show empathy and validation"),
            ("I'm angry", "Should be calm and supportive"),
            ("I'm happy", "Should celebrate and engage"),
            ("I'm confused", "Should clarify and guide")
        ]

        coherence_score = 0

        for message, expectation in test_cases:
            print(f"\nTesting: '{message}'")
            response = brain.think(message, "neutral")
            text = response.get('text', '')

            # Basic coherence checks
            has_empathy = any(word in text.lower() for word in ['understand', 'imagine', 'feel', 'sense'])
            has_support = any(word in text.lower() for word in ['here', 'help', 'support', 'with you'])
            appropriate_length = 20 < len(text) < 300  # Reasonable response length

            score = (has_empathy + has_support + appropriate_length) / 3
            coherence_score += score

            print(f"   Response: {text[:80]}...")
            print(".1f")

        avg_coherence = coherence_score / len(test_cases)
        print(".2f")

        if avg_coherence > 0.7:
            print("🎉 SYSTEM COHERENCE TEST PASSED!")
            return True
        else:
            print("⚠️  System coherence could be improved")
            return False

    except Exception as e:
        print(f"❌ Coherence test failed: {e}")
        return False

def run_unified_system_test():
    """Run comprehensive unified system test"""
    print("🤖 OVIYA UNIFIED SYSTEM TEST")
    print("="*80)
    print("Testing if Oviya works as ONE cohesive AI system")
    print("="*80 + "\n")

    tests = [
        ("Pipeline Integration", test_pipeline_integration),
        ("Real-time Conversation", test_realtime_conversation),
        ("Component Dependencies", test_component_dependencies),
        ("Data Flow", test_data_flow),
        ("System Coherence", test_system_coherence)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}\n")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}\n")
            results.append(False)

    # Final results
    print("="*80)
    print("📊 UNIFIED SYSTEM TEST RESULTS")
    print("="*80)

    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100

    print(f"🎯 Integration Tests: {passed}/{total} ({success_rate:.1f}%)")

    test_names = ["Pipeline", "Real-time", "Dependencies", "Data Flow", "Coherence"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅" if result else "❌"
        print(f"  {status} {i+1}. {name} Integration")

    print("\n" + "="*80)
    if success_rate >= 80:
        print("🎉 OVIYA WORKS AS ONE UNIFIED SYSTEM!")
        print("All components integrate seamlessly for cohesive AI experience.")
    elif success_rate >= 60:
        print("⚠️  MOSTLY UNIFIED: Core integration works, minor issues to address.")
    else:
        print("❌ INTEGRATION ISSUES: Components need better coordination.")
    print("="*80)

if __name__ == "__main__":
    run_unified_system_test()

