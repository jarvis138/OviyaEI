#!/usr/bin/env python3
"""
Oviya Production Integration Test

Tests all four layers of the Oviya pipeline:
1. Emotion Detector
2. Brain (LLM)
3. Emotion Controller
4. Voice (OpenVoiceV2)
"""

import sys
from pathlib import Path
import time

# Add root directory to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from production.emotion_detector.detector import EmotionDetector
from production.brain.llm_brain import OviyaBrain
from production.emotion_controller.controller import EmotionController
from production.voice.openvoice_tts import HybridVoiceEngine


def test_emotion_detector():
    """Test emotion detection."""
    print("\n🧪 Testing Emotion Detector")
    print("=" * 40)
    
    detector = EmotionDetector()
    
    test_cases = [
        "I'm feeling really stressed about work",
        "I got promoted today! I'm so excited!",
        "I'm feeling sad and lonely",
        "I'm frustrated with everything",
        "Just checking in, how are you?"
    ]
    
    for text in test_cases:
        result = detector.detect_emotion(text)
        print(f"Text: '{text}'")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Intensity: {result['intensity']:.2f}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print()


def test_brain():
    """Test LLM brain."""
    print("\n🧪 Testing Brain (LLM)")
    print("=" * 40)
    
    brain = OviyaBrain()
    
    test_cases = [
        ("I'm feeling stressed", "calm_supportive"),
        ("I got promoted!", "joyful_excited"),
        ("I'm sad", "empathetic_sad")
    ]
    
    for user_msg, user_emotion in test_cases:
        print(f"User: {user_msg}")
        print(f"User Emotion: {user_emotion}")
        
        try:
            response = brain.think(user_msg, user_emotion=user_emotion)
            print(f"Oviya: {response['text']}")
            print(f"  Emotion: {response['emotion']}")
            print(f"  Intensity: {response['intensity']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print()


def test_emotion_controller():
    """Test emotion controller."""
    print("\n🧪 Testing Emotion Controller")
    print("=" * 40)
    
    controller = EmotionController()
    
    test_emotions = [
        ("calm_supportive", 0.8),
        ("empathetic_sad", 0.7),
        ("joyful_excited", 0.9),
        ("neutral", 0.5)
    ]
    
    for emotion, intensity in test_emotions:
        params = controller.map_emotion(emotion, intensity)
        print(f"Emotion: {emotion} (intensity: {intensity})")
        print(f"  Style Token: {params['style_token']}")
        print(f"  Pitch: {params['pitch_scale']:.2f}")
        print(f"  Rate: {params['rate_scale']:.2f}")
        print(f"  Energy: {params['energy_scale']:.2f}")
        print()


def test_voice():
    """Test hybrid voice generation."""
    print("\n🧪 Testing Hybrid Voice Engine")
    print("=" * 40)
    
    engine = HybridVoiceEngine()
    
    test_cases = [
        {
            "text": "I'm here with you.",
            "emotion_params": {
                "style_token": "#calm",
                "pitch_scale": 0.9,
                "rate_scale": 0.9,
                "energy_scale": 0.8,
                "emotion_label": "calm_supportive",
                "intensity": 0.7
            },
            "engine": "auto"
        },
        {
            "text": "That's wonderful!",
            "emotion_params": {
                "style_token": "#joy",
                "pitch_scale": 1.15,
                "rate_scale": 1.05,
                "energy_scale": 1.2,
                "emotion_label": "joyful_excited",
                "intensity": 0.8
            },
            "engine": "auto"
        },
        {
            "text": "I understand how you feel.",
            "emotion_params": {
                "style_token": "#sad",
                "pitch_scale": 0.85,
                "rate_scale": 0.85,
                "energy_scale": 0.6,
                "emotion_label": "empathetic_sad",
                "intensity": 0.7
            },
            "engine": "csm"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['text']}")
        print(f"  Emotion: {case['emotion_params']['emotion_label']}")
        print(f"  Engine: {case['engine']}")
        
        try:
            audio = engine.generate(
                text=case['text'],
                emotion_params=case['emotion_params'],
                engine=case['engine']
            )
            
            duration = audio.shape[0] / 24000  # Default sample rate
            print(f"  Duration: {duration:.2f}s")
            print(f"  Samples: {audio.shape[0]}")
            
            # Save test audio
            output_path = f"output/test_voice_{i}.wav"
            Path("output").mkdir(exist_ok=True)
            engine.save_audio(audio, output_path)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print()


def test_full_pipeline():
    """Test the complete pipeline."""
    print("\n🧪 Testing Full Pipeline")
    print("=" * 40)
    
    from pipeline import OviyaPipeline
    
    pipeline = OviyaPipeline()
    
    test_messages = [
        "I'm feeling really stressed about work today",
        "I got promoted! I'm so excited!",
        "I'm feeling sad and alone",
        "I'm frustrated with everything",
        "Just checking in, how are you?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            response = pipeline.process(message)
            total_time = time.time() - start_time
            
            print(f"✅ Response: {response['text']}")
            print(f"   Emotion: {response['emotion']}")
            print(f"   User Emotion: {response['user_emotion']}")
            print(f"   Total Time: {total_time*1000:.0f}ms")
            
            # Save audio
            if response['audio'] is not None:
                output_path = f"output/pipeline_test_{i}.wav"
                pipeline.save_response_audio(response['audio'], output_path)
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all tests."""
    print("🚀 Oviya Production Integration Test")
    print("=" * 50)
    
    # Test individual components
    test_emotion_detector()
    test_brain()
    test_emotion_controller()
    test_voice()
    
    # Test full pipeline
    test_full_pipeline()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("📁 Check output/ directory for generated audio files")
    print("=" * 50)


if __name__ == "__main__":
    main()
