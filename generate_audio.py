#!/usr/bin/env python3
"""
Simple audio generation and download script for Oviya EI
"""

import os
import sys
import torch
import torchaudio
from pathlib import Path

# Add production directory to path
sys.path.insert(0, str(Path(__file__).parent / "production"))

try:
    from production.voice.openvoice_tts import HybridVoiceEngine
    print("✅ Successfully imported HybridVoiceEngine")
except ImportError as e:
    print(f"❌ Failed to import HybridVoiceEngine: {e}")
    print("Please ensure you're in the correct directory and dependencies are installed")
    sys.exit(1)


def generate_oviya_intro():
    """Generate an introduction audio file for Oviya"""

    # Initialize the voice engine
    print("🎤 Initializing Hybrid Voice Engine...")
    engine = HybridVoiceEngine()

    # Define the intro text and emotion parameters
    text = "Hello! I'm Oviya, your empathetic AI companion. I'm here to listen, support, and help you through whatever you're going through. How are you feeling today?"

    emotion_params = {
        "style_token": "#warm_welcoming",
        "pitch_scale": 0.95,  # Slightly lower pitch for warmth
        "rate_scale": 0.9,    # Slightly slower for friendliness
        "energy_scale": 0.8,  # Calm and approachable
        "emotion_label": "calm_supportive",
        "intensity": 0.7
    }

    print(f"📝 Generating audio for: '{text[:50]}...'")
    print(f"🎭 Emotion: {emotion_params['emotion_label']}")

    try:
        # Generate the audio
        audio = engine.generate(text, emotion_params)
        print("✅ Audio generated successfully!")
        print(f"   Duration: {audio.shape[0] / 24000:.2f} seconds")
        print(f"   Shape: {audio.shape}")

        # Save to file
        output_path = "oviya_intro.wav"
        engine.save_audio(audio, output_path)

        print(f"💾 Audio saved to: {output_path}")
        print("🎧 You can now play it with: afplay oviya_intro.wav")

        return output_path

    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_custom_audio(text, emotion="calm_supportive", filename="custom_audio.wav"):
    """Generate custom audio with specified text and emotion"""

    print(f"🎤 Generating custom audio: '{text}'")

    # Initialize engine
    engine = HybridVoiceEngine()

    # Set emotion parameters based on emotion type
    emotion_configs = {
        "calm_supportive": {
            "pitch_scale": 0.95,
            "rate_scale": 0.9,
            "energy_scale": 0.8,
            "intensity": 0.7
        },
        "joyful_excited": {
            "pitch_scale": 1.1,
            "rate_scale": 1.0,
            "energy_scale": 1.0,
            "intensity": 0.8
        },
        "empathetic_sad": {
            "pitch_scale": 0.9,
            "rate_scale": 0.8,
            "energy_scale": 0.7,
            "intensity": 0.6
        },
        "confident": {
            "pitch_scale": 1.0,
            "rate_scale": 0.95,
            "energy_scale": 0.9,
            "intensity": 0.8
        }
    }

    config = emotion_configs.get(emotion, emotion_configs["calm_supportive"])
    emotion_params = {
        "emotion_label": emotion,
        **config
    }

    try:
        # Generate and save
        audio = engine.generate(text, emotion_params)
        engine.save_audio(audio, filename)

        print(f"✅ Custom audio saved to: {filename}")
        print(f"   Duration: {audio.shape[0] / 24000:.2f} seconds")

        return filename

    except Exception as e:
        print(f"❌ Custom audio generation failed: {e}")
        return None


if __name__ == "__main__":
    print("🎵 Oviya EI Audio Generation Tool")
    print("=" * 50)

    if len(sys.argv) == 1:
        # Generate default intro
        print("Generating Oviya introduction audio...")
        generate_oviya_intro()

    elif len(sys.argv) == 2:
        # Generate custom text with default emotion
        text = sys.argv[1]
        generate_custom_audio(text)

    elif len(sys.argv) == 3:
        # Generate custom text with specified emotion
        text = sys.argv[1]
        emotion = sys.argv[2]
        filename = f"{emotion}_{len(text)}_audio.wav"
        generate_custom_audio(text, emotion, filename)

    else:
        print("Usage:")
        print("  python generate_audio.py")
        print("    -> Generate default Oviya introduction")
        print("  python generate_audio.py 'your text here'")
        print("    -> Generate audio with custom text (calm_supportive)")
        print("  python generate_audio.py 'your text here' joyful_excited")
        print("    -> Generate audio with custom text and emotion")
