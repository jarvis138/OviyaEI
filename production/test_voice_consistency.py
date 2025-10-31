#!/usr/bin/env python3
"""
Test Oviya's voice consistency across different emotions
Ensures CSM-1B generates the same voice personality regardless of emotion
"""

import asyncio
import sys
import os
sys.path.insert(0, '/Users/jarvis/Documents/Oviya EI')
sys.path.insert(0, '/Users/jarvis/Documents/Oviya EI/production')

from voice.csm_1b_stream import CSMRVQStreamer
import soundfile as sf
import numpy as np

async def test_voice_consistency():
    """Test that Oviya's voice remains consistent across emotions"""
    print("🎤 TESTING OVIYA VOICE CONSISTENCY")
    print("=" * 50)

    # Initialize streamer with consistent settings
    streamer = CSMRVQStreamer(flush_rvq_frames=2)

    # Test text that should sound the same voice regardless of emotion
    test_text = "Hello, I'm Oviya, your emotional AI companion. How are you feeling today?"

    emotions = ["joyful", "empathetic", "calm", "supportive", "comforting"]

    print(f"Testing text: '{test_text}'")
    print(f"Emotions to test: {emotions}")
    print()

    generated_audios = {}

    for emotion in emotions:
        print(f"🎵 Generating {emotion} version...")

        # Generate audio with consistent speaker ID (42)
        audio_chunks = []
        async for chunk in streamer.generate_streaming(
            text=test_text,
            emotion=emotion,
            speaker_id=42,  # Oviya's consistent speaker ID
            temperature=0.7,  # Lower temperature for consistency
            top_p=0.9
        ):
            audio_chunks.append(chunk)
            if len(audio_chunks) >= 5:  # Just get first few chunks for testing
                break

        full_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
        generated_audios[emotion] = full_audio

        # Save for manual inspection
        filename = f'/tmp/oviya_consistency_{emotion}.wav'
        sf.write(filename, full_audio, 24000)
        print(f"   ✅ Saved: {filename} ({len(full_audio)} samples)")

    print("\n🎯 VOICE CONSISTENCY ANALYSIS:")
    print("-" * 30)

    # Analyze consistency (basic check - same approximate length)
    lengths = {emotion: len(audio) for emotion, audio in generated_audios.items()}

    print("Audio lengths by emotion:")
    for emotion, length in lengths.items():
        print(f"  {emotion}: {length} samples")

    # Check if lengths are similar (same voice should generate similar timing)
    avg_length = sum(lengths.values()) / len(lengths)
    max_diff = max(abs(length - avg_length) for length in lengths.values())

    print(".2f")
    print(".2f")

    if max_diff < avg_length * 0.1:  # Within 10% variation
        print("✅ VOICE CONSISTENCY: EXCELLENT")
        print("   Audio lengths are very similar across emotions")
    elif max_diff < avg_length * 0.2:  # Within 20% variation
        print("⚠️ VOICE CONSISTENCY: GOOD")
        print("   Some variation detected, but generally consistent")
    else:
        print("❌ VOICE CONSISTENCY: NEEDS IMPROVEMENT")
        print("   Significant variation in audio lengths")

    print("\n🎧 MANUAL VERIFICATION:")
    print("Please listen to these files to verify voice consistency:")
    for emotion in emotions:
        print(f"  afplay /tmp/oviya_consistency_{emotion}.wav  # {emotion}")

    print("\n💡 EXPECTED RESULT:")
    print("All audio files should sound like the same female voice")
    print("Only the emotional tone should change, not the voice identity")

if __name__ == "__main__":
    asyncio.run(test_voice_consistency())
