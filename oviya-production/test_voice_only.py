#!/usr/bin/env python3
"""
Test voice generation only - isolate the speaker_id error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice.openvoice_tts import HybridVoiceEngine
from emotion_controller.controller import EmotionController

def test_voice_generation():
    """Test voice generation with mock data."""
    print("🧪 Testing Voice Generation Only")
    print("=" * 40)
    
    # Initialize voice engine
    print("1. Initializing Hybrid Voice Engine...")
    voice_engine = HybridVoiceEngine(
        csm_url="http://45.78.17.160:6006/generate",
        default_engine="auto"
    )
    
    # Initialize emotion controller
    print("2. Initializing Emotion Controller...")
    emotion_controller = EmotionController("config/emotions.json")
    
    # Test data
    test_text = "Hello, this is a test."
    emotion_params = emotion_controller.map_emotion("neutral", intensity=0.7)
    
    print(f"3. Testing voice generation...")
    print(f"   Text: {test_text}")
    print(f"   Emotion: {emotion_params['emotion_label']}")
    
    try:
        # Generate voice
        audio = voice_engine.generate(
            text=test_text,
            emotion_params=emotion_params,
            speaker_id="oviya_v1"
        )
        
        print(f"✅ Voice generation successful!")
        print(f"   Audio shape: {audio.shape}")
        print(f"   Duration: {audio.shape[0]/24000:.2f}s")
        
        # Save audio
        import torchaudio
        torchaudio.save("test_voice_output.wav", audio.unsqueeze(0), 24000)
        print(f"💾 Saved: test_voice_output.wav")
        
    except Exception as e:
        print(f"❌ Voice generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_voice_generation()

