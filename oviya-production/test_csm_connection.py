#!/usr/bin/env python3
"""
CSM Connection Test

Tests connection to CSM service running on Vast.ai
"""

import requests
import json
import base64
import io
import torchaudio
from pathlib import Path


def test_csm_connection(csm_url: str = "http://45.78.17.160:6006/generate"):
    """Test CSM service connection."""
    print("🧪 Testing CSM Connection")
    print("=" * 40)
    
    # Test 1: Health check
    print("Test 1: Health check")
    try:
        health_url = csm_url.replace('/generate', '/health')
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print("✅ CSM health check passed")
        else:
            print(f"⚠️ Health check returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: Basic generation
    print("\nTest 2: Basic generation")
    try:
        payload = {
            "prompt": "Hello, this is a test.",
            "speaker_id": 0,
            "max_audio_length_ms": 5000,
            "temperature": 0.7,
            "topk": 50
        }
        
        response = requests.post(csm_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CSM generation successful")
            print(f"   Response keys: {list(result.keys())}")
            
            # Check audio
            if "audio_base64" in result:
                audio_base64 = result["audio_base64"]
                print(f"   Audio length: {len(audio_base64)} characters")
                
                # Decode and save audio
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_buffer = io.BytesIO(audio_bytes)
                    
                    audio, sample_rate = torchaudio.load(audio_buffer)
                    duration = audio.shape[-1] / sample_rate
                    
                    print(f"   Audio duration: {duration:.2f}s")
                    print(f"   Sample rate: {sample_rate}Hz")
                    print(f"   Channels: {audio.shape[0]}")
                    
                    # Save test audio
                    Path("output").mkdir(exist_ok=True)
                    torchaudio.save("output/csm_test.wav", audio, sample_rate)
                    print("   💾 Saved: output/csm_test.wav")
                    
                except Exception as e:
                    print(f"   ❌ Audio decode failed: {e}")
            else:
                print("   ⚠️ No audio in response")
        else:
            print(f"❌ CSM generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ CSM generation error: {e}")
    
    # Test 3: Different emotions
    print("\nTest 3: Different emotions")
    emotions = [
        {"prompt": "I'm feeling calm and peaceful.", "temperature": 0.6},
        {"prompt": "I'm so excited and happy!", "temperature": 0.8},
        {"prompt": "I'm feeling sad and lonely.", "temperature": 0.5}
    ]
    
    for i, emotion_test in enumerate(emotions, 1):
        try:
            payload = {
                "prompt": emotion_test["prompt"],
                "speaker_id": 0,
                "max_audio_length_ms": 3000,
                "temperature": emotion_test["temperature"],
                "topk": 50
            }
            
            response = requests.post(csm_url, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if "audio_base64" in result:
                    print(f"✅ Emotion test {i}: {emotion_test['prompt'][:30]}...")
                    
                    # Save emotion-specific audio
                    audio_bytes = base64.b64decode(result["audio_base64"])
                    audio_buffer = io.BytesIO(audio_bytes)
                    audio, sample_rate = torchaudio.load(audio_buffer)
                    
                    output_path = f"output/csm_emotion_{i}.wav"
                    torchaudio.save(output_path, audio, sample_rate)
                    print(f"   💾 Saved: {output_path}")
                else:
                    print(f"❌ Emotion test {i}: No audio")
            else:
                print(f"❌ Emotion test {i}: Failed ({response.status_code})")
        
        except Exception as e:
            print(f"❌ Emotion test {i}: Error - {e}")


def main():
    """Run CSM connection tests."""
    print("🚀 CSM Connection Test")
    print("=" * 50)
    
    # Test Vast.ai connection
    test_csm_connection("http://45.78.17.160:6006/generate")
    
    print("\n" + "=" * 50)
    print("✅ CSM connection test completed!")
    print("📁 Check output/ directory for generated audio files")
    print("=" * 50)


if __name__ == "__main__":
    main()

