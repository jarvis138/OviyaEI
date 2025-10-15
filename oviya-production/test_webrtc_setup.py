#!/usr/bin/env python3
"""
Test script to verify WebRTC setup is working correctly
"""

import sys
import asyncio

def test_imports():
    """Test that all required packages are installed"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("   ✅ torch")
    except ImportError:
        print("   ❌ torch - MISSING")
        return False
    
    try:
        import aiortc
        print("   ✅ aiortc")
    except ImportError:
        print("   ❌ aiortc - MISSING (pip3 install aiortc)")
        return False
    
    try:
        import av
        print("   ✅ av")
    except ImportError:
        print("   ❌ av - MISSING (pip3 install av)")
        return False
    
    try:
        import numpy as np
        print("   ✅ numpy")
    except ImportError:
        print("   ❌ numpy - MISSING")
        return False
    
    try:
        import requests
        print("   ✅ requests")
    except ImportError:
        print("   ❌ requests - MISSING")
        return False
    
    try:
        import torchaudio
        print("   ✅ torchaudio")
    except ImportError:
        print("   ❌ torchaudio - MISSING")
        return False
    
    try:
        from fastapi import FastAPI
        print("   ✅ fastapi")
    except ImportError:
        print("   ❌ fastapi - MISSING")
        return False
    
    return True


def test_silero_vad():
    """Test Silero VAD download and initialization"""
    print("\n🎤 Testing Silero VAD...")
    
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        print("   ✅ Silero VAD loaded successfully")
        
        # Test with dummy audio (512 samples = 32ms at 16kHz)
        audio = torch.randn(512)
        speech_prob = model(audio, 16000).item()
        print(f"   ✅ VAD test successful (speech_prob: {speech_prob:.3f})")
        
        return True
    except Exception as e:
        print(f"   ❌ Silero VAD failed: {e}")
        return False


def test_services():
    """Test connection to Vast.ai services"""
    print("\n🔗 Testing Vast.ai services...")
    
    import requests
    
    services = {
        "WhisperX": "https://msgid-enquiries-williams-lands.trycloudflare.com/health",
        "Ollama": "https://prime-show-visit-lock.trycloudflare.com/api/tags",
        "CSM TTS": "https://astronomy-initiative-paso-cream.trycloudflare.com/health"
    }
    
    all_ok = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {name}")
            else:
                print(f"   ⚠️  {name} - Status {response.status_code}")
                all_ok = False
        except Exception as e:
            print(f"   ❌ {name} - {str(e)[:50]}")
            all_ok = False
    
    return all_ok


def test_webrtc_classes():
    """Test WebRTC server classes can be imported"""
    print("\n🧪 Testing WebRTC server classes...")
    
    try:
        from voice_server_webrtc import (
            SileroVAD,
            RemoteWhisperXClient,
            RemoteCSMClient,
            AudioStreamTrack,
            OviyaVoiceConnection
        )
        print("   ✅ All classes imported successfully")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False


async def test_whisperx_client():
    """Test WhisperX client with dummy audio"""
    print("\n🎤 Testing WhisperX client...")
    
    try:
        from voice_server_webrtc import RemoteWhisperXClient
        import numpy as np
        
        client = RemoteWhisperXClient()
        
        # Create 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)
        
        result = await client.transcribe(audio)
        
        if result and 'text' in result:
            print(f"   ✅ WhisperX client working")
            print(f"      Result: {result.get('text', '(empty)')}")
            return True
        else:
            print(f"   ⚠️  WhisperX returned unexpected result: {result}")
            return False
    except Exception as e:
        print(f"   ❌ WhisperX client failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 OVIYA WEBRTC SETUP TEST")
    print("=" * 70)
    print()
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test Silero VAD
    results.append(("Silero VAD", test_silero_vad()))
    
    # Test services
    results.append(("Vast.ai Services", test_services()))
    
    # Test WebRTC classes
    results.append(("WebRTC Classes", test_webrtc_classes()))
    
    # Test WhisperX client (async)
    try:
        whisperx_result = asyncio.run(test_whisperx_client())
        results.append(("WhisperX Client", whisperx_result))
    except Exception as e:
        print(f"   ❌ WhisperX async test failed: {e}")
        results.append(("WhisperX Client", False))
    
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
        print("\n🎉 All tests passed! WebRTC setup is ready.")
        print("\nNext steps:")
        print("   1. Run: ./start_webrtc.sh")
        print("   2. Open: http://localhost:8000/")
        print("   3. Click microphone and start talking!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("   • Missing packages: pip3 install -r requirements_webrtc.txt")
        print("   • Services down: Check Vast.ai tunnels")
        print("   • Import errors: Ensure Python 3.9+")
        return 1


if __name__ == "__main__":
    sys.exit(main())

