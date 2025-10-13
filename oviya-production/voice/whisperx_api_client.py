"""
WhisperX API Client for Oviya
Connects to WhisperX server running on RTX 5880 Ada via Cloudflare tunnel
"""

import requests
import base64
import numpy as np
from typing import Dict, Optional, List
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.service_urls import WHISPERX_URL, WHISPERX_TRANSCRIBE, WHISPERX_HEALTH


class WhisperXAPIClient:
    """
    Client for WhisperX API running on RTX 5880 Ada
    Provides real-time transcription with word-level timestamps
    """
    
    def __init__(self, whisperx_url: str = WHISPERX_URL):
        self.whisperx_url = whisperx_url
        self.transcribe_endpoint = f"{whisperx_url}/transcribe"
        self.health_endpoint = f"{whisperx_url}/health"
        
        print(f"🎤 WhisperX API Client initialized")
        print(f"   URL: {whisperx_url}")
        
    def check_health(self) -> Dict:
        """Check if WhisperX API is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {'status': 'error', 'message': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def transcribe_audio(
        self, 
        audio_array: np.ndarray,
        batch_size: int = 8,
        language: str = "en"
    ) -> Optional[Dict]:
        """
        Transcribe audio with word-level timestamps
        
        Args:
            audio_array: Audio data as numpy array (float32, 16kHz)
            batch_size: Batch size for processing (default 8 for RTX 5880)
            language: Language code (default "en")
            
        Returns:
            Dictionary with transcription results or None on error
        """
        try:
            start_time = time.time()
            
            # Convert audio to base64
            audio_base64 = base64.b64encode(audio_array.tobytes()).decode('utf-8')
            
            # Send to WhisperX API
            response = requests.post(
                self.transcribe_endpoint,
                json={
                    'audio': audio_base64,
                    'batch_size': batch_size,
                    'language': language
                },
                timeout=60  # 60 second timeout for transcription
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add client-side timing
                result['client_processing_time'] = time.time() - start_time
                
                return result
            else:
                print(f"❌ WhisperX API error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ WhisperX transcription error: {e}")
            return None
    
    def transcribe_file(self, audio_file_path: str) -> Optional[Dict]:
        """
        Transcribe audio file
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary with transcription results or None on error
        """
        try:
            import soundfile as sf
            
            # Load audio file
            audio, sample_rate = sf.read(audio_file_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Convert to float32
            audio = audio.astype(np.float32)
            
            # Transcribe
            return self.transcribe_audio(audio)
            
        except Exception as e:
            print(f"❌ File transcription error: {e}")
            return None


def test_whisperx_api():
    """Test WhisperX API functionality"""
    print("=" * 60)
    print("Testing WhisperX API Client")
    print("=" * 60)
    
    # Initialize client
    client = WhisperXAPIClient()
    
    # Test health
    print("\n🧪 Testing health endpoint...")
    health = client.check_health()
    print(f"   Status: {health.get('status')}")
    print(f"   Device: {health.get('device')}")
    print(f"   Model: {health.get('model')}")
    print(f"   GPU Memory: {health.get('gpu_memory_gb', 0):.1f}GB")
    
    # Test transcription with dummy audio
    print("\n🎤 Testing transcription...")
    test_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
    
    result = client.transcribe_audio(test_audio, batch_size=8)
    
    if result:
        print(f"   ✅ Transcription successful!")
        print(f"   Status: {result.get('status')}")
        print(f"   Duration: {result.get('duration', 0):.2f}s")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   Client time: {result.get('client_processing_time', 0):.2f}s")
        print(f"   Text: {result.get('text', 'N/A')}")
        print(f"   Word timestamps: {len(result.get('word_timestamps', []))}")
    else:
        print("   ❌ Transcription failed")
    
    print("\n" + "=" * 60)
    print("✅ WhisperX API test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_whisperx_api()

