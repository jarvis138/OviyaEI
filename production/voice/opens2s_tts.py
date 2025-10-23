"""
OpenVoiceV2 Voice Integration

This module integrates OpenVoiceV2 for emotional voice generation.
OpenVoiceV2 is a production-ready voice cloning and synthesis model.

Setup:
    1. Clone OpenVoice: git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice
    2. Install: cd external/OpenVoice && pip install -e .
    3. Download model: huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ./models/OpenVoiceV2

Repository: https://github.com/myshell-ai/OpenVoice
Paper: https://arxiv.org/abs/2312.01479
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path
import json


class OpenVoiceV2TTS:
    """
    OpenVoiceV2 TTS wrapper for Oviya.
    
    Integrates OpenVoiceV2's voice cloning and synthesis with our emotion controller.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize OpenVoiceV2 TTS."""
        self.device = device
        self.model_path = model_path or "external/OpenVoice/models/OpenVoiceV2"
        self.model = None
        self.sample_rate = 24000  # OpenVoiceV2 default
        
        # Try to load OpenVoiceV2
        self._load_openvoice()
    
    def _load_openvoice(self):
        """Load OpenVoiceV2 model."""
        try:
            # Check if OpenVoice is available
            openvoice_path = Path("external/OpenVoice")
            if openvoice_path.exists():
                sys.path.insert(0, str(openvoice_path))
                
                # Import OpenVoiceV2 (actual implementation depends on OpenVoice structure)
                # This is a placeholder - actual imports will depend on OpenVoice's API
                print("⚠️ OpenVoiceV2 integration pending - see setup instructions")
                print("   1. Clone: git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice")
                print("   2. Install: cd external/OpenVoice && pip install -e .")
                print("   3. Download: huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ./models/OpenVoiceV2")
                
                # For now, we'll use a mock that shows the interface
                self.model = MockOpenVoiceV2(self.device)
                print(f"✅ OpenVoiceV2 TTS initialized (mock mode) on {self.device}")
            else:
                print("⚠️ OpenVoice not found. Using mock TTS.")
                print("   Clone it: git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice")
                self.model = MockOpenVoiceV2(self.device)
        
        except Exception as e:
            print(f"❌ Error loading OpenVoiceV2: {e}")
            print("   Using mock TTS for development")
            self.model = MockOpenVoiceV2(self.device)
    
    def generate(
        self,
        text: str,
        emotion_params: Dict,
        speaker_id: str = "oviya_v1",
        reference_audio: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate speech with emotion using OpenVoiceV2.
        
        Args:
            text: Text to speak
            emotion_params: Dict from emotion controller with:
                - style_token: Emotion style token
                - pitch_scale: Pitch scaling factor
                - rate_scale: Speed scaling factor
                - energy_scale: Energy/volume scaling
            speaker_id: Speaker identity (for LoRA adapter)
            reference_audio: Path to reference audio for voice cloning
        
        Returns:
            Audio tensor (1D, sample_rate Hz)
        """
        if self.model is None:
            raise RuntimeError("OpenVoiceV2 model not loaded")
        
        # Generate audio with emotion parameters
        audio = self.model.synthesize(
            text=text,
            emotion_params=emotion_params,
            speaker_id=speaker_id,
            reference_audio=reference_audio
        )
        
        return audio
    
    def clone_voice(
        self,
        reference_audio_path: str,
        speaker_name: str = "oviya"
    ) -> bool:
        """
        Clone a voice from reference audio.
        
        Args:
            reference_audio_path: Path to reference audio file
            speaker_name: Name for the cloned voice
        
        Returns:
            True if successful
        """
        try:
            # Load reference audio
            audio, sr = torchaudio.load(reference_audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            # Extract voice characteristics (placeholder)
            voice_characteristics = self._extract_voice_characteristics(audio)
            
            # Save voice profile
            voice_profile = {
                "speaker_name": speaker_name,
                "sample_rate": self.sample_rate,
                "characteristics": voice_characteristics,
                "reference_path": reference_audio_path
            }
            
            profile_path = f"voice/adapters/{speaker_name}_profile.json"
            Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(profile_path, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            print(f"✅ Voice cloned: {speaker_name}")
            return True
            
        except Exception as e:
            print(f"❌ Voice cloning failed: {e}")
            return False
    
    def _extract_voice_characteristics(self, audio: torch.Tensor) -> Dict:
        """Extract voice characteristics from audio (placeholder)."""
        # This would use OpenVoiceV2's voice extraction
        return {
            "pitch_mean": float(torch.mean(audio)),
            "energy_mean": float(torch.mean(torch.abs(audio))),
            "duration": audio.shape[-1] / self.sample_rate
        }
    
    def save_audio(self, audio: torch.Tensor, output_path: str):
        """Save audio to file."""
        torchaudio.save(
            output_path,
            audio.unsqueeze(0).cpu(),
            self.sample_rate
        )
        print(f"💾 Saved audio: {output_path}")


class MockOpenVoiceV2:
    """
    Mock OpenVoiceV2 for development before OpenVoiceV2 is installed.
    
    This generates simple sine wave audio with emotion-based variations
    so you can test the pipeline structure.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.sample_rate = 24000
        print("🔧 Using Mock TTS (install OpenVoiceV2 for real voice)")
    
    def synthesize(
        self,
        text: str,
        emotion_params: Dict,
        speaker_id: str = "oviya_v1",
        reference_audio: Optional[str] = None
    ) -> torch.Tensor:
        """Generate mock audio with emotion variations."""
        # Calculate duration based on text length (rough estimate)
        duration = len(text.split()) * 0.5  # ~0.5s per word
        num_samples = int(duration * self.sample_rate)
        
        # Get emotion parameters
        pitch_scale = emotion_params.get("pitch_scale", 1.0)
        energy_scale = emotion_params.get("energy_scale", 1.0)
        rate_scale = emotion_params.get("rate_scale", 1.0)
        
        # Adjust duration by rate
        duration = duration / rate_scale
        num_samples = int(duration * self.sample_rate)
        
        # Generate complex tone (mock audio)
        t = torch.linspace(0, duration, num_samples)
        
        # Base frequency scaled by pitch
        base_freq = 200 * pitch_scale
        
        # Create more complex waveform
        audio = (
            0.4 * energy_scale * torch.sin(2 * torch.pi * base_freq * t) +
            0.2 * energy_scale * torch.sin(2 * torch.pi * base_freq * 2 * t) +
            0.1 * energy_scale * torch.sin(2 * torch.pi * base_freq * 3 * t)
        )
        
        # Add emotion-specific modulation
        emotion = emotion_params.get("emotion_label", "neutral")
        if emotion == "joyful_excited":
            # Add vibrato
            vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
            audio = audio * (1 + vibrato)
        elif emotion == "empathetic_sad":
            # Lower pitch variation
            audio = audio * torch.sin(2 * torch.pi * 0.5 * t) * 0.5 + audio * 0.5
        
        # Add some variation
        audio = audio + 0.02 * torch.randn(num_samples)
        
        print(f"🔊 Generated mock audio: {duration:.2f}s, pitch: {pitch_scale:.2f}, energy: {energy_scale:.2f}")
        
        return audio.to(self.device)


# Example usage
if __name__ == "__main__":
    # Initialize TTS
    tts = OpenVoiceV2TTS()
    
    # Test with emotion parameters
    emotion_params = {
        "style_token": "#calm",
        "pitch_scale": 0.9,
        "rate_scale": 0.9,
        "energy_scale": 0.8,
        "emotion_label": "calm_supportive"
    }
    
    text = "I'm here with you, take a deep breath."
    
    print(f"\n🧪 Testing OpenVoiceV2 TTS\n")
    print(f"Text: {text}")
    print(f"Emotion params: {emotion_params}\n")
    
    audio = tts.generate(text, emotion_params)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Duration: {audio.shape[0] / tts.sample_rate:.2f}s")
    
    # Save to file
    tts.save_audio(audio, "test_output.wav")

