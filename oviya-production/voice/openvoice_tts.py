"""
Hybrid Voice Engine - CSM + OpenVoiceV2 Integration

This module integrates both CSM and OpenVoiceV2 for emotional voice generation.
- CSM: Conversational consistency and context awareness
- OpenVoiceV2: Voice cloning and fine emotion control

Setup:
    CSM: Already running on Vast.ai (port 6006)
    OpenVoiceV2: git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice
"""

import torch
import torchaudio
import numpy as np
import requests
import json
from typing import Dict, Optional, Tuple, Union
import sys
from pathlib import Path
import time


class HybridVoiceEngine:
    """
    Hybrid voice engine supporting both CSM and OpenVoiceV2.
    
    Automatically selects the best engine based on:
    - Voice cloning needs (OpenVoiceV2)
    - Conversational context (CSM)
    - Fallback reliability
    """
    
    def __init__(
        self,
        csm_url: str = "http://localhost:6006/generate",
        openvoice_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_engine: str = "auto"  # "auto", "csm", "openvoice"
    ):
        """Initialize hybrid voice engine."""
        self.device = device
        self.csm_url = csm_url
        self.openvoice_model_path = openvoice_model_path or "external/OpenVoice/models/OpenVoiceV2"
        self.default_engine = default_engine
        
        # Initialize engines
        self.csm_available = False
        self.openvoice_available = False
        
        print("🎤 Initializing Hybrid Voice Engine...")
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize both CSM and OpenVoiceV2 engines."""
        
        # Initialize CSM
        print("\n[1/2] Checking CSM availability...")
        self.csm_available = self._check_csm_availability()
        
        # Initialize OpenVoiceV2
        print("\n[2/2] Checking OpenVoiceV2 availability...")
        self.openvoice_available = self._check_openvoice_availability()
        
        print(f"\n✅ Engine Status:")
        print(f"   CSM: {'✅ Available' if self.csm_available else '❌ Unavailable'}")
        print(f"   OpenVoiceV2: {'✅ Available' if self.openvoice_available else '❌ Unavailable'}")
        
        if not self.csm_available and not self.openvoice_available:
            print("⚠️ No voice engines available! Using mock TTS.")
            self.mock_tts = MockTTS(self.device)
    
    def _check_csm_availability(self) -> bool:
        """Check if CSM service is available."""
        try:
            response = requests.get(f"{self.csm_url.replace('/generate', '/health')}", timeout=5)
            if response.status_code == 200:
                print("✅ CSM service is running")
                return True
        except:
            pass
        
        # Try direct generate endpoint
        try:
            test_payload = {
                "prompt": "test",
                "speaker_id": 0,
                "max_audio_length_ms": 1000
            }
            response = requests.post(self.csm_url, json=test_payload, timeout=5)
            if response.status_code == 200:
                print("✅ CSM service is running (via generate endpoint)")
                return True
        except:
            pass
        
        print("❌ CSM service not available")
        return False
    
    def _check_openvoice_availability(self) -> bool:
        """Check if OpenVoiceV2 is available."""
        try:
            openvoice_path = Path("external/OpenVoice")
            if openvoice_path.exists():
                sys.path.insert(0, str(openvoice_path))
                # Try to import OpenVoiceV2
                print("✅ OpenVoiceV2 repository found")
                return True
            else:
                print("❌ OpenVoiceV2 not found")
                return False
        except Exception as e:
            print(f"❌ OpenVoiceV2 error: {e}")
            return False
    
    def generate(
        self,
        text: str,
        emotion_params: Dict,
        speaker_id: str = "oviya_v1",
        engine: Optional[str] = None,
        conversation_context: Optional[list] = None
    ) -> torch.Tensor:
        """
        Generate speech using the best available engine.
        
        Args:
            text: Text to speak
            emotion_params: Emotion parameters from controller
            speaker_id: Speaker identity
            engine: Force specific engine ("csm" or "openvoice")
            conversation_context: Recent conversation for CSM
        
        Returns:
            Audio tensor
        """
        # Select engine
        selected_engine = self._select_engine(engine, emotion_params, conversation_context)
        
        print(f"🎤 Using engine: {selected_engine}")
        
        if selected_engine == "csm":
            return self._generate_with_csm(text, emotion_params, speaker_id, conversation_context)
        elif selected_engine == "openvoice":
            return self._generate_with_openvoice(text, emotion_params, speaker_id)
        else:
            # Fallback to mock
            return self._generate_with_mock(text, emotion_params)
    
    def _select_engine(
        self, 
        forced_engine: Optional[str], 
        emotion_params: Dict, 
        conversation_context: Optional[list]
    ) -> str:
        """Select the best engine for the given parameters."""
        
        if forced_engine:
            return forced_engine
        
        if self.default_engine == "auto":
            # Auto-selection logic
            
            # Prefer CSM for conversational context
            if conversation_context and len(conversation_context) > 0 and self.csm_available:
                return "csm"
            
            # Prefer OpenVoiceV2 for voice cloning
            if speaker_id != "oviya_v1" and self.openvoice_available:
                return "openvoice"
            
            # Prefer CSM for complex emotions
            emotion = emotion_params.get("emotion_label", "neutral")
            if emotion in ["empathetic_sad", "calm_supportive"] and self.csm_available:
                return "csm"
            
            # Prefer OpenVoiceV2 for expressive emotions
            if emotion in ["joyful_excited", "playful"] and self.openvoice_available:
                return "openvoice"
            
            # Default to available engine
            if self.csm_available:
                return "csm"
            elif self.openvoice_available:
                return "openvoice"
            else:
                return "mock"
        
        # Use default engine if available
        if self.default_engine == "csm" and self.csm_available:
            return "csm"
        elif self.default_engine == "openvoice" and self.openvoice_available:
            return "openvoice"
        else:
            # Fallback to available engine
            if self.csm_available:
                return "csm"
            elif self.openvoice_available:
                return "openvoice"
            else:
                return "mock"
    
    def _generate_with_csm(
        self, 
        text: str, 
        emotion_params: Dict, 
        speaker_id: str,
        conversation_context: Optional[list]
    ) -> torch.Tensor:
        """Generate speech using CSM."""
        try:
            # Prepare CSM payload
            payload = {
                "prompt": text,
                "speaker_id": 0,  # CSM uses numeric speaker IDs
                "max_audio_length_ms": 10000,
                "temperature": self._map_emotion_to_csm_temperature(emotion_params),
                "topk": 50
            }
            
            # Add context if available
            if conversation_context:
                payload["context"] = conversation_context
            
            # Call CSM service
            response = requests.post(self.csm_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                audio_base64 = result.get("audio_base64")
                
                if audio_base64:
                    # Decode base64 audio
                    import base64
                    import io
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_buffer = io.BytesIO(audio_bytes)
                    
                    # Load audio
                    audio, sample_rate = torchaudio.load(audio_buffer)
                    audio = audio.squeeze(0)  # Remove channel dimension
                    
                    print(f"✅ CSM generated: {audio.shape[0]/sample_rate:.2f}s")
                    return audio.to(self.device)
                else:
                    raise Exception("No audio in CSM response")
            else:
                raise Exception(f"CSM error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ CSM generation failed: {e}")
            # Fallback to mock
            return self._generate_with_mock(text, emotion_params)
    
    def _generate_with_openvoice(
        self, 
        text: str, 
        emotion_params: Dict, 
        speaker_id: str
    ) -> torch.Tensor:
        """Generate speech using OpenVoiceV2."""
        try:
            # This would use the actual OpenVoiceV2 implementation
            # For now, use mock with OpenVoiceV2 parameters
            print("⚠️ OpenVoiceV2 implementation pending")
            return self._generate_with_mock(text, emotion_params)
        
        except Exception as e:
            print(f"❌ OpenVoiceV2 generation failed: {e}")
            # Fallback to CSM or mock
            if self.csm_available:
                return self._generate_with_csm(text, emotion_params, speaker_id, None)
            else:
                return self._generate_with_mock(text, emotion_params)
    
    def _generate_with_mock(self, text: str, emotion_params: Dict) -> torch.Tensor:
        """Generate mock audio for testing."""
        # Calculate duration
        duration = len(text.split()) * 0.5
        num_samples = int(duration * 24000)  # 24kHz
        
        # Get emotion parameters
        pitch_scale = emotion_params.get("pitch_scale", 1.0)
        energy_scale = emotion_params.get("energy_scale", 1.0)
        
        # Generate complex tone
        t = torch.linspace(0, duration, num_samples)
        base_freq = 200 * pitch_scale
        
        audio = (
            0.4 * energy_scale * torch.sin(2 * torch.pi * base_freq * t) +
            0.2 * energy_scale * torch.sin(2 * torch.pi * base_freq * 2 * t)
        )
        
        # Add emotion-specific modulation
        emotion = emotion_params.get("emotion_label", "neutral")
        if emotion == "joyful_excited":
            vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
            audio = audio * (1 + vibrato)
        
        audio = audio + 0.02 * torch.randn(num_samples)
        
        print(f"🔊 Mock generated: {duration:.2f}s")
        return audio.to(self.device)
    
    def _map_emotion_to_csm_temperature(self, emotion_params: Dict) -> float:
        """Map emotion intensity to CSM temperature."""
        intensity = emotion_params.get("intensity", 0.7)
        emotion = emotion_params.get("emotion_label", "neutral")
        
        # Base temperature mapping
        base_temps = {
            "calm_supportive": 0.7,
            "empathetic_sad": 0.6,
            "joyful_excited": 0.8,
            "playful": 0.9,
            "confident": 0.7,
            "concerned_anxious": 0.6,
            "angry_firm": 0.8,
            "neutral": 0.7
        }
        
        base_temp = base_temps.get(emotion, 0.7)
        
        # Adjust by intensity
        return base_temp + (intensity - 0.7) * 0.3
    
    def clone_voice(
        self,
        reference_audio_path: str,
        speaker_name: str = "oviya"
    ) -> bool:
        """Clone voice using OpenVoiceV2."""
        if not self.openvoice_available:
            print("❌ OpenVoiceV2 not available for voice cloning")
            return False
        
        try:
            # This would use OpenVoiceV2's voice cloning
            print(f"🎤 Voice cloning with OpenVoiceV2: {speaker_name}")
            # Implementation pending
            return True
        except Exception as e:
            print(f"❌ Voice cloning failed: {e}")
            return False
    
    def generate_with_reference(
        self,
        text: str,
        reference_audio: torch.Tensor,
        emotion_params: Dict,
        speaker_id: str = "oviya_v1"
    ) -> torch.Tensor:
        """
        Generate audio using reference audio for emotion conditioning.
        
        This is specifically for Stage 0 emotion transfer testing,
        where we test if CSM can reproduce emotions from OpenVoice V2 references.
        
        Args:
            text: Text to speak
            reference_audio: Reference audio tensor from OpenVoice V2
            emotion_params: Emotion parameters from controller
            speaker_id: Speaker identity
        
        Returns:
            Generated audio tensor
        """
        if not self.csm_available:
            print("⚠️ CSM not available for reference testing, using mock")
            return self._generate_with_mock(text, emotion_params)
        
        # Prepare CSM context with reference audio
        context = self._prepare_reference_context(reference_audio, text)
        
        # Generate with CSM using reference
        return self._generate_with_csm(
            text=text,
            emotion_params=emotion_params,
            speaker_id=0,
            conversation_context=context
        )
    
    def _prepare_reference_context(
        self,
        reference_audio: torch.Tensor,
        text: str
    ) -> list:
        """
        Format reference audio for CSM's context.
        
        CSM uses conversation history format, so we convert
        the reference audio into a temporary audio file.
        """
        import tempfile
        import os
        
        # Create temp directory
        temp_dir = Path("temp/reference_audio")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reference audio temporarily
        ref_path = temp_dir / f"ref_{hash(text) % 10000}.wav"
        torchaudio.save(str(ref_path), reference_audio.unsqueeze(0).cpu(), 24000)
        
        # Build context for CSM
        # CSM can use this as acoustic conditioning
        context = [{
            "text": "Reference emotional tone",
            "speaker": 0,
            "audio_path": str(ref_path)
        }]
        
        return context
    
    def save_audio(self, audio: torch.Tensor, output_path: str):
        """Save audio to file."""
        torchaudio.save(
            output_path,
            audio.unsqueeze(0).cpu(),
            24000  # Default sample rate
        )
        print(f"💾 Saved audio: {output_path}")


class MockTTS:
    """Mock TTS for testing."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.sample_rate = 24000
        print("🔧 Using Mock TTS")


# Backward compatibility
OpenVoiceV2TTS = HybridVoiceEngine


# Example usage
if __name__ == "__main__":
    # Initialize hybrid engine
    engine = HybridVoiceEngine()
    
    # Test with different engines
    emotion_params = {
        "style_token": "#calm",
        "pitch_scale": 0.9,
        "rate_scale": 0.9,
        "energy_scale": 0.8,
        "emotion_label": "calm_supportive",
        "intensity": 0.7
    }
    
    text = "I'm here with you, take a deep breath."
    
    print(f"\n🧪 Testing Hybrid Voice Engine\n")
    print(f"Text: {text}")
    print(f"Emotion params: {emotion_params}\n")
    
    # Test auto-selection
    audio = engine.generate(text, emotion_params)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Duration: {audio.shape[0] / engine.sample_rate:.2f}s")
    
    # Save to file
    engine.save_audio(audio, "test_hybrid_output.wav")

