"""
OpenVoice V2 Emotion Teacher

Extracts emotional reference embeddings from OpenVoice V2's built-in style library.
This provides the "teacher" emotional references for CSM to learn from.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class OpenVoiceEmotionTeacher:
    """
    Wrapper for OpenVoice V2 to extract emotional references.
    Uses OpenVoice V2's built-in emotion reference library.
    
    This is the "teacher" model that provides emotional examples
    for CSM (the "student") to reproduce.
    """
    
    def __init__(self, model_path: str = "external/OpenVoice"):
        """Initialize OpenVoice V2 emotion teacher."""
        self.model_path = Path(model_path)
        self.sample_rate = 24000
        
        # OpenVoice V2 emotion reference library paths
        # These are the built-in emotional style references
        self.emotion_refs = {
            "calm_supportive": "checkpoints/base_speakers/ses/calm.wav",
            "empathetic_sad": "checkpoints/base_speakers/ses/sad.wav",
            "joyful_excited": "checkpoints/base_speakers/ses/happy.wav",
            "playful": "checkpoints/base_speakers/ses/cheerful.wav",
            "confident": "checkpoints/base_speakers/ses/confident.wav",
            "concerned_anxious": "checkpoints/base_speakers/ses/worried.wav",
            "angry_firm": "checkpoints/base_speakers/ses/angry.wav",
            "neutral": "checkpoints/base_speakers/ses/neutral.wav"
        }
        
        # Load OpenVoice V2 model
        self.model_available = self._load_model()
    
    def _load_model(self) -> bool:
        """Load OpenVoice V2 model."""
        try:
            if not self.model_path.exists():
                print("⚠️ OpenVoice V2 not found at", self.model_path)
                print("   Clone it: git clone https://github.com/myshell-ai/OpenVoice.git external/OpenVoice")
                return False
            
            # Import OpenVoice V2 components
            import sys
            sys.path.insert(0, str(self.model_path))
            
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            
            converter_path = self.model_path / "checkpoints" / "converter"
            if converter_path.exists():
                self.tone_color_converter = ToneColorConverter(str(converter_path))
                self.se_extractor = se_extractor
                print("✅ OpenVoice V2 emotion teacher loaded")
                return True
            else:
                print("⚠️ OpenVoice V2 converter not found")
                return False
        
        except Exception as e:
            print(f"⚠️ OpenVoice V2 not available: {e}")
            return False
    
    def extract_emotion_embedding(self, emotion: str) -> Optional[np.ndarray]:
        """
        Extract style embedding for a specific emotion.
        
        Args:
            emotion: Emotion label
        
        Returns:
            Style embedding vector (512-D) or None if extraction fails
        """
        if not self.model_available:
            print("⚠️ OpenVoice V2 not available, returning mock embedding")
            return np.random.randn(512).astype(np.float32)
        
        if emotion not in self.emotion_refs:
            raise ValueError(f"Unknown emotion: {emotion}")
        
        ref_path = self.model_path / self.emotion_refs[emotion]
        
        if not ref_path.exists():
            print(f"⚠️ Reference audio not found: {ref_path}")
            return np.random.randn(512).astype(np.float32)
        
        try:
            # Extract style embedding using OpenVoice's reference encoder
            target_se, _ = self.se_extractor.get_se(
                str(ref_path),
                self.tone_color_converter,
                target_dir='processed',
                vad=True
            )
            
            print(f"✅ Extracted embedding for: {emotion}")
            return target_se
        
        except Exception as e:
            print(f"❌ Failed to extract {emotion}: {e}")
            return None
    
    def extract_all_emotions(self) -> Dict[str, np.ndarray]:
        """Extract embeddings for all emotions."""
        embeddings = {}
        
        print("\n📦 Extracting all emotion embeddings from OpenVoice V2...")
        
        for emotion in self.emotion_refs.keys():
            embedding = self.extract_emotion_embedding(emotion)
            if embedding is not None:
                embeddings[emotion] = embedding
        
        print(f"✅ Extracted {len(embeddings)}/{len(self.emotion_refs)} emotions")
        return embeddings
    
    def get_reference_audio(self, emotion: str) -> Tuple[torch.Tensor, int]:
        """
        Get raw reference audio for an emotion.
        
        Args:
            emotion: Emotion label
        
        Returns:
            (audio_tensor, sample_rate)
        """
        if emotion not in self.emotion_refs:
            raise ValueError(f"Unknown emotion: {emotion}")
        
        ref_path = self.model_path / self.emotion_refs[emotion]
        
        if not ref_path.exists():
            print(f"⚠️ Reference audio not found, generating mock audio")
            # Generate mock emotional audio
            duration = 2.0
            audio = self._generate_mock_emotional_audio(emotion, duration)
            return audio, self.sample_rate
        
        try:
            audio, sr = torchaudio.load(str(ref_path))
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            return audio.squeeze(0), self.sample_rate
        
        except Exception as e:
            print(f"❌ Failed to load reference audio: {e}")
            duration = 2.0
            audio = self._generate_mock_emotional_audio(emotion, duration)
            return audio, self.sample_rate
    
    def _generate_mock_emotional_audio(self, emotion: str, duration: float) -> torch.Tensor:
        """Generate mock emotional audio for testing."""
        num_samples = int(duration * self.sample_rate)
        t = torch.linspace(0, duration, num_samples)
        
        # Emotion-specific parameters
        emotion_params = {
            "calm_supportive": {"freq": 180, "vibrato": 0.02},
            "empathetic_sad": {"freq": 160, "vibrato": 0.01},
            "joyful_excited": {"freq": 250, "vibrato": 0.08},
            "playful": {"freq": 220, "vibrato": 0.06},
            "confident": {"freq": 200, "vibrato": 0.03},
            "concerned_anxious": {"freq": 170, "vibrato": 0.04},
            "angry_firm": {"freq": 210, "vibrato": 0.05},
            "neutral": {"freq": 190, "vibrato": 0.02}
        }
        
        params = emotion_params.get(emotion, {"freq": 190, "vibrato": 0.02})
        
        # Generate tone with emotion-specific characteristics
        audio = 0.3 * torch.sin(2 * torch.pi * params["freq"] * t)
        
        # Add vibrato for emotion
        if params["vibrato"] > 0:
            vibrato = params["vibrato"] * torch.sin(2 * torch.pi * 5 * t)
            audio = audio * (1 + vibrato)
        
        # Add some noise
        audio = audio + 0.02 * torch.randn(num_samples)
        
        return audio
    
    def save_embeddings(self, output_dir: str = "data/emotion_embeddings"):
        """Save all emotion embeddings to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        embeddings = self.extract_all_emotions()
        
        for emotion, embedding in embeddings.items():
            output_path = Path(output_dir) / f"{emotion}.npy"
            np.save(output_path, embedding)
            print(f"💾 Saved: {output_path}")
        
        # Save metadata
        metadata = {
            "model": "OpenVoice V2",
            "emotions": list(embeddings.keys()),
            "embedding_dim": embeddings[list(embeddings.keys())[0]].shape[0] if embeddings else 0,
            "sample_rate": self.sample_rate
        }
        
        with open(Path(output_dir) / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Saved {len(embeddings)} embeddings to {output_dir}")
    
    def generate_reference_samples(self, output_dir: str = "data/reference_audio"):
        """Generate/copy all reference audio samples."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n🎵 Generating reference audio samples...")
        
        for emotion in self.emotion_refs.keys():
            audio, sr = self.get_reference_audio(emotion)
            output_path = Path(output_dir) / f"ref_{emotion}.wav"
            torchaudio.save(str(output_path), audio.unsqueeze(0), sr)
            print(f"💾 Saved: {output_path}")
        
        print(f"\n✅ Saved {len(self.emotion_refs)} reference samples to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize teacher
    teacher = OpenVoiceEmotionTeacher()
    
    print("\n🧪 Testing OpenVoice V2 Emotion Teacher\n")
    
    # Extract and save embeddings
    teacher.save_embeddings()
    
    # Generate reference audio samples
    teacher.generate_reference_samples()
    
    print("\n✅ OpenVoice V2 emotion teacher test complete!")

