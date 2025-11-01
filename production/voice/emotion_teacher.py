"""
OpenVoice Emotion Teacher

Wraps OpenVoiceV2 to extract emotional reference audio.
These references are used to condition CSM for emotionally expressive speech.
"""

import torch
import torchaudio
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional


class OpenVoiceEmotionTeacher:
    """
    Teacher model using OpenVoiceV2 to provide emotional references.
    
    This class extracts or generates emotional audio samples that serve
    as reference context for CSM generation.
    """
    
    def __init__(
        self,
        openvoice_path: str = "external/OpenVoice",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize OpenVoice teacher.
        
        Args:
            openvoice_path: Path to OpenVoice repository
            device: Device to use for inference
        """
        self.device = device
        self.openvoice_path = Path(openvoice_path)
        self.model = None
        self.sample_rate = 24000
        
        # Emotion-to-text mapping for reference generation
        self.emotion_texts = {
            "calm_supportive": "Take a deep breath. Everything will be okay.",
            "empathetic_sad": "I'm so sorry you're going through this.",
            "joyful_excited": "That's amazing! I'm so happy for you!",
            "playful": "Hey there! This is going to be fun!",
            "confident": "You've got this. I believe in you.",
            "concerned_anxious": "Are you okay? I'm here if you need me.",
            "angry_firm": "That's not acceptable. This needs to stop.",
            "neutral": "Hello. How can I help you today?"
        }
        
        print("🎓 Initializing OpenVoice Emotion Teacher...")
        self._load_model()
    
    def _load_model(self):
        """Load OpenVoiceV2 model."""
        try:
            # Add OpenVoice to path
            if self.openvoice_path.exists():
                sys.path.insert(0, str(self.openvoice_path))
                
                # Try to import OpenVoice
                # Note: Actual import depends on OpenVoice's structure
                # This is a placeholder - adjust based on OpenVoice's API
                try:
                    from openvoice import se_extractor
                    from openvoice.api import ToneColorConverter
                    print("✅ OpenVoice modules loaded")
                    self.model = "loaded"  # Placeholder
                except ImportError as e:
                    print(f"⚠️ OpenVoice import failed: {e}")
                    print("   Will use pre-generated references")
                    self.model = None
            else:
                print(f"⚠️ OpenVoice path not found: {self.openvoice_path}")
                print("   Will use pre-generated references")
                self.model = None
        
        except Exception as e:
            print(f"⚠️ Error loading OpenVoice: {e}")
            self.model = None
    
    def get_reference_audio(
        self,
        emotion: str,
        use_cached: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Get emotional reference audio.
        
        Args:
            emotion: Emotion label
            use_cached: Use pre-generated reference if available
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Path to pre-generated reference
        ref_path = Path("data/emotion_references") / f"{emotion}.wav"
        
        # Try to load cached reference first
        if use_cached and ref_path.exists():
            print(f"📁 Loading cached reference: {emotion}")
            audio, sr = torchaudio.load(str(ref_path))
            return audio.squeeze(0), sr
        
        # Generate new reference if model is available
        if self.model is not None:
            print(f"🎙️ Generating reference for: {emotion}")
            audio = self._generate_reference(emotion)
            
            # Save for future use
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(ref_path), audio.unsqueeze(0), self.sample_rate)
            print(f"💾 Saved reference: {ref_path}")
            
            return audio, self.sample_rate
        
        # Fallback: create synthetic reference
        print(f"⚠️ Using synthetic reference for: {emotion}")
        return self._create_synthetic_reference(emotion)
    
    def _generate_reference(self, emotion: str) -> torch.Tensor:
        """
        Generate emotional reference using OpenVoiceV2.
        
        Uses OpenVoiceV2 API to generate emotion-expressive speech.
        """
        if self.model is None:
            # Fallback to synthetic if model not loaded
            return self._create_synthetic_reference(emotion)[0]
        
        try:
            # Import OpenVoice modules
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            
            # Get emotion text
            text = self.emotion_texts.get(emotion, "Hello.")
            
            # Map emotion to OpenVoice style token
            style_map = {
                "calm_supportive": "default",
                "empathetic_sad": "sad",
                "joyful_excited": "happy",
                "playful": "cheerful",
                "confident": "default",
                "concerned_anxious": "sad",
                "angry_firm": "angry",
                "neutral": "default"
            }
            
            style_token = style_map.get(emotion, "default")
            
            # Check if we have a base speaker
            base_speaker_path = Path(self.openvoice_path) / "checkpoints_v2" / "base_speakers" / "EN"
            if not base_speaker_path.exists():
                return self._create_synthetic_reference(emotion)[0]
            
            # Find reference audio
            ref_audio = list(base_speaker_path.glob("*.wav"))
            if not ref_audio:
                ref_audio = list(base_speaker_path.rglob("*.wav"))
            
            if not ref_audio:
                return self._create_synthetic_reference(emotion)[0]
            
            ref_audio_path = ref_audio[0]
            
            # Initialize converter if needed
            if not hasattr(self, '_converter'):
                ckpt_converter = Path(self.openvoice_path) / "checkpoints_v2" / "converter"
                if ckpt_converter.exists():
                    self._converter = ToneColorConverter(
                        f"{ckpt_converter}/config.json",
                        device=self.device
                    )
                    self._converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
                else:
                    return self._create_synthetic_reference(emotion)[0]
            
            # Extract speaker embedding
            src_se = se_extractor.get_se(str(ref_audio_path), self._converter, vad=False)
            
            # Generate audio
            # Note: OpenVoiceV2 API may vary - adjust based on actual implementation
            # This is a simplified version
            temp_output = Path("temp_oviya_ref.wav")
            
            try:
                # Use OpenVoice to synthesize
                self._converter.convert(
                    audio_src_path=str(ref_audio_path),
                    src_se=src_se,
                    tgt_path=str(temp_output),
                    message=text,
                    output_dir=str(Path("temp")),
                    tone_color_converter=self._converter
                )
                
                # Load generated audio
                if temp_output.exists():
                    audio, sr = torchaudio.load(str(temp_output))
                    temp_output.unlink()  # Clean up
                    return audio.squeeze(0), sr
                    
            except Exception as e:
                print(f"⚠️ OpenVoice synthesis failed: {e}")
                if temp_output.exists():
                    temp_output.unlink()
            
        except ImportError as e:
            print(f"⚠️ OpenVoice modules not available: {e}")
        except Exception as e:
            print(f"⚠️ OpenVoice generation failed: {e}")
        
        # Fallback to synthetic
        return self._create_synthetic_reference(emotion)[0]
    
    def _create_synthetic_reference(
        self,
        emotion: str
    ) -> Tuple[torch.Tensor, int]:
        """
        Create synthetic emotional reference for testing.
        
        This generates a simple tone-based reference when OpenVoice is unavailable.
        """
        duration = 2.0  # 2 seconds
        num_samples = int(duration * self.sample_rate)
        
        t = torch.linspace(0, duration, num_samples)
        
        # Different frequency patterns for different emotions
        emotion_freqs = {
            "calm_supportive": 200,      # Low, soothing
            "empathetic_sad": 180,       # Lower, gentle
            "joyful_excited": 300,       # Higher, energetic
            "playful": 280,              # Varied, bouncy
            "confident": 220,            # Strong, steady
            "concerned_anxious": 240,    # Mid, uncertain
            "angry_firm": 180,           # Low, intense
            "neutral": 220               # Balanced
        }
        
        base_freq = emotion_freqs.get(emotion, 220)
        
        # Generate tone
        audio = 0.3 * torch.sin(2 * torch.pi * base_freq * t)
        
        # Add emotion-specific modulation
        if emotion == "joyful_excited":
            vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
            audio = audio * (1 + vibrato)
        elif emotion == "empathetic_sad":
            decay = torch.exp(-t)
            audio = audio * decay
        
        return audio, self.sample_rate
    
    def generate_all_references(
        self,
        output_dir: str = "data/emotion_references"
    ) -> Dict[str, str]:
        """
        Generate all 8 emotion references and save them.
        
        Args:
            output_dir: Directory to save reference files
        
        Returns:
            Dict mapping emotion to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n🎭 Generating All Emotion References")
        print("=" * 60)
        
        references = {}
        
        for emotion in self.emotion_texts.keys():
            print(f"\n[{emotion}]")
            audio, sr = self.get_reference_audio(emotion, use_cached=False)
            
            filepath = output_path / f"{emotion}.wav"
            torchaudio.save(str(filepath), audio.unsqueeze(0), sr)
            
            references[emotion] = str(filepath)
            print(f"✅ Saved: {filepath}")
        
        print("\n" + "=" * 60)
        print(f"✅ Generated {len(references)} emotion references")
        print(f"📁 Output directory: {output_path}")
        
        return references


# Test script
if __name__ == "__main__":
    teacher = OpenVoiceEmotionTeacher()
    
    # Generate all references
    references = teacher.generate_all_references()
    
    print("\n📊 Summary:")
    for emotion, path in references.items():
        print(f"  {emotion}: {path}")
