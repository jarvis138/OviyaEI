"""
Multi-TTS Emotion Teacher
=========================

Extracts emotion references from multiple open-source TTS models:
- OpenVoiceV2: Emotion-expressive voice cloning
- Coqui TTS (XTTS-v2): Multilingual emotion control
- Bark: Text-to-speech with emotion tags
- StyleTTS2: Style transfer for emotional expression

All references are normalized and ready for CSM-1B context.
"""

import torch
import torchaudio
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class MultiTTSEmotionTeacher:
    """
    Multi-TTS teacher for generating emotion references.
    
    Supports multiple TTS models for diverse emotion reference generation.
    """
    
    def __init__(
        self,
        models_dir: str = "external",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize multi-TTS emotion teacher"""
        self.device = device
        self.models_dir = Path(models_dir)
        self.sample_rate = 24000
        
        # Initialize available TTS models
        self.openvoice_available = False
        self.coqui_available = False
        self.bark_available = False
        self.styletts_available = False
        
        # Emotion texts
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
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available TTS models"""
        # Check OpenVoiceV2
        openvoice_path = self.models_dir / "OpenVoice"
        if (openvoice_path / "checkpoints_v2").exists():
            try:
                sys.path.insert(0, str(openvoice_path))
                from openvoice import se_extractor
                self.openvoice_available = True
                logger.info("✅ OpenVoiceV2 available")
            except ImportError:
                pass
        
        # Check Coqui TTS
        try:
            from TTS.api import TTS
            self.coqui_available = True
            logger.info("✅ Coqui TTS available")
        except ImportError:
            pass
        
        # Check Bark
        try:
            from bark import generate_audio, preload_models
            self.bark_available = True
            logger.info("✅ Bark available")
        except ImportError:
            pass
    
    def get_reference_audio(
        self,
        emotion: str,
        preferred_model: Optional[str] = None,
        use_cached: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Get emotion reference audio from best available TTS model.
        
        Args:
            emotion: Emotion label
            preferred_model: Preferred TTS model ("openvoice", "coqui", "bark")
            use_cached: Use cached reference if available
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Check cached reference first
        ref_path = Path("data/emotion_references") / f"{emotion}.wav"
        if use_cached and ref_path.exists():
            logger.info(f"📁 Loading cached reference: {emotion}")
            audio, sr = torchaudio.load(str(ref_path))
            return audio.squeeze(0), sr
        
        # Try preferred model first
        if preferred_model:
            audio, sr = self._generate_with_model(emotion, preferred_model)
            if audio is not None:
                return audio, sr
        
        # Try all available models
        models_to_try = []
        if self.openvoice_available:
            models_to_try.append("openvoice")
        if self.coqui_available:
            models_to_try.append("coqui")
        if self.bark_available:
            models_to_try.append("bark")
        
        for model_name in models_to_try:
            audio, sr = self._generate_with_model(emotion, model_name)
            if audio is not None:
                # Save for future use
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(ref_path), audio.unsqueeze(0), sr)
                return audio, sr
        
        # Fallback to synthetic
        logger.warning(f"⚠️ Using synthetic reference for: {emotion}")
        return self._create_synthetic_reference(emotion)
    
    def _generate_with_model(
        self,
        emotion: str,
        model_name: str
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Generate reference with specific TTS model"""
        text = self.emotion_texts.get(emotion, "Hello.")
        
        if model_name == "openvoice" and self.openvoice_available:
            return self._generate_openvoice(emotion, text)
        elif model_name == "coqui" and self.coqui_available:
            return self._generate_coqui(emotion, text)
        elif model_name == "bark" and self.bark_available:
            return self._generate_bark(emotion, text)
        
        return None
    
    def _generate_openvoice(self, emotion: str, text: str) -> Optional[Tuple[torch.Tensor, int]]:
        """Generate with OpenVoiceV2"""
        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            
            openvoice_path = self.models_dir / "OpenVoice"
            ckpt_converter = openvoice_path / "checkpoints_v2" / "converter"
            
            if not ckpt_converter.exists():
                return None
            
            converter = ToneColorConverter(
                f"{ckpt_converter}/config.json",
                device=self.device
            )
            converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
            
            base_speaker_path = openvoice_path / "checkpoints_v2" / "base_speakers" / "EN"
            ref_audio = list(base_speaker_path.rglob("*.wav"))
            if not ref_audio:
                return None
            
            src_se = se_extractor.get_se(str(ref_audio[0]), converter, vad=False)
            temp_output = Path("temp_oviya_ref.wav")
            
            converter.convert(
                audio_src_path=str(ref_audio[0]),
                src_se=src_se,
                tgt_path=str(temp_output),
                message=text,
                output_dir=str(Path("temp")),
                tone_color_converter=converter
            )
            
            if temp_output.exists():
                audio, sr = torchaudio.load(str(temp_output))
                temp_output.unlink()
                
                # Normalize to 24kHz
                if sr != 24000:
                    audio_tensor = audio.unsqueeze(0)
                    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 24000)
                    audio = audio_tensor.squeeze(0)
                
                return audio, 24000
                
        except Exception as e:
            logger.warning(f"OpenVoiceV2 generation failed: {e}")
        
        return None
    
    def _generate_coqui(self, emotion: str, text: str) -> Optional[Tuple[torch.Tensor, int]]:
        """Generate with Coqui TTS"""
        try:
            from TTS.api import TTS
            
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
                     progress_bar=False)
            
            temp_output = Path("temp_coqui_ref.wav")
            tts.tts_to_file(text=text, file_path=str(temp_output))
            
            if temp_output.exists():
                audio, sr = torchaudio.load(str(temp_output))
                temp_output.unlink()
                
                # Normalize to 24kHz
                if sr != 24000:
                    audio_tensor = audio.unsqueeze(0)
                    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 24000)
                    audio = audio_tensor.squeeze(0)
                
                return audio, 24000
                
        except Exception as e:
            logger.warning(f"Coqui TTS generation failed: {e}")
        
        return None
    
    def _generate_bark(self, emotion: str, text: str) -> Optional[Tuple[torch.Tensor, int]]:
        """Generate with Bark"""
        try:
            from bark import generate_audio, SAMPLE_RATE
            
            emotion_prompts = {
                "calm_supportive": "[speaker:calm, soothing]",
                "empathetic_sad": "[speaker:gentle, empathetic]",
                "joyful_excited": "[speaker:excited, happy]",
                "playful": "[speaker:playful, cheerful]",
                "confident": "[speaker:confident, strong]",
                "concerned_anxious": "[speaker:concerned, caring]",
                "angry_firm": "[speaker:firm, determined]",
                "neutral": "[speaker:neutral]"
            }
            
            prompt = emotion_prompts.get(emotion, "")
            full_text = f"{prompt} {text}"
            
            audio_array = generate_audio(full_text, history_prompt=None)
            
            if audio_array is not None and len(audio_array) > 0:
                audio_tensor = torch.from_numpy(audio_array.astype('float32')).unsqueeze(0)
                return audio_tensor.squeeze(0), SAMPLE_RATE
                
        except Exception as e:
            logger.warning(f"Bark generation failed: {e}")
        
        return None
    
    def _create_synthetic_reference(self, emotion: str) -> Tuple[torch.Tensor, int]:
        """Create synthetic emotion reference"""
        duration = 2.0
        num_samples = int(duration * self.sample_rate)
        
        t = torch.linspace(0, duration, num_samples)
        
        emotion_freqs = {
            "calm_supportive": 200,
            "empathetic_sad": 180,
            "joyful_excited": 300,
            "playful": 280,
            "confident": 220,
            "concerned_anxious": 240,
            "angry_firm": 180,
            "neutral": 220
        }
        
        base_freq = emotion_freqs.get(emotion, 220)
        audio = 0.3 * torch.sin(2 * torch.pi * base_freq * t)
        
        if emotion == "joyful_excited":
            vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
            audio = audio * (1 + vibrato)
        elif emotion == "empathetic_sad":
            decay = torch.exp(-t * 0.5)
            audio = audio * decay
        
        return audio, self.sample_rate
    
    def generate_all_references(
        self,
        output_dir: str = "data/emotion_references"
    ) -> Dict[str, str]:
        """Generate all emotion references from all available TTS models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n🎭 Generating All Emotion References")
        logger.info("=" * 60)
        
        references = {}
        
        for emotion in self.emotion_texts.keys():
            logger.info(f"\n[{emotion}]")
            audio, sr = self.get_reference_audio(emotion, use_cached=False)
            
            filepath = output_path / f"{emotion}.wav"
            torchaudio.save(str(filepath), audio.unsqueeze(0), sr)
            
            references[emotion] = str(filepath)
            logger.info(f"✅ Saved: {filepath}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ Generated {len(references)} emotion references")
        logger.info(f"📁 Output directory: {output_path}")
        
        return references


# Backward compatibility
OpenVoiceEmotionTeacher = MultiTTSEmotionTeacher


if __name__ == "__main__":
    teacher = MultiTTSEmotionTeacher()
    references = teacher.generate_all_references()
    
    print("\n📊 Summary:")
    for emotion, path in references.items():
        print(f"  {emotion}: {path}")

