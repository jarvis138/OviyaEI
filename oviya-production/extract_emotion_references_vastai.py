"""
Emotion Reference Extractor for Vast.ai

Generates 8 emotion reference audio files using OpenVoiceV2.
These references are used as CSM context for emotionally expressive speech.

Run this on your Vast.ai server after installing OpenVoiceV2.
"""

import torch
import torchaudio
import sys
from pathlib import Path
import json


# Add OpenVoice to path
OPENVOICE_PATH = Path("/workspace/OpenVoice")
if OPENVOICE_PATH.exists():
    sys.path.insert(0, str(OPENVOICE_PATH))


def load_emotion_config():
    """Load emotion mapping configuration."""
    config_path = Path("config/emotion_reference_mapping.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return get_default_config()


def get_default_config():
    """Default emotion configuration."""
    return {
        "emotions": {
            "calm_supportive": {
                "reference_text": "Take a deep breath. Everything will be okay.",
                "openvoice_style": "calm"
            },
            "empathetic_sad": {
                "reference_text": "I'm so sorry you're going through this.",
                "openvoice_style": "sad"
            },
            "joyful_excited": {
                "reference_text": "That's amazing! I'm so happy for you!",
                "openvoice_style": "happy"
            },
            "playful": {
                "reference_text": "Hey there! This is going to be fun!",
                "openvoice_style": "cheerful"
            },
            "confident": {
                "reference_text": "You've got this. I believe in you.",
                "openvoice_style": "confident"
            },
            "concerned_anxious": {
                "reference_text": "Are you okay? I'm here if you need me.",
                "openvoice_style": "worried"
            },
            "angry_firm": {
                "reference_text": "That's not acceptable. This needs to stop.",
                "openvoice_style": "angry"
            },
            "neutral": {
                "reference_text": "Hello. How can I help you today?",
                "openvoice_style": "neutral"
            }
        }
    }


def generate_synthetic_reference(emotion: str, text: str, sample_rate: int = 24000):
    """
    Generate synthetic emotional reference for testing.
    
    This is a fallback when OpenVoice is not available.
    """
    duration = 2.0  # 2 seconds
    num_samples = int(duration * sample_rate)
    
    t = torch.linspace(0, duration, num_samples)
    
    # Different frequency patterns for different emotions
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
    
    # Generate tone
    audio = 0.3 * torch.sin(2 * torch.pi * base_freq * t)
    
    # Add emotion-specific modulation
    if emotion == "joyful_excited":
        vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
        audio = audio * (1 + vibrato)
    elif emotion == "empathetic_sad":
        decay = torch.exp(-t * 0.5)
        audio = audio * decay
    
    return audio, sample_rate


def generate_openvoice_reference(emotion: str, text: str, openvoice_style: str):
    """
    Generate reference using OpenVoiceV2.
    
    This is a placeholder - implement based on OpenVoice's actual API.
    """
    try:
        # TODO: Implement actual OpenVoice V2 generation
        # Example structure (adjust based on OpenVoice API):
        # 
        # from openvoice import se_extractor
        # from openvoice.api import ToneColorConverter
        # 
        # model = ToneColorConverter(...)
        # audio = model.convert(
        #     text=text,
        #     emotion=openvoice_style,
        #     speaker_id=0
        # )
        
        print(f"   ⚠️ OpenVoice generation not yet implemented")
        print(f"   Using synthetic reference instead")
        return None
    
    except Exception as e:
        print(f"   ❌ OpenVoice generation failed: {e}")
        return None


def main():
    """Generate all emotion references."""
    print("\n" + "="*60)
    print("🎭 EMOTION REFERENCE EXTRACTION")
    print("="*60)
    
    # Load configuration
    print("\n📋 Loading emotion configuration...")
    config = load_emotion_config()
    emotions = config["emotions"]
    
    # Create output directory
    output_dir = Path("/workspace/emotion_references")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Check if OpenVoice is available
    openvoice_available = OPENVOICE_PATH.exists()
    if openvoice_available:
        print("✅ OpenVoice found at /workspace/OpenVoice")
    else:
        print("⚠️ OpenVoice not found - will use synthetic references")
    
    print(f"\n🎙️ Generating {len(emotions)} emotion references...")
    print("="*60)
    
    generated_refs = {}
    
    for emotion, config in emotions.items():
        print(f"\n[{emotion}]")
        print(f"   Text: \"{config['reference_text']}\"")
        print(f"   Style: {config['openvoice_style']}")
        
        # Try OpenVoice first if available
        audio = None
        if openvoice_available:
            audio = generate_openvoice_reference(
                emotion,
                config['reference_text'],
                config['openvoice_style']
            )
        
        # Fallback to synthetic
        if audio is None:
            audio, sr = generate_synthetic_reference(
                emotion,
                config['reference_text']
            )
        else:
            sr = 24000  # OpenVoice sample rate
        
        # Save reference
        output_path = output_dir / f"{emotion}.wav"
        torchaudio.save(str(output_path), audio.unsqueeze(0), sr)
        
        duration = audio.shape[0] / sr
        print(f"   ✅ Generated: {duration:.2f}s")
        print(f"   💾 Saved: {output_path}")
        
        generated_refs[emotion] = str(output_path)
    
    print("\n" + "="*60)
    print(f"✅ Successfully generated {len(generated_refs)} references")
    print(f"📁 Location: {output_dir}")
    print("="*60)
    
    print("\n📊 Summary:")
    for emotion, path in generated_refs.items():
        print(f"   {emotion}: {Path(path).name}")
    
    print("\n🎯 Next steps:")
    print("   1. Test references with CSM")
    print("   2. Run Stage 0 evaluation")
    print("   3. Fine-tune based on results")


if __name__ == "__main__":
    main()


