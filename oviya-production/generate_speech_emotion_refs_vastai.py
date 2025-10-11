#!/usr/bin/env python3
"""
Generate SPEECH-based Emotion References using CSM

Instead of synthetic tones, we'll use CSM to generate actual speech
with the emotional texts. These will serve as better emotion references.
"""

import sys
sys.path.insert(0, '/workspace/csm/csm')

import torch
import torchaudio
from generator import load_csm_1b
from pathlib import Path

# Emotion texts - CSM will generate these with natural prosody
EMOTION_TEXTS = {
    "calm_supportive": "Take a deep breath. Everything will be okay.",
    "empathetic_sad": "I'm so sorry you're going through this.",
    "joyful_excited": "That's amazing! I'm so happy for you!",
    "playful": "Hey there! This is going to be fun!",
    "confident": "You've got this. I believe in you.",
    "concerned_anxious": "Are you okay? I'm here if you need me.",
    "angry_firm": "That's not acceptable. This needs to stop.",
    "neutral": "Hello. How can I help you today?"
}

def main():
    print("\n" + "="*60)
    print("🎙️ GENERATING SPEECH-BASED EMOTION REFERENCES")
    print("="*60)
    
    # Load CSM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📦 Loading CSM on {device}...")
    generator = load_csm_1b(device=device)
    print("✅ CSM loaded!")
    
    # Create output directory
    output_dir = Path("/workspace/emotion_references_speech")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n🎤 Generating {len(EMOTION_TEXTS)} speech-based references...")
    print("="*60)
    
    for emotion, text in EMOTION_TEXTS.items():
        print(f"\n[{emotion}]")
        print(f"   Text: \"{text}\"")
        
        try:
            # Generate speech using CSM
            audio = generator.generate(
                text=text,
                speaker=0,
                context=[],  # No context for base references
                max_audio_length_ms=10000
            )
            
            # Save reference
            output_path = output_dir / f"{emotion}.wav"
            torchaudio.save(
                str(output_path),
                audio.unsqueeze(0).cpu(),
                generator.sample_rate
            )
            
            duration = audio.shape[-1] / generator.sample_rate
            print(f"   ✅ Generated: {duration:.2f}s")
            print(f"   💾 Saved: {output_path}")
        
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n" + "="*60)
    print(f"✅ Generated speech-based emotion references")
    print(f"📁 Location: {output_dir}")
    
    # Backup old synthetic references
    old_dir = Path("/workspace/emotion_references")
    if old_dir.exists():
        backup_dir = Path("/workspace/emotion_references_synthetic_backup")
        print(f"\n📦 Backing up synthetic references to: {backup_dir}")
        import shutil
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(old_dir, backup_dir)
    
    print("\n🔄 To use these new references, run:")
    print("   rm -rf /workspace/emotion_references")
    print("   mv /workspace/emotion_references_speech /workspace/emotion_references")
    print("   # Then restart CSM server")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import os
    os.environ["NO_TORCH_COMPILE"] = "1"
    main()


