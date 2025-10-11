"""
Generate Expanded Emotion References on Vast.ai
Uses CSM to generate speech-based emotion references for all blended emotions
"""

import sys
import os
sys.path.insert(0, '/workspace/csm/csm')
os.environ["NO_TORCH_COMPILE"] = "1"

import torch
import torchaudio
from pathlib import Path
from generator import load_csm_1b

# All emotions with their text templates
EMOTION_TEXTS = {
    # Base 8 emotions (regenerate with better text)
    "calm_supportive": "Take a deep breath. Everything will be okay.",
    "empathetic_sad": "I'm so sorry you're going through this.",
    "joyful_excited": "Wow! That is wonderful! I am so excited!",  # Updated
    "playful": "Hey there! This is going to be fun!",
    "confident": "You've got this. I believe in you.",
    "concerned_anxious": "Are you okay? I'm here if you need me.",
    "angry_firm": "That's not acceptable. This needs to stop.",
    "neutral": "Hello. How can I help you today.",
    
    # Tier 1: Core everyday emotions (new)
    "comforting": "It's okay. I'm here for you, everything will be alright.",
    "encouraging": "You can do this! I believe in you completely.",
    "thoughtful": "Let me think about that for a moment. That's interesting.",
    "affectionate": "I care about you so much. You mean a lot to me.",
    "reassuring": "Don't worry. You're safe, and everything is going to be fine.",
    
    # Tier 2: Contextual emotions (new)
    "melancholy": "Sometimes things are hard, and that's okay to feel.",
    "wistful": "I remember those days. It feels like a distant dream now.",
    "tired": "It's been a long day. I'm feeling a bit worn out.",
    "curious": "Really? Tell me more! I want to know everything about it.",
    "dreamy": "Imagine a peaceful place where everything is calm and beautiful.",
    "relieved": "Oh thank goodness! I'm so glad that worked out.",
    "proud": "Look at what you've accomplished! That's truly impressive.",
    
    # Tier 3: Expressive emotions (new)
    "sarcastic": "Oh yeah, that's exactly what I meant. Totally.",
    "mischievous": "I have an idea, and you're going to love this.",
    "tender": "You're so precious to me. I want you to know that.",
    "amused": "Ha! That's actually pretty funny when you think about it.",
    "sympathetic": "I understand how you feel. That must be really difficult.",
    "reflective": "Looking back, I can see how all of this connects.",
    "grateful": "Thank you so much. I really appreciate everything you've done.",
    "apologetic": "I'm truly sorry. I didn't mean for things to turn out this way."
}

def generate_emotion_references():
    """Generate all emotion references using CSM"""
    
    print("=" * 70)
    print("🎨 EXPANDED EMOTION REFERENCE GENERATOR")
    print("=" * 70)
    print(f"\nGenerating {len(EMOTION_TEXTS)} emotion references...")
    print("This will take ~15-20 minutes\n")
    
    # Load CSM model
    print("🔄 Loading CSM model...")
    generator = load_csm_1b(device="cuda")
    print("✅ CSM loaded successfully!\n")
    
    # Create output directory
    output_dir = Path("/workspace/emotion_references")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate each emotion
    stats = {"success": 0, "failed": 0}
    
    for i, (emotion, text) in enumerate(EMOTION_TEXTS.items(), 1):
        try:
            print(f"[{i}/{len(EMOTION_TEXTS)}] Generating '{emotion}'...")
            print(f"   Text: \"{text[:60]}...\"")
            
            # Generate audio
            audio = generator.generate(
                text=text,
                speaker=0,
                context=[],
                max_audio_length_ms=15000  # 15 seconds max
            )
            
            # Save as WAV
            output_path = output_dir / f"{emotion}.wav"
            torchaudio.save(
                str(output_path),
                audio.unsqueeze(0).cpu(),
                generator.sample_rate
            )
            
            duration = audio.shape[-1] / generator.sample_rate
            stats["success"] += 1
            print(f"   ✅ Saved {emotion}.wav ({duration:.2f}s)\n")
            
        except Exception as e:
            stats["failed"] += 1
            print(f"   ❌ Failed: {e}\n")
    
    # Summary
    print("=" * 70)
    print("📊 GENERATION COMPLETE")
    print("=" * 70)
    print(f"✅ Successfully generated: {stats['success']}/{len(EMOTION_TEXTS)}")
    if stats["failed"] > 0:
        print(f"❌ Failed: {stats['failed']}")
    print(f"\n💾 All files saved to: {output_dir}")
    print("\n🔄 Next steps:")
    print("   1. Extract embeddings using OpenVoice encoder (if needed)")
    print("   2. Update CSM server to use new references")
    print("   3. Test with Oviya pipeline")
    print("=" * 70)


if __name__ == "__main__":
    generate_emotion_references()

