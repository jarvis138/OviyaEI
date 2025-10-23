"""
Emotion Blender - Creates expanded emotion library through embedding interpolation
Blends existing emotion references to create nuanced emotional expressions
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class EmotionBlender:
    """Blend emotion embeddings to create new emotional expressions"""
    
    # Base 8 emotions (from OpenVoice V2 references)
    BASE_EMOTIONS = [
        "calm_supportive",
        "empathetic_sad", 
        "joyful_excited",
        "playful",
        "confident",
        "concerned_anxious",
        "angry_firm",
        "neutral"
    ]
    
    # Blending recipes: {new_emotion: {base_emotion: weight, ...}}
    BLEND_RECIPES = {
        # Tier 1: Core everyday emotions
        "comforting": {"calm_supportive": 0.6, "empathetic_sad": 0.4},
        "encouraging": {"joyful_excited": 0.5, "confident": 0.3, "calm_supportive": 0.2},
        "thoughtful": {"calm_supportive": 0.6, "neutral": 0.4},
        "affectionate": {"calm_supportive": 0.6, "joyful_excited": 0.4},
        "reassuring": {"calm_supportive": 0.5, "confident": 0.5},
        
        # Tier 2: Contextual emotions
        "melancholy": {"empathetic_sad": 0.8, "calm_supportive": 0.2},
        "wistful": {"empathetic_sad": 0.5, "calm_supportive": 0.5},
        "tired": {"neutral": 0.7, "empathetic_sad": 0.3},
        "curious": {"neutral": 0.5, "joyful_excited": 0.3, "playful": 0.2},
        "dreamy": {"calm_supportive": 0.7, "neutral": 0.3},
        "relieved": {"calm_supportive": 0.6, "joyful_excited": 0.4},
        "proud": {"confident": 0.6, "joyful_excited": 0.4},
        
        # Tier 3: Expressive emotions
        "sarcastic": {"neutral": 0.6, "playful": 0.4},
        "mischievous": {"playful": 0.7, "joyful_excited": 0.3},
        "tender": {"calm_supportive": 0.7, "empathetic_sad": 0.3},
        "amused": {"playful": 0.6, "joyful_excited": 0.4},
        "sympathetic": {"empathetic_sad": 0.6, "calm_supportive": 0.4},
        "reflective": {"thoughtful": 0.6, "calm_supportive": 0.4},  # will use thoughtful once generated
        "grateful": {"joyful_excited": 0.5, "calm_supportive": 0.5},
        "apologetic": {"empathetic_sad": 0.6, "concerned_anxious": 0.4}
    }
    
    # Text templates for generating speech-based references
    EMOTION_TEXTS = {
        "comforting": "It's okay. I'm here for you, everything will be alright.",
        "encouraging": "You can do this! I believe in you completely.",
        "thoughtful": "Let me think about that for a moment. That's interesting.",
        "affectionate": "I care about you so much. You mean a lot to me.",
        "reassuring": "Don't worry. You're safe, and everything is going to be fine.",
        "melancholy": "Sometimes things are hard, and that's okay to feel.",
        "wistful": "I remember those days. It feels like a distant dream now.",
        "tired": "It's been a long day. I'm feeling a bit worn out.",
        "curious": "Really? Tell me more! I want to know everything about it.",
        "dreamy": "Imagine a peaceful place where everything is calm and beautiful.",
        "relieved": "Oh thank goodness! I'm so glad that worked out.",
        "proud": "Look at what you've accomplished! That's truly impressive.",
        "sarcastic": "Oh yeah, that's exactly what I meant. Totally.",
        "mischievous": "I have an idea, and you're going to love this.",
        "tender": "You're so precious to me. I want you to know that.",
        "amused": "Ha! That's actually pretty funny when you think about it.",
        "sympathetic": "I understand how you feel. That must be really difficult.",
        "reflective": "Looking back, I can see how all of this connects.",
        "grateful": "Thank you so much. I really appreciate everything you've done.",
        "apologetic": "I'm truly sorry. I didn't mean for things to turn out this way."
    }
    
    def __init__(self, reference_dir: Path = Path("/workspace/emotion_references")):
        """Initialize emotion blender"""
        self.reference_dir = Path(reference_dir)
        self.base_embeddings = {}
        self.blended_embeddings = {}
        
    def load_base_embeddings(self) -> Dict[str, np.ndarray]:
        """Load base emotion embeddings from .npy files"""
        print("📂 Loading base emotion embeddings...")
        
        for emotion in self.BASE_EMOTIONS:
            npy_path = self.reference_dir / f"{emotion}.npy"
            if npy_path.exists():
                self.base_embeddings[emotion] = np.load(npy_path)
                print(f"  ✅ Loaded {emotion}")
            else:
                print(f"  ⚠️  Missing {emotion}.npy")
        
        return self.base_embeddings
    
    def blend_embeddings(self) -> Dict[str, np.ndarray]:
        """Create blended emotion embeddings from recipes"""
        print("\n🎨 Blending emotions...")
        
        if not self.base_embeddings:
            self.load_base_embeddings()
        
        for new_emotion, recipe in self.BLEND_RECIPES.items():
            # Skip if any base emotion is missing
            if not all(base in self.base_embeddings for base in recipe.keys()):
                print(f"  ⚠️  Skipping {new_emotion} (missing base emotions)")
                continue
            
            # Weighted sum of base embeddings
            blended = np.zeros_like(self.base_embeddings[self.BASE_EMOTIONS[0]])
            for base_emotion, weight in recipe.items():
                blended += weight * self.base_embeddings[base_emotion]
            
            # Normalize (optional - keeps magnitude consistent)
            # blended = blended / np.linalg.norm(blended)
            
            self.blended_embeddings[new_emotion] = blended
            
            recipe_str = " + ".join([f"{w:.1f}×{e}" for e, w in recipe.items()])
            print(f"  ✅ Created {new_emotion} ({recipe_str})")
        
        return self.blended_embeddings
    
    def save_blended_embeddings(self, output_dir: Optional[Path] = None):
        """Save blended embeddings to .npy files"""
        if output_dir is None:
            output_dir = self.reference_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n💾 Saving blended embeddings to {output_dir}...")
        
        for emotion, embedding in self.blended_embeddings.items():
            output_path = output_dir / f"{emotion}.npy"
            np.save(output_path, embedding)
            print(f"  ✅ Saved {emotion}.npy")
    
    def generate_emotion_library_config(self, output_path: Path):
        """Generate JSON config for the expanded emotion library"""
        config = {
            "version": "1.0",
            "total_emotions": len(self.BASE_EMOTIONS) + len(self.BLEND_RECIPES),
            "tiers": {
                "tier1_core": [
                    "calm_supportive", "empathetic_sad", "joyful_excited", 
                    "confident", "neutral", "comforting", "encouraging",
                    "thoughtful", "affectionate", "reassuring"
                ],
                "tier2_contextual": [
                    "playful", "concerned_anxious", "melancholy", "wistful",
                    "tired", "curious", "dreamy", "relieved", "proud"
                ],
                "tier3_expressive": [
                    "angry_firm", "sarcastic", "mischievous", "tender",
                    "amused", "sympathetic", "reflective", "grateful", "apologetic"
                ]
            },
            "base_emotions": self.BASE_EMOTIONS,
            "blended_emotions": list(self.BLEND_RECIPES.keys()),
            "blend_recipes": self.BLEND_RECIPES,
            "emotion_texts": self.EMOTION_TEXTS
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n📝 Saved emotion library config to {output_path}")
        return config
    
    def get_all_emotions(self) -> List[str]:
        """Get list of all available emotions (base + blended)"""
        return self.BASE_EMOTIONS + list(self.BLEND_RECIPES.keys())
    
    def get_emotion_tier(self, emotion: str) -> str:
        """Get the tier classification for an emotion"""
        if emotion in ["calm_supportive", "empathetic_sad", "joyful_excited", 
                       "confident", "neutral", "comforting", "encouraging",
                       "thoughtful", "affectionate", "reassuring"]:
            return "tier1_core"
        elif emotion in ["playful", "concerned_anxious", "melancholy", "wistful",
                         "tired", "curious", "dreamy", "relieved", "proud"]:
            return "tier2_contextual"
        else:
            return "tier3_expressive"


def main():
    """Run emotion blending pipeline"""
    print("🎨 Emotion Blender - Expanded Library Generator")
    print("=" * 60)
    
    blender = EmotionBlender()
    
    # Step 1: Load base embeddings
    blender.load_base_embeddings()
    
    # Step 2: Blend to create new emotions
    blender.blend_embeddings()
    
    # Step 3: Save blended embeddings
    blender.save_blended_embeddings()
    
    # Step 4: Generate library config
    config_path = Path("/workspace/emotion_references/emotion_library.json")
    blender.generate_emotion_library_config(config_path)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Emotion Library Complete!")
    print(f"   Base emotions: {len(blender.BASE_EMOTIONS)}")
    print(f"   Blended emotions: {len(blender.blended_embeddings)}")
    print(f"   Total emotions: {len(blender.get_all_emotions())}")
    print("\n   Tier 1 (Core): 10 emotions")
    print("   Tier 2 (Contextual): 9 emotions")
    print("   Tier 3 (Expressive): 9 emotions")
    print("=" * 60)


if __name__ == "__main__":
    main()

