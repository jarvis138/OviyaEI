"""
Emotion Library Manager
Manages expanded emotion library with 28+ emotions across 3 tiers
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


class EmotionLibrary:
    """Manages emotion selection and mapping for Oviya"""
    
    # Emotion tier weights (for sampling during training/usage)
    TIER_WEIGHTS = {
        "tier1_core": 0.70,      # 70% - everyday emotions
        "tier2_contextual": 0.25, # 25% - situational
        "tier3_expressive": 0.05  # 5% - rare/dramatic
    }
    
    # Emotion mappings from LLM outputs to library
    EMOTION_ALIASES = {
        "happy": "joyful_excited",
        "sad": "empathetic_sad",
        "angry": "angry_firm",
        "worried": "concerned_anxious",
        "calm": "calm_supportive",
        "excited": "joyful_excited",
        "supportive": "comforting",
        "understanding": "empathetic_sad",
        "enthusiastic": "encouraging",
        "caring": "affectionate",
        "peaceful": "calm_supportive",
        "stressed": "concerned_anxious",
        "frustrated": "angry_firm",
        "loving": "tender",
        "funny": "amused",
        "serious": "confident",
        "pensive": "thoughtful"
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize emotion library"""
        self.config_path = config_path or Path("config/emotion_library.json")
        self.emotions = {}
        self.tiers = {}
        self.load_config()
    
    def load_config(self):
        """Load emotion library configuration"""
        if not self.config_path.exists():
            # Use default config if file doesn't exist
            self._load_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.tiers = config.get("tiers", {})
            
            # Build flat emotion list with tier info
            for tier_name, emotion_list in self.tiers.items():
                for emotion in emotion_list:
                    self.emotions[emotion] = tier_name
            
            print(f"📚 Loaded {len(self.emotions)} emotions from library")
            
        except Exception as e:
            print(f"⚠️  Failed to load emotion config: {e}")
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default emotion configuration"""
        self.tiers = {
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
        }
        
        for tier_name, emotion_list in self.tiers.items():
            for emotion in emotion_list:
                self.emotions[emotion] = tier_name
    
    def get_emotion(self, emotion_label: str) -> str:
        """
        Get library emotion name from label (with alias resolution)
        
        Args:
            emotion_label: Emotion from LLM or user input
            
        Returns:
            Library emotion name
        """
        # Direct match
        if emotion_label in self.emotions:
            return emotion_label
        
        # Check aliases
        if emotion_label in self.EMOTION_ALIASES:
            return self.EMOTION_ALIASES[emotion_label]
        
        # Fallback to neutral
        print(f"⚠️  Unknown emotion '{emotion_label}', using 'neutral'")
        return "neutral"
    
    def get_tier(self, emotion: str) -> str:
        """Get tier classification for an emotion"""
        return self.emotions.get(emotion, "tier1_core")
    
    def sample_emotion(self, tier: Optional[str] = None) -> str:
        """
        Sample a random emotion (optionally from specific tier)
        
        Args:
            tier: Optional tier to sample from
            
        Returns:
            Random emotion name
        """
        if tier and tier in self.tiers:
            return random.choice(self.tiers[tier])
        
        # Weighted sampling across all tiers
        tier_choice = random.choices(
            list(self.tiers.keys()),
            weights=[self.TIER_WEIGHTS.get(t, 0.33) for t in self.tiers.keys()],
            k=1
        )[0]
        
        return random.choice(self.tiers[tier_choice])
    
    def get_tier_emotions(self, tier: str) -> List[str]:
        """Get all emotions in a tier"""
        return self.tiers.get(tier, [])
    
    def get_all_emotions(self) -> List[str]:
        """Get all emotion names"""
        return list(self.emotions.keys())
    
    def get_emotion_stats(self) -> Dict:
        """Get statistics about the emotion library"""
        return {
            "total_emotions": len(self.emotions),
            "tier1_count": len(self.tiers.get("tier1_core", [])),
            "tier2_count": len(self.tiers.get("tier2_contextual", [])),
            "tier3_count": len(self.tiers.get("tier3_expressive", [])),
            "aliases": len(self.EMOTION_ALIASES)
        }
    
    def validate_emotion(self, emotion: str) -> Tuple[bool, str]:
        """
        Validate if an emotion exists in the library
        
        Returns:
            (is_valid, resolved_emotion)
        """
        if emotion in self.emotions:
            return True, emotion
        
        if emotion in self.EMOTION_ALIASES:
            return True, self.EMOTION_ALIASES[emotion]
        
        return False, "neutral"
    
    def get_similar_emotions(self, emotion: str, count: int = 3) -> List[str]:
        """
        Get similar emotions from the same tier
        
        Args:
            emotion: Source emotion
            count: Number of similar emotions to return
            
        Returns:
            List of similar emotion names
        """
        tier = self.get_tier(emotion)
        tier_emotions = [e for e in self.get_tier_emotions(tier) if e != emotion]
        
        if len(tier_emotions) <= count:
            return tier_emotions
        
        return random.sample(tier_emotions, count)


# Global emotion library instance
_emotion_library = None

def get_emotion_library() -> EmotionLibrary:
    """Get global emotion library instance (singleton)"""
    global _emotion_library
    if _emotion_library is None:
        _emotion_library = EmotionLibrary()
    return _emotion_library


def main():
    """Test emotion library"""
    print("🧪 Testing Emotion Library\n")
    
    library = EmotionLibrary()
    
    # Stats
    stats = library.get_emotion_stats()
    print("📊 Library Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test aliases
    print("\n🔄 Testing Emotion Aliases:")
    test_aliases = ["happy", "sad", "worried", "unknown_emotion"]
    for alias in test_aliases:
        resolved = library.get_emotion(alias)
        print(f"   '{alias}' → '{resolved}'")
    
    # Sample emotions
    print("\n🎲 Random Sampling:")
    for _ in range(5):
        emotion = library.sample_emotion()
        tier = library.get_tier(emotion)
        print(f"   {emotion} ({tier})")
    
    # Similar emotions
    print("\n🔍 Similar Emotions:")
    test_emotion = "joyful_excited"
    similar = library.get_similar_emotions(test_emotion)
    print(f"   Similar to '{test_emotion}': {', '.join(similar)}")


if __name__ == "__main__":
    main()

