"""
Emotion Controller - Maps emotion labels to acoustic parameters

This is the bridge between Oviya's brain (LLM) and voice (OpenS2S).
It translates high-level emotion labels into concrete acoustic control parameters.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import re


class EmotionController:
    """
    Maps emotion labels to acoustic parameters for voice generation.
    
    This is a lightweight, pure-Python mapping layer that requires no ML training.
    """
    
    def __init__(self, emotions_config_path: str = "config/emotions.json"):
        """Initialize emotion controller with emotion mappings."""
        self.config_path = Path(emotions_config_path)
        self.emotions = self._load_emotions()
        print(f"✅ Emotion Controller initialized with {len(self.emotions)} emotions")
    
    def _load_emotions(self) -> Dict:
        """Load emotion mappings from JSON config."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config["emotion_labels"]
        except FileNotFoundError:
            print(f"⚠️ Emotions config not found at {self.config_path}, using defaults")
            return self._get_default_emotions()
        except Exception as e:
            print(f"❌ Error loading emotions config: {e}")
            return self._get_default_emotions()
    
    def _get_default_emotions(self) -> Dict:
        """Fallback default emotion mappings."""
        return {
            "calm_supportive": {
                "style_token": "#calm",
                "pitch_scale": 0.9,
                "rate_scale": 0.9,
                "energy_scale": 0.8
            },
            "empathetic_sad": {
                "style_token": "#sad",
                "pitch_scale": 0.85,
                "rate_scale": 0.85,
                "energy_scale": 0.6
            },
            "joyful_excited": {
                "style_token": "#joy",
                "pitch_scale": 1.15,
                "rate_scale": 1.05,
                "energy_scale": 1.2
            },
            "neutral": {
                "style_token": "#neutral",
                "pitch_scale": 1.0,
                "rate_scale": 1.0,
                "energy_scale": 1.0
            }
        }
    
    def map_emotion(
        self, 
        emotion_label: str, 
        intensity: float = 0.7
    ) -> Dict:
        """
        Map emotion label to acoustic parameters.
        
        Args:
            emotion_label: Emotion label (e.g., "calm_supportive")
            intensity: Emotion intensity 0.0-1.0 (scales the parameters)
        
        Returns:
            Dict with style_token and acoustic parameters
        """
        # Get base emotion parameters
        if emotion_label not in self.emotions:
            print(f"⚠️ Unknown emotion '{emotion_label}', using neutral")
            emotion_label = "neutral"
        
        emotion_params = self.emotions[emotion_label]
        
        # Scale parameters by intensity
        scaled_params = {
            "style_token": emotion_params["style_token"],
            "pitch_scale": self._scale_param(
                emotion_params["pitch_scale"], 
                intensity, 
                neutral_value=1.0
            ),
            "rate_scale": self._scale_param(
                emotion_params["rate_scale"], 
                intensity, 
                neutral_value=1.0
            ),
            "energy_scale": self._scale_param(
                emotion_params["energy_scale"], 
                intensity, 
                neutral_value=1.0
            ),
            "prosody_hint": emotion_params.get("prosody_hint", ""),
            "emotion_label": emotion_label,
            "intensity": intensity
        }
        
        return scaled_params
    
    def _scale_param(
        self, 
        param_value: float, 
        intensity: float, 
        neutral_value: float = 1.0
    ) -> float:
        """
        Scale a parameter by intensity.
        
        Interpolates between neutral_value and param_value based on intensity.
        intensity=0.0 -> neutral_value
        intensity=1.0 -> param_value
        """
        return neutral_value + (param_value - neutral_value) * intensity
    
    def detect_emotion_from_text(self, text: str) -> str:
        """
        Simple keyword-based emotion detection from text.
        
        This is a fallback if the LLM doesn't provide an emotion label.
        """
        text_lower = text.lower()
        
        # Load keyword mappings
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            emotion_keywords = config.get("emotion_keywords", {})
        except:
            emotion_keywords = {}
        
        # Check keywords for each emotion
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion
        
        return "neutral"
    
    def validate_emotion_label(self, emotion_label: str) -> bool:
        """Check if emotion label is valid."""
        return emotion_label in self.emotions
    
    def get_available_emotions(self) -> list:
        """Get list of all available emotion labels."""
        return list(self.emotions.keys())
    
    def get_emotion_description(self, emotion_label: str) -> str:
        """Get human-readable description of an emotion."""
        if emotion_label in self.emotions:
            return self.emotions[emotion_label].get("description", "")
        return ""


# Example usage
if __name__ == "__main__":
    # Initialize controller
    controller = EmotionController()
    
    # Test emotion mapping
    print("\n🧪 Testing Emotion Controller\n")
    
    test_emotions = [
        ("calm_supportive", 0.8),
        ("empathetic_sad", 0.7),
        ("joyful_excited", 0.9),
        ("neutral", 0.5)
    ]
    
    for emotion, intensity in test_emotions:
        params = controller.map_emotion(emotion, intensity)
        print(f"Emotion: {emotion} (intensity: {intensity})")
        print(f"  Style Token: {params['style_token']}")
        print(f"  Pitch: {params['pitch_scale']:.2f}")
        print(f"  Rate: {params['rate_scale']:.2f}")
        print(f"  Energy: {params['energy_scale']:.2f}")
        print()
    
    # Test emotion detection
    print("\n🧪 Testing Emotion Detection\n")
    test_texts = [
        "I'm feeling so stressed and overwhelmed",
        "I got promoted today!",
        "I'm really sad and lonely"
    ]
    
    for text in test_texts:
        detected = controller.detect_emotion_from_text(text)
        print(f"Text: '{text}'")
        print(f"  Detected: {detected}\n")

