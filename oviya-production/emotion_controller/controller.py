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
    
    def __init__(self, emotions_config_path: str = "config/emotions_49.json"):
        """Initialize emotion controller with emotion mappings."""
        self.config_path = Path(emotions_config_path)
        self.emotions = self._load_emotions()
        
        # Count emotions by tier
        tier_counts = {}
        for emotion, params in self.emotions.items():
            # Skip non-dict entries (comments)
            if not isinstance(params, dict):
                continue
            tier = params.get("tier", 0)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        print(f"✅ Emotion Controller initialized with {len(self.emotions)} emotions")
        if tier_counts:
            print(f"   Tier 1 (Core): {tier_counts.get(1, 0)}")
            print(f"   Tier 2 (Contextual): {tier_counts.get(2, 0)}")
            print(f"   Tier 3 (Expressive): {tier_counts.get(3, 0)}")
    
    def _load_emotions(self) -> Dict:
        """Load emotion mappings from JSON config."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            emotions = config["emotion_labels"]
            
            # Filter out comment keys (starting with _comment)
            emotions = {k: v for k, v in emotions.items() if not k.startswith("_comment")}
            
            return emotions
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
        intensity: float = 0.7,
        contextual_modifiers: Optional[Dict] = None
    ) -> Dict:
        """
        Map emotion label to acoustic parameters with enhanced intensity mapping.
        
        Args:
            emotion_label: Emotion label (e.g., "calm_supportive")
            intensity: Emotion intensity 0.0-1.0 (scales the parameters)
            contextual_modifiers: Optional modifiers from emotional memory
        
        Returns:
            Dict with style_token and acoustic parameters
        """
        # Handle blended emotions (e.g., "calm_supportive|empathetic_sad")
        if "|" in emotion_label:
            # Split and use the first emotion
            parts = emotion_label.split("|")
            original = emotion_label
            emotion_label = parts[0]
            if emotion_label not in self.emotions:
                emotion_label = "neutral"
            print(f"   🎨 Blended emotion: {original} → using {emotion_label}")
        elif emotion_label not in self.emotions:
            # Try to map common variations
            emotion_map = {
                "anxious": "concerned_anxious",
                "stressed": "concerned_anxious",
                "uncertain": "hesitant",
                "excited": "joyful_excited",
                "happy": "joyful_excited",
                "sad": "empathetic_sad"
            }
            mapped = emotion_map.get(emotion_label)
            if mapped and mapped in self.emotions:
                print(f"   🔄 Mapped emotion: {emotion_label} → {mapped}")
                emotion_label = mapped
            else:
                print(f"⚠️ Unknown emotion '{emotion_label}', using neutral")
                emotion_label = "neutral"
        
        emotion_params = self.emotions[emotion_label]
        
        # Enhanced intensity scaling with non-linear curves
        # Low intensity (0.0-0.3): subtle hints
        # Mid intensity (0.3-0.7): noticeable but natural
        # High intensity (0.7-1.0): strong but not overdone
        intensity_curve = self._apply_intensity_curve(intensity)
        
        # Scale parameters by intensity
        scaled_params = {
            "style_token": emotion_params["style_token"],
            "pitch_scale": self._scale_param_enhanced(
                emotion_params["pitch_scale"], 
                intensity_curve, 
                neutral_value=1.0,
                param_type="pitch"
            ),
            "rate_scale": self._scale_param_enhanced(
                emotion_params["rate_scale"], 
                intensity_curve, 
                neutral_value=1.0,
                param_type="rate"
            ),
            "energy_scale": self._scale_param_enhanced(
                emotion_params["energy_scale"], 
                intensity_curve, 
                neutral_value=1.0,
                param_type="energy"
            ),
            "prosody_hint": emotion_params.get("prosody_hint", ""),
            "emotion_label": emotion_label,
            "intensity": intensity,
            "intensity_curve": intensity_curve
        }
        
        # Apply contextual modifiers if provided
        if contextual_modifiers:
            scaled_params = self._apply_contextual_modifiers(
                scaled_params,
                contextual_modifiers
            )
        
        return scaled_params
    
    def _apply_intensity_curve(self, intensity: float) -> float:
        """
        Apply non-linear intensity curve for more natural scaling.
        
        Low intensities are compressed, high intensities are emphasized.
        """
        import math
        
        # Sigmoid-like curve
        # Maps [0, 1] → [0, 1] with steeper middle section
        if intensity < 0.5:
            # Gentle increase for low intensities
            return 0.5 * math.pow(2 * intensity, 1.5)
        else:
            # Steeper increase for high intensities
            return 0.5 + 0.5 * math.pow(2 * (intensity - 0.5), 0.7)
    
    def _scale_param_enhanced(
        self, 
        param_value: float, 
        intensity: float, 
        neutral_value: float = 1.0,
        param_type: str = "generic"
    ) -> float:
        """
        Enhanced parameter scaling with type-specific behavior.
        
        Args:
            param_value: Target parameter value at full intensity
            intensity: Scaled intensity (0.0-1.0)
            neutral_value: Neutral baseline value
            param_type: Type of parameter (pitch, rate, energy)
        
        Returns:
            Scaled parameter value
        """
        
        # Different scaling strategies per parameter type
        if param_type == "pitch":
            # Pitch changes are most noticeable, scale conservatively
            # At 50% intensity, only apply 30% of the pitch change
            scaling_factor = 0.6 * intensity + 0.4 * (intensity ** 2)
        elif param_type == "rate":
            # Rate changes need to be subtle to avoid sounding unnatural
            # At 50% intensity, apply 40% of rate change
            scaling_factor = 0.8 * intensity + 0.2 * (intensity ** 2)
        elif param_type == "energy":
            # Energy can be more dynamic
            # At 50% intensity, apply 60% of energy change
            scaling_factor = intensity
        else:
            # Generic linear scaling
            scaling_factor = intensity
        
        return neutral_value + (param_value - neutral_value) * scaling_factor
    
    def _apply_contextual_modifiers(
        self,
        params: Dict,
        modifiers: Dict
    ) -> Dict:
        """
        Apply contextual modifiers from emotional memory.
        
        This blends the current emotion with the conversation's overall mood.
        """
        
        # Apply energy level from emotional memory
        if "energy_scale" in modifiers:
            params["energy_scale"] *= (0.8 + 0.4 * modifiers["energy_scale"])
        
        # Apply pace from emotional memory
        if "pace_scale" in modifiers:
            params["rate_scale"] *= (0.9 + 0.2 * modifiers["pace_scale"])
        
        # Apply warmth (affects pitch subtly)
        if "warmth_scale" in modifiers:
            warmth_offset = (modifiers["warmth_scale"] - 0.5) * 0.05
            params["pitch_scale"] *= (1.0 + warmth_offset)
        
        return params
    
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

