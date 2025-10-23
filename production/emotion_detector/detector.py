"""
Emotion Detector - Analyzes user's emotional state from text

This module detects the user's emotional state to provide context
to Oviya's brain for more empathetic responses.
"""

import json
import re
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class EmotionDetector:
    """
    Detects user's emotional state from text input.
    
    Uses keyword-based detection with confidence scoring.
    Can be extended with ML models for more sophisticated detection.
    """
    
    def __init__(self, emotions_config_path: str = "config/emotions.json"):
        """Initialize emotion detector."""
        self.config_path = Path(emotions_config_path)
        self.emotion_keywords = self._load_emotion_keywords()
        self.intensity_keywords = self._load_intensity_keywords()
        
        print(f"Emotion Detector initialized with {len(self.emotion_keywords)} emotion categories")
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """Load emotion keywords from config."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get("emotion_keywords", {})
        except FileNotFoundError:
            print(f"Warning: Emotions config not found, using defaults")
            return self._get_default_emotion_keywords()
        except Exception as e:
            print(f"Error loading emotions config: {e}")
            return self._get_default_emotion_keywords()
    
    def _get_default_emotion_keywords(self) -> Dict[str, List[str]]:
        """Fallback default emotion keywords."""
        return {
            "calm_supportive": ["stressed", "overwhelmed", "anxious", "worried", "nervous", "scared", "panic"],
            "empathetic_sad": ["sad", "depressed", "down", "lonely", "hurt", "alone", "crying", "grief"],
            "joyful_excited": ["excited", "happy", "great", "amazing", "wonderful", "promoted", "fantastic", "thrilled"],
            "playful": ["funny", "laugh", "joke", "silly", "fun", "humor"],
            "confident": ["can", "will", "strong", "capable", "confident", "proud"],
            "concerned_anxious": ["concern", "worry", "issue", "problem", "trouble"],
            "angry_firm": ["angry", "mad", "frustrated", "furious", "annoyed", "irritated"],
            "neutral": []
        }
    
    def _load_intensity_keywords(self) -> Dict[str, float]:
        """Load intensity modifiers."""
        return {
            # High intensity
            "extremely": 0.9,
            "very": 0.8,
            "really": 0.8,
            "so": 0.7,
            "incredibly": 0.9,
            "terribly": 0.8,
            "awfully": 0.8,
            
            # Medium intensity
            "quite": 0.6,
            "pretty": 0.6,
            "rather": 0.5,
            "somewhat": 0.4,
            
            # Low intensity
            "a bit": 0.3,
            "slightly": 0.2,
            "little": 0.2,
            "kind of": 0.3
        }
    
    def detect_emotion(
        self, 
        text: str, 
        context: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Detect user's emotional state from text.
        
        Args:
            text: User's input text
            context: Additional context (optional)
        
        Returns:
            Dict with emotion, intensity, confidence, and reasoning
        """
        text_lower = text.lower()
        
        # Score each emotion category
        emotion_scores = {}
        matched_keywords = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            matched = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Calculate keyword score based on intensity modifiers
                    intensity = self._calculate_keyword_intensity(text_lower, keyword)
                    score += intensity
                    matched.append(f"{keyword}({intensity:.2f})")
            
            if score > 0:
                emotion_scores[emotion] = score
                matched_keywords[emotion] = matched
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(emotion_scores[primary_emotion] / 2.0, 1.0)  # Normalize to 0-1
            intensity = min(emotion_scores[primary_emotion], 1.0)
        else:
            primary_emotion = "neutral"
            confidence = 0.5
            intensity = 0.5
        
        # Additional context analysis
        context_emotion = self._analyze_context(text_lower)
        if context_emotion and context_emotion != primary_emotion:
            # Blend context emotion
            primary_emotion = self._blend_emotions(primary_emotion, context_emotion)
        
        return {
            "emotion": primary_emotion,
            "intensity": intensity,
            "confidence": confidence,
            "matched_keywords": matched_keywords.get(primary_emotion, []),
            "reasoning": f"Detected '{primary_emotion}' from keywords: {matched_keywords.get(primary_emotion, [])}",
            "all_scores": emotion_scores
        }
    
    def _calculate_keyword_intensity(self, text: str, keyword: str) -> float:
        """Calculate intensity of a keyword based on modifiers."""
        base_intensity = 1.0
        
        # Find intensity modifiers near the keyword
        keyword_pos = text.find(keyword)
        if keyword_pos == -1:
            return base_intensity
        
        # Check words before keyword
        words_before = text[max(0, keyword_pos-50):keyword_pos].split()
        for word in words_before[-3:]:  # Last 3 words before keyword
            if word in self.intensity_keywords:
                base_intensity *= self.intensity_keywords[word]
                break
        
        return base_intensity
    
    def _analyze_context(self, text: str) -> Optional[str]:
        """Analyze contextual clues for emotion."""
        # Question patterns suggest uncertainty/concern
        if any(pattern in text for pattern in ["what should", "what do you think", "help me", "advice"]):
            return "concerned_anxious"
        
        # Exclamation patterns suggest excitement
        if text.count('!') > 1:
            return "joyful_excited"
        
        # Negative patterns
        if any(word in text for word in ["not", "no", "never", "can't", "won't"]):
            return "empathetic_sad"
        
        return None
    
    def _blend_emotions(self, emotion1: str, emotion2: str) -> str:
        """Blend two emotions (simple heuristic)."""
        # Priority mapping for blending
        blend_map = {
            ("concerned_anxious", "empathetic_sad"): "empathetic_sad",
            ("joyful_excited", "playful"): "joyful_excited",
            ("calm_supportive", "concerned_anxious"): "calm_supportive",
        }
        
        blend_key = tuple(sorted([emotion1, emotion2]))
        return blend_map.get(blend_key, emotion1)
    
    def get_emotion_summary(self, detection_result: Dict) -> str:
        """Get human-readable emotion summary."""
        emotion = detection_result["emotion"]
        intensity = detection_result["intensity"]
        confidence = detection_result["confidence"]
        
        intensity_desc = "very strong" if intensity > 0.8 else "strong" if intensity > 0.6 else "moderate" if intensity > 0.4 else "mild"
        
        return f"User emotion: {emotion} ({intensity_desc}, confidence: {confidence:.2f})"
    
    def detect_multiple_emotions(self, text: str) -> List[Dict]:
        """Detect multiple emotions present in text."""
        text_lower = text.lower()
        emotions = []
        
        for emotion, keywords in self.emotion_keywords.items():
            matched = [kw for kw in keywords if kw in text_lower]
            if matched:
                intensity = len(matched) / len(keywords)  # Simple scoring
                emotions.append({
                    "emotion": emotion,
                    "intensity": intensity,
                    "keywords": matched
                })
        
        # Sort by intensity
        emotions.sort(key=lambda x: x["intensity"], reverse=True)
        return emotions


# Example usage
if __name__ == "__main__":
    detector = EmotionDetector()
    
    print("\nTesting Emotion Detector\n")
    
    test_texts = [
        "I'm feeling really stressed about work today",
        "I got promoted! I'm so excited!",
        "I'm feeling sad and lonely",
        "I'm frustrated with everything",
        "Just checking in, how are you?",
        "I'm extremely worried about my future",
        "This is incredibly frustrating!"
    ]
    
    for text in test_texts:
        print(f"Text: '{text}'")
        result = detector.detect_emotion(text)
        print(f"  Emotion: {result['emotion']}")
        print(f"  Intensity: {result['intensity']:.2f}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Keywords: {result['matched_keywords']}")
        print(f"  Summary: {detector.get_emotion_summary(result)}")
        print()

