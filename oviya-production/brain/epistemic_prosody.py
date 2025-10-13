"""
Epistemic Prosody Module
Detects uncertainty in text and modulates prosody accordingly
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np


class EpistemicProsodyAnalyzer:
    """Analyzes text for epistemic markers (uncertainty, confidence) and adjusts prosody"""
    
    # Uncertainty markers that trigger rising intonation
    UNCERTAINTY_MARKERS = {
        # High uncertainty
        "high": [
            "maybe", "perhaps", "possibly", "potentially", "presumably",
            "I think", "I believe", "I guess", "I suppose", "I assume",
            "probably", "likely", "might be", "could be", "seems like",
            "not sure", "not certain", "uncertain", "unsure"
        ],
        # Medium uncertainty
        "medium": [
            "fairly", "somewhat", "kind of", "sort of", "rather",
            "appears to", "seems to", "looks like", "sounds like"
        ],
        # Questions (inherent uncertainty)
        "question": [
            "?", "right?", "isn't it?", "don't you think?", "you know?"
        ]
    }
    
    # Confidence markers that trigger falling intonation
    CONFIDENCE_MARKERS = {
        "high": [
            "definitely", "absolutely", "certainly", "surely", "clearly",
            "obviously", "undoubtedly", "without doubt", "for sure",
            "I know", "I'm certain", "I'm sure", "I'm confident",
            "certain this will", "absolutely certain"
        ],
        "medium": [
            "indeed", "in fact", "actually", "really", "truly"
        ]
    }
    
    # Thinking/processing markers (creaky voice, slower pace)
    THINKING_MARKERS = [
        "hmm", "umm", "uh", "well", "let me think", "let's see",
        "how should I put this", "what I mean is"
    ]
    
    def __init__(self):
        """Initialize the epistemic prosody analyzer"""
        self.pitch_modulation_factors = {
            "high_uncertainty": 1.15,  # 15% pitch rise
            "medium_uncertainty": 1.08,  # 8% pitch rise
            "question": 1.20,  # 20% pitch rise for questions
            "high_confidence": 0.92,  # 8% pitch drop
            "medium_confidence": 0.96,  # 4% pitch drop
            "thinking": 0.95  # 5% pitch drop with creaky quality
        }
        
        self.pace_modulation_factors = {
            "uncertainty": 0.9,  # Slower when uncertain
            "confidence": 1.1,   # Faster when confident
            "thinking": 0.7      # Much slower when thinking
        }
    
    def analyze_epistemic_state(self, text: str) -> Dict:
        """
        Analyze text for epistemic markers and return prosody modifications
        
        Returns:
            Dict with epistemic state and prosody parameters
        """
        text_lower = text.lower()
        
        # Initialize results
        result = {
            "epistemic_state": "neutral",
            "confidence_level": 0.5,  # 0=uncertain, 1=confident
            "pitch_contour": 1.0,  # Multiplier for pitch
            "pace_factor": 1.0,  # Multiplier for speaking pace
            "markers_found": [],
            "prosody_hints": []
        }
        
        # Check for uncertainty markers
        uncertainty_level = self._check_uncertainty(text_lower)
        confidence_level = self._check_confidence(text_lower)
        is_thinking = self._check_thinking(text_lower)
        is_question = "?" in text
        
        # Determine dominant state
        if is_thinking:
            result["epistemic_state"] = "thinking"
            result["confidence_level"] = 0.3
            result["pitch_contour"] = self.pitch_modulation_factors["thinking"]
            result["pace_factor"] = self.pace_modulation_factors["thinking"]
            result["prosody_hints"].append("add_creaky_voice")
            
        elif is_question or uncertainty_level > confidence_level:
            if is_question:
                result["epistemic_state"] = "questioning"
                result["pitch_contour"] = self.pitch_modulation_factors["question"]
            elif uncertainty_level >= 0.7:
                result["epistemic_state"] = "high_uncertainty"
                result["pitch_contour"] = self.pitch_modulation_factors["high_uncertainty"]
            else:
                result["epistemic_state"] = "medium_uncertainty"
                result["pitch_contour"] = self.pitch_modulation_factors["medium_uncertainty"]
            
            result["confidence_level"] = 1.0 - uncertainty_level
            result["pace_factor"] = self.pace_modulation_factors["uncertainty"]
            result["prosody_hints"].append("rising_intonation")
            
        elif confidence_level > 0.5:
            if confidence_level >= 0.8:
                result["epistemic_state"] = "high_confidence"
                result["pitch_contour"] = self.pitch_modulation_factors["high_confidence"]
            else:
                result["epistemic_state"] = "medium_confidence"
                result["pitch_contour"] = self.pitch_modulation_factors["medium_confidence"]
            
            result["confidence_level"] = confidence_level
            result["pace_factor"] = self.pace_modulation_factors["confidence"]
            result["prosody_hints"].append("falling_intonation")
        
        return result
    
    def _check_uncertainty(self, text: str) -> float:
        """Check text for uncertainty markers and return level (0-1)"""
        found_markers = []
        
        # Check high uncertainty markers
        for marker in self.UNCERTAINTY_MARKERS["high"]:
            if marker in text:
                found_markers.append(("high", marker))
        
        # Check medium uncertainty markers
        for marker in self.UNCERTAINTY_MARKERS["medium"]:
            if marker in text:
                found_markers.append(("medium", marker))
        
        # Check question markers
        for marker in self.UNCERTAINTY_MARKERS["question"]:
            if marker in text:
                found_markers.append(("question", marker))
        
        if not found_markers:
            return 0.0
        
        # Calculate uncertainty level based on markers found
        high_count = sum(1 for level, _ in found_markers if level == "high")
        medium_count = sum(1 for level, _ in found_markers if level == "medium")
        question_count = sum(1 for level, _ in found_markers if level == "question")
        
        # Weighted scoring
        score = (high_count * 1.0 + medium_count * 0.5 + question_count * 1.2)
        return min(1.0, score / 3.0)  # Normalize to 0-1
    
    def _check_confidence(self, text: str) -> float:
        """Check text for confidence markers and return level (0-1)"""
        found_markers = []
        
        # Check high confidence markers
        for marker in self.CONFIDENCE_MARKERS["high"]:
            if marker in text:
                found_markers.append(("high", marker))
        
        # Check medium confidence markers
        for marker in self.CONFIDENCE_MARKERS["medium"]:
            if marker in text:
                found_markers.append(("medium", marker))
        
        if not found_markers:
            return 0.0
        
        # Calculate confidence level
        high_count = sum(1 for level, _ in found_markers if level == "high")
        medium_count = sum(1 for level, _ in found_markers if level == "medium")
        
        score = (high_count * 1.0 + medium_count * 0.5)
        return min(1.0, score / 2.0)  # Normalize to 0-1
    
    def _check_thinking(self, text: str) -> bool:
        """Check if text contains thinking/processing markers"""
        return any(marker in text for marker in self.THINKING_MARKERS)
    
    def apply_to_prosodic_markup(self, text: str, existing_markup: str) -> str:
        """
        Apply epistemic prosody to existing prosodic markup
        
        Args:
            text: Original text
            existing_markup: Text with existing prosodic markup
            
        Returns:
            Enhanced markup with epistemic markers
        """
        analysis = self.analyze_epistemic_state(text)
        
        # Add epistemic tags based on state
        if analysis["epistemic_state"] == "thinking":
            # Add thinking markers
            existing_markup = f"<thinking> {existing_markup}"
            
        elif "rising_intonation" in analysis["prosody_hints"]:
            # Add uncertainty markers
            existing_markup = f"<uncertain> {existing_markup} <rising>"
            
        elif "falling_intonation" in analysis["prosody_hints"]:
            # Add confidence markers
            existing_markup = f"<confident> {existing_markup}"
        
        return existing_markup
    
    def get_audio_modulation_params(self, text: str) -> Dict:
        """
        Get audio modulation parameters for epistemic prosody
        
        Returns:
            Dict with audio processing parameters
        """
        analysis = self.analyze_epistemic_state(text)
        
        params = {
            "pitch_shift_cents": 0,  # Pitch shift in cents (100 cents = 1 semitone)
            "tempo_factor": analysis["pace_factor"],
            "add_creaky": False,
            "add_shimmer": False,
            "formant_shift": 1.0
        }
        
        # Convert pitch contour to cents
        if analysis["pitch_contour"] != 1.0:
            # Convert ratio to cents: cents = 1200 * log2(ratio)
            import math
            params["pitch_shift_cents"] = int(1200 * math.log2(analysis["pitch_contour"]))
        
        # Add voice quality modifications
        if analysis["epistemic_state"] == "thinking":
            params["add_creaky"] = True
            params["formant_shift"] = 0.95  # Slightly lower formants
            
        elif analysis["epistemic_state"] in ["high_uncertainty", "questioning"]:
            params["add_shimmer"] = True  # Slight amplitude variation
            
        elif analysis["epistemic_state"] == "high_confidence":
            params["formant_shift"] = 1.05  # Slightly raise formants for assertiveness
        
        return params


def test_epistemic_prosody():
    """Test the epistemic prosody analyzer"""
    analyzer = EpistemicProsodyAnalyzer()
    
    test_sentences = [
        "I think maybe we should try that approach.",
        "I'm absolutely certain this will work!",
        "Hmm, let me think about that for a moment.",
        "Could this possibly be the right answer?",
        "Obviously, this is the best solution.",
        "Well, I suppose it might work, perhaps.",
        "I know exactly what to do here.",
        "Um, I'm not really sure about this.",
        "Definitely the way to go!",
        "What do you think about this idea?"
    ]
    
    print("🧠 Testing Epistemic Prosody Analyzer\n")
    print("=" * 60)
    
    for sentence in test_sentences:
        analysis = analyzer.analyze_epistemic_state(sentence)
        params = analyzer.get_audio_modulation_params(sentence)
        
        print(f"\nText: {sentence}")
        print(f"State: {analysis['epistemic_state']}")
        print(f"Confidence: {analysis['confidence_level']:.2f}")
        print(f"Pitch: {params['pitch_shift_cents']:+d} cents")
        print(f"Tempo: {params['tempo_factor']:.2f}x")
        if params['add_creaky']:
            print("Voice: Creaky")
        if params['add_shimmer']:
            print("Voice: Shimmer")
        print("-" * 40)


if __name__ == "__main__":
    test_epistemic_prosody()
