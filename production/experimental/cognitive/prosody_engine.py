#!/usr/bin/env python3
"""
Phase 8: Prosody Computation (Voice Modulation)

Computes voice parameters for emotional expression in speech synthesis.
Controls fundamental frequency, energy, speech rate, and pitch dynamics
based on detected emotion, personality, and therapeutic context.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ProsodyEngine:
    """
    Phase 8: Voice Prosody Computation

    Calculates acoustic parameters for emotional voice modulation in CSM-1B.
    """

    def __init__(self):
        """Initialize prosody engine"""
        # Base prosody parameters for neutral speech
        self.neutral_prosody = {
            'f0_mean': 0.0,      # Fundamental frequency (semitones from neutral)
            'f0_range': 0.0,     # Pitch variation range
            'energy': 0.0,       # Volume/energy level
            'speech_rate': 1.0,  # Speaking rate multiplier
            'pause_probability': 0.1,  # Probability of mid-sentence pauses
            'intonation_curve': 'neutral'  # Overall intonation pattern
        }

        # Emotion-specific prosody mappings
        self.emotion_prosody = {
            'anxiety': {
                'f0_mean': 0.15,     # Slightly higher pitch (tension)
                'f0_range': 0.25,    # More pitch variation (uncertainty)
                'energy': -0.1,      # Slightly softer (hesitation)
                'speech_rate': 1.1,  # Slightly faster (rushed)
                'pause_probability': 0.3,  # More pauses (gathering thoughts)
                'intonation_curve': 'rising-falling'  # Uncertainty pattern
            },
            'sadness': {
                'f0_mean': -0.2,     # Lower pitch (depression)
                'f0_range': -0.1,    # Reduced variation (monotone)
                'energy': -0.3,      # Softer (lack of energy)
                'speech_rate': 0.85, # Slower (deliberate)
                'pause_probability': 0.4,  # More pauses (processing grief)
                'intonation_curve': 'falling'  # Downward intonation
            },
            'grief': {
                'f0_mean': -0.25,    # Significantly lower pitch
                'f0_range': -0.2,    # Very reduced variation
                'energy': -0.4,      # Much softer
                'speech_rate': 0.75, # Much slower (contemplative)
                'pause_probability': 0.5,  # Frequent pauses (deep processing)
                'intonation_curve': 'very_falling'  # Strongly downward
            },
            'anger': {
                'f0_mean': 0.3,      # Higher pitch (intensity)
                'f0_range': 0.4,     # Wide variation (emphasis)
                'energy': 0.4,       # Louder (forceful)
                'speech_rate': 1.2,  # Faster (urgency)
                'pause_probability': 0.1,  # Fewer pauses (continuous)
                'intonation_curve': 'sharp_rising'  # Sharp rises
            },
            'joy': {
                'f0_mean': 0.25,     # Higher pitch (excitement)
                'f0_range': 0.35,    # Wide variation (enthusiasm)
                'energy': 0.2,       # More energetic
                'speech_rate': 1.05, # Slightly faster (lively)
                'pause_probability': 0.15,  # Moderate pauses
                'intonation_curve': 'rising'  # Upward intonation
            },
            'calm': {
                'f0_mean': -0.05,    # Slightly lower (steady)
                'f0_range': -0.05,   # Reduced variation (stable)
                'energy': -0.1,      # Slightly softer (gentle)
                'speech_rate': 0.9,  # Slightly slower (measured)
                'pause_probability': 0.2,  # Moderate pauses
                'intonation_curve': 'flat'  # Even intonation
            },
            'fear': {
                'f0_mean': 0.2,      # Higher pitch (alarm)
                'f0_range': 0.3,     # Wide variation (trembling)
                'energy': 0.1,       # Moderate energy
                'speech_rate': 1.15, # Faster (urgency)
                'pause_probability': 0.35,  # More pauses (catching breath)
                'intonation_curve': 'trembling'  # Shaky intonation
            },
            'hope': {
                'f0_mean': 0.1,      # Slightly higher (optimism)
                'f0_range': 0.2,     # Moderate variation
                'energy': 0.15,      # More energetic
                'speech_rate': 0.95, # Slightly slower (thoughtful)
                'pause_probability': 0.25,  # Moderate pauses
                'intonation_curve': 'hopeful_rising'  # Rising at end
            },
            'confusion': {
                'f0_mean': 0.05,     # Slightly higher (questioning)
                'f0_range': 0.15,    # Moderate variation
                'energy': -0.05,     # Slightly softer
                'speech_rate': 0.9,  # Slower (processing)
                'pause_probability': 0.4,  # More pauses (thinking)
                'intonation_curve': 'questioning'  # Rising questions
            },
            'gratitude': {
                'f0_mean': -0.1,     # Slightly lower (sincere)
                'f0_range': 0.1,     # Moderate variation
                'energy': -0.1,      # Slightly softer (warm)
                'speech_rate': 0.9,  # Slightly slower (meaningful)
                'pause_probability': 0.3,  # More pauses (feeling)
                'intonation_curve': 'warm_falling'  # Warm, downward
            },
            'shame': {
                'f0_mean': -0.15,    # Lower pitch (withdrawn)
                'f0_range': -0.1,    # Reduced variation (hesitant)
                'energy': -0.25,     # Softer (reticent)
                'speech_rate': 0.8,  # Slower (careful)
                'pause_probability': 0.45,  # Many pauses (hesitation)
                'intonation_curve': 'hesitant'  # Hesitant pattern
            }
        }

        logger.info("🎭 Prosody Engine initialized")

    def compute_prosody_parameters(
        self,
        emotion: str,
        intensity: float,
        personality_vector: Dict[str, Any],
        response_text: str,
        reciprocal_emotion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute complete prosody parameters for voice modulation

        Args:
            emotion: Primary detected emotion
            intensity: Emotion intensity (0-1)
            personality_vector: Oviya's personality state
            response_text: Text being synthesized
            reciprocal_emotion: Oviya's reciprocal emotion

        Returns:
            Complete prosody parameter set
        """
        # Start with neutral prosody
        prosody = self.neutral_prosody.copy()

        # Apply emotion-specific prosody
        if emotion in self.emotion_prosody:
            emotion_params = self.emotion_prosody[emotion]

            # Blend emotion parameters based on intensity
            blend_factor = min(1.0, intensity * 1.2)  # Max 120% emotion influence

            for param in prosody:
                if param in emotion_params:
                    emotion_value = emotion_params[param]
                    # Only apply blend_factor to numeric parameters
                    if isinstance(emotion_value, (int, float)) and isinstance(prosody[param], (int, float)):
                        prosody[param] += emotion_value * blend_factor
                    else:
                        # For non-numeric parameters (like intonation_curve), use directly
                        prosody[param] = emotion_value

        # Apply personality modulation
        prosody = self._apply_personality_modulation(prosody, personality_vector)

        # Apply reciprocal emotion influence
        prosody = self._apply_reciprocal_influence(prosody, reciprocal_emotion)

        # Adjust for text characteristics
        prosody = self._adjust_for_text_characteristics(prosody, response_text)

        # Normalize and validate parameters
        prosody = self._normalize_prosody_parameters(prosody)

        # Generate prosody markers for CSM-1B
        prosody_markers = self._generate_prosody_markers(prosody, response_text)

        return {
            'parameters': prosody,
            'markers': prosody_markers,
            'emotion_influence': emotion,
            'intensity_influence': intensity,
            'personality_influence': personality_vector.get('dominant_pillar', 'balanced'),
            'explanation': self._generate_prosody_explanation(prosody, emotion, intensity)
        }

    def _apply_personality_modulation(
        self,
        prosody: Dict[str, float],
        personality_vector: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Apply personality-based modulation to prosody

        Args:
            prosody: Current prosody parameters
            personality_vector: Personality state

        Returns:
            Personality-modulated prosody
        """
        pillars = personality_vector.get('pillars', {})

        # Ma (contemplative space) → slower speech, more pauses
        ma_weight = pillars.get('ma', 0.8)
        if ma_weight > 0.8:
            prosody['speech_rate'] *= 0.9  # Slower
            prosody['pause_probability'] += 0.1  # More pauses

        # Ahimsa (compassion) → warmer, gentler prosody
        ahimsa_weight = pillars.get('ahimsa', 0.9)
        if ahimsa_weight > 0.85:
            prosody['f0_mean'] -= 0.05  # Slightly lower (warmer)
            prosody['energy'] -= 0.05  # Softer (gentler)

        # Jeong (emotional connection) → more expressive intonation
        jeong_weight = pillars.get('jeong', 0.85)
        if jeong_weight > 0.8:
            prosody['f0_range'] += 0.1  # More pitch variation (expressive)
            prosody['intonation_curve'] = 'expressive_' + str(prosody.get('intonation_curve', 'neutral'))

        # Logos (rational grounding) → more measured, stable prosody
        logos_weight = pillars.get('logos', 0.7)
        if logos_weight > 0.75:
            prosody['f0_range'] -= 0.05  # Less variation (measured)
            prosody['speech_rate'] *= 0.95  # Slightly slower (deliberate)

        # Lagom (balance) → moderate all parameters
        lagom_weight = pillars.get('lagom', 0.75)
        if lagom_weight > 0.8:
            # Pull extreme values toward center
            for param in ['f0_mean', 'f0_range', 'energy', 'speech_rate']:
                if abs(prosody[param]) > 0.3:
                    prosody[param] *= 0.8  # Reduce extremes

        return prosody

    def _apply_reciprocal_influence(
        self,
        prosody: Dict[str, float],
        reciprocal_emotion: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Apply reciprocal emotion influence to prosody

        Args:
            prosody: Current prosody parameters
            reciprocal_emotion: Oviya's reciprocal emotion

        Returns:
            Reciprocally influenced prosody
        """
        ovi_emotion = reciprocal_emotion.get('ovi_emotion', 'neutral')
        ovi_intensity = reciprocal_emotion.get('intensity', 0.5)

        # Adjust prosody based on Oviya's emotional state
        if 'grounded_calm' in ovi_emotion:
            prosody['f0_mean'] -= 0.1 * ovi_intensity  # Lower pitch (steady)
            prosody['speech_rate'] *= (0.9 + 0.1 * ovi_intensity)  # Slower (measured)
        elif 'gentle_compassion' in ovi_emotion:
            prosody['energy'] -= 0.15 * ovi_intensity  # Softer (gentle)
            prosody['f0_range'] -= 0.1 * ovi_intensity  # Less variation (warm)
        elif 'protective_concern' in ovi_emotion:
            prosody['f0_mean'] -= 0.05 * ovi_intensity  # Slightly lower (serious)
            prosody['energy'] += 0.1 * ovi_intensity  # Slightly louder (emphasis)
        elif 'shared_joy' in ovi_emotion:
            prosody['f0_mean'] += 0.1 * ovi_intensity  # Higher pitch (enthusiastic)
            prosody['f0_range'] += 0.15 * ovi_intensity  # More variation (excited)

        return prosody

    def _adjust_for_text_characteristics(
        self,
        prosody: Dict[str, float],
        response_text: str
    ) -> Dict[str, float]:
        """
        Adjust prosody based on text characteristics

        Args:
            prosody: Current prosody parameters
            response_text: Text being synthesized

        Returns:
            Text-adjusted prosody
        """
        text_lower = response_text.lower()

        # Question detection → rising intonation
        if '?' in response_text:
            prosody['intonation_curve'] = 'questioning'
            prosody['f0_mean'] += 0.05

        # Exclamation detection → more energy
        if '!' in response_text:
            prosody['energy'] += 0.1
            prosody['f0_range'] += 0.1

        # Length-based adjustments
        word_count = len(response_text.split())
        if word_count > 50:  # Longer responses
            prosody['speech_rate'] *= 0.95  # Slightly slower
            prosody['pause_probability'] += 0.1  # More pauses
        elif word_count < 20:  # Short responses
            prosody['speech_rate'] *= 1.05  # Slightly faster

        # Emotional word detection
        emotional_words = ['feel', 'emotion', 'heart', 'pain', 'joy', 'sad', 'anxious', 'worried']
        emotional_count = sum(1 for word in emotional_words if word in text_lower)

        if emotional_count > 0:
            prosody['energy'] -= 0.05  # Slightly softer for emotional content
            prosody['speech_rate'] *= 0.95  # Slightly slower

        return prosody

    def _normalize_prosody_parameters(self, prosody: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize prosody parameters to safe ranges

        Args:
            prosody: Raw prosody parameters

        Returns:
            Normalized prosody parameters
        """
        # Define safe ranges
        ranges = {
            'f0_mean': (-0.5, 0.5),      # Semitones from neutral
            'f0_range': (-0.3, 0.5),     # Pitch variation range
            'energy': (-0.5, 0.5),       # Volume adjustment
            'speech_rate': (0.5, 1.5),   # Rate multiplier
            'pause_probability': (0.0, 0.6)  # Pause probability
        }

        for param, (min_val, max_val) in ranges.items():
            if param in prosody:
                prosody[param] = max(min_val, min(max_val, prosody[param]))

        return prosody

    def _generate_prosody_markers(
        self,
        prosody: Dict[str, float],
        response_text: str
    ) -> List[str]:
        """
        Generate prosody markers for CSM-1B integration

        Args:
            prosody: Computed prosody parameters
            response_text: Response text

        Returns:
            List of prosody markers
        """
        markers = []

        # F0 markers
        if abs(prosody['f0_mean']) > 0.1:
            direction = "higher" if prosody['f0_mean'] > 0 else "lower"
            markers.append(f"[PITCH:{direction}_{abs(prosody['f0_mean']):.2f}]")

        # Energy markers
        if abs(prosody['energy']) > 0.1:
            level = "louder" if prosody['energy'] > 0 else "softer"
            markers.append(f"[VOLUME:{level}_{abs(prosody['energy']):.2f}]")

        # Speech rate markers
        if abs(prosody['speech_rate'] - 1.0) > 0.1:
            speed = "faster" if prosody['speech_rate'] > 1.0 else "slower"
            markers.append(f"[SPEED:{speed}_{abs(prosody['speech_rate'] - 1.0):.2f}]")

        # Intonation markers
        if prosody['intonation_curve'] != 'neutral':
            markers.append(f"[CURVE:{prosody['intonation_curve']}]")

        # Pause probability markers (for potential silence insertion)
        if prosody['pause_probability'] > 0.3:
            markers.append(f"[PAUSE_PROB:{prosody['pause_probability']:.2f}]")

        return markers

    def _generate_prosody_explanation(
        self,
        prosody: Dict[str, float],
        emotion: str,
        intensity: float
    ) -> str:
        """
        Generate human-readable explanation of prosody choices

        Args:
            prosody: Final prosody parameters
            emotion: Primary emotion
            intensity: Emotion intensity

        Returns:
            Explanation string
        """
        explanation = f"For {emotion} (intensity: {intensity:.1f}), prosody adjusts: "

        adjustments = []
        if abs(prosody['f0_mean']) > 0.05:
            direction = "higher" if prosody['f0_mean'] > 0 else "lower"
            adjustments.append(f"pitch {direction}")

        if abs(prosody['energy']) > 0.05:
            level = "louder" if prosody['energy'] > 0 else "softer"
            adjustments.append(f"volume {level}")

        if abs(prosody['speech_rate'] - 1.0) > 0.05:
            speed = "faster" if prosody['speech_rate'] > 1.0 else "slower"
            adjustments.append(f"speech {speed}")

        if adjustments:
            explanation += ", ".join(adjustments)
        else:
            explanation += "moderate adjustments for natural expression"

        explanation += ". This creates authentic emotional voice modulation."

        return explanation


# Global prosody engine instance
_prosody_engine = None

def get_prosody_engine() -> ProsodyEngine:
    """Get or create global prosody engine"""
    global _prosody_engine
    if _prosody_engine is None:
        _prosody_engine = ProsodyEngine()
    return _prosody_engine


# Test function
def test_prosody_engine():
    """Test the prosody engine"""
    print("🧪 TESTING PROSODY ENGINE")
    print("=" * 50)

    engine = get_prosody_engine()

    # Test cases
    test_cases = [
        {
            'emotion': 'anxiety',
            'intensity': 0.8,
            'personality': {'pillars': {'ma': 0.8, 'ahimsa': 0.9, 'jeong': 0.85}},
            'reciprocal': {'ovi_emotion': 'grounded_calm', 'intensity': 0.8},
            'text': 'I hear your anxiety and I am here with steady calm.'
        },
        {
            'emotion': 'grief',
            'intensity': 0.9,
            'personality': {'pillars': {'ma': 0.95, 'ahimsa': 0.95, 'jeong': 1.0}},
            'reciprocal': {'ovi_emotion': 'deep_sadness', 'intensity': 0.6},
            'text': 'This grief you carry, I feel it too in this shared sadness.'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\\nTest Case {i}: {test_case['emotion']} (intensity: {test_case['intensity']})")

        result = engine.compute_prosody_parameters(
            emotion=test_case['emotion'],
            intensity=test_case['intensity'],
            personality_vector=test_case['personality'],
            response_text=test_case['text'],
            reciprocal_emotion=test_case['reciprocal']
        )

        params = result['parameters']
        markers = result['markers']

        print("Prosody Parameters:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"   Curve: {params['intonation_curve']}")

        print(f"\\nProsody Markers ({len(markers)}):")
        for marker in markers:
            print(f"   {marker}")

        print(f"\\nExplanation: {result['explanation']}")

    print("\\n✅ Prosody Engine test completed")


if __name__ == "__main__":
    test_prosody_engine()
