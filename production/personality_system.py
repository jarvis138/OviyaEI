#!/usr/bin/env python3
"""
Phase 5: Personality Computation and Strategic Silence

Implements Oviya's 5-pillar personality system and calculates strategic silence
based on emotional context and contemplative presence.
"""

import time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FivePillarPersonality:
    """
    Phase 5: 5-Pillar Personality System

    Computes personality vector based on 5 fundamental principles:
    - Ma (間): Contemplative space/intention
    - Ahimsa (अहिंसा): Non-harm/compassion
    - Jeong (정): Deep emotional connection
    - Logos (λόγος): Rational grounding
    - Lagom (lagom): Balance/moderation
    """

    def __init__(self):
        """Initialize 5-pillar personality system"""
        # Base personality weights (Oviya's core personality)
        self.base_personality = {
            'ma': 0.8,        # High contemplative presence
            'ahimsa': 0.9,    # Deep compassion and non-harm
            'jeong': 0.85,    # Strong emotional connection
            'logos': 0.7,     # Good rational grounding
            'lagom': 0.75     # Balanced approach
        }

        # Emotion-personality mapping
        self.emotion_personality_map = {
            'anxiety': {'ma': 0.9, 'ahimsa': 0.95, 'jeong': 0.9, 'logos': 0.8, 'lagom': 0.7},
            'sadness': {'ma': 0.95, 'ahimsa': 0.9, 'jeong': 0.95, 'logos': 0.6, 'lagom': 0.8},
            'anger': {'ma': 0.85, 'ahimsa': 0.8, 'jeong': 0.7, 'logos': 0.9, 'lagom': 0.6},
            'joy': {'ma': 0.6, 'ahimsa': 0.85, 'jeong': 0.8, 'logos': 0.7, 'lagom': 0.9},
            'calm': {'ma': 0.8, 'ahimsa': 0.9, 'jeong': 0.85, 'logos': 0.8, 'lagom': 0.95},
            'grief': {'ma': 1.0, 'ahimsa': 0.95, 'jeong': 1.0, 'logos': 0.5, 'lagom': 0.8},
            'fear': {'ma': 0.9, 'ahimsa': 0.9, 'jeong': 0.9, 'logos': 0.8, 'lagom': 0.7},
            'hope': {'ma': 0.7, 'ahimsa': 0.8, 'jeong': 0.8, 'logos': 0.9, 'lagom': 0.8},
            'confusion': {'ma': 0.8, 'ahimsa': 0.85, 'jeong': 0.75, 'logos': 0.95, 'lagom': 0.7},
            'gratitude': {'ma': 0.7, 'ahimsa': 0.9, 'jeong': 0.9, 'logos': 0.8, 'lagom': 0.85}
        }

        logger.info("🧘 5-Pillar Personality System initialized")

    def compute_personality_vector(
        self,
        detected_emotion: str,
        emotion_intensity: float,
        conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """
        Compute personality vector based on emotion and context

        Args:
            detected_emotion: Primary detected emotion
            emotion_intensity: Emotion intensity (0-1)
            conversation_context: Previous conversation turns

        Returns:
            Personality vector with 5 pillars
        """
        # Start with base personality
        personality = self.base_personality.copy()

        # Adjust based on detected emotion
        if detected_emotion in self.emotion_personality_map:
            emotion_weights = self.emotion_personality_map[detected_emotion]

            # Blend base personality with emotion-specific adjustments
            # Higher intensity = stronger emotion influence
            blend_factor = min(0.7, emotion_intensity * 0.8)  # Max 70% emotion influence

            for pillar in personality:
                emotion_weight = emotion_weights.get(pillar, personality[pillar])
                personality[pillar] = personality[pillar] * (1 - blend_factor) + emotion_weight * blend_factor

        # Adjust based on conversation context (recent emotional patterns)
        if conversation_context:
            context_adjustment = self._analyze_conversation_context(conversation_context)
            for pillar in personality:
                personality[pillar] = min(1.0, max(0.0, personality[pillar] + context_adjustment.get(pillar, 0)))

        # Normalize to ensure balanced personality
        personality = self._normalize_personality(personality)

        return {
            'pillars': personality,
            'dominant_pillar': max(personality, key=personality.get),
            'balance_score': self._calculate_balance_score(personality),
            'emotional_influence': emotion_intensity,
            'detected_emotion': detected_emotion
        }

    def _analyze_conversation_context(self, conversation_context: List[Dict]) -> Dict[str, float]:
        """
        Analyze recent conversation to adjust personality

        Args:
            conversation_context: Recent conversation turns

        Returns:
            Personality adjustments based on context
        """
        adjustments = {'ma': 0, 'ahimsa': 0, 'jeong': 0, 'logos': 0, 'lagom': 0}

        # Analyze last 3 turns
        recent_turns = conversation_context[-3:]

        for turn in recent_turns:
            emotion = turn.get('emotion', 'neutral')
            intensity = turn.get('intensity', 0.5)

            # Increase Ma for intense emotions (need more space)
            if intensity > 0.7:
                adjustments['ma'] += 0.1

            # Increase Jeong for repeated emotional patterns
            if emotion in ['sadness', 'grief', 'anxiety']:
                adjustments['jeong'] += 0.05

            # Increase Ahimsa for anger/frustration
            if emotion in ['anger', 'frustration']:
                adjustments['ahimsa'] += 0.08

        return adjustments

    def _normalize_personality(self, personality: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize personality to ensure balance and prevent extremes

        Args:
            personality: Raw personality scores

        Returns:
            Normalized personality scores
        """
        # Ensure minimum values (Oviya always has some of each pillar)
        min_values = {'ma': 0.5, 'ahimsa': 0.6, 'jeong': 0.5, 'logos': 0.4, 'lagom': 0.5}

        for pillar, min_val in min_values.items():
            personality[pillar] = max(min_val, personality[pillar])

        # Ensure maximum values (prevent extremes)
        max_values = {'ma': 1.0, 'ahimsa': 1.0, 'jeong': 1.0, 'logos': 0.9, 'lagom': 0.9}

        for pillar, max_val in max_values.items():
            personality[pillar] = min(max_val, personality[pillar])

        return personality

    def _calculate_balance_score(self, personality: Dict[str, float]) -> float:
        """
        Calculate balance score (how well-balanced the personality is)

        Args:
            personality: Personality pillar scores

        Returns:
            Balance score (0-1, higher = more balanced)
        """
        values = list(personality.values())
        mean = sum(values) / len(values)

        # Calculate variance (lower variance = more balanced)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        balance_score = 1.0 - min(1.0, variance * 4)  # Scale variance to 0-1

        return balance_score


class StrategicSilenceCalculator:
    """
    Strategic Silence Calculation

    Computes optimal silence duration based on Ma (contemplative space),
    emotion intensity, and therapeutic context.
    """

    def __init__(self):
        """Initialize strategic silence calculator"""
        # Base silence durations for different emotional contexts
        self.base_silence_durations = {
            'grief': {'min': 2.0, 'max': 3.5, 'ma_multiplier': 1.5},
            'sadness': {'min': 1.5, 'max': 3.0, 'ma_multiplier': 1.3},
            'anxiety': {'min': 1.0, 'max': 2.5, 'ma_multiplier': 1.2},
            'anger': {'min': 0.8, 'max': 2.0, 'ma_multiplier': 1.1},
            'fear': {'min': 1.2, 'max': 2.8, 'ma_multiplier': 1.4},
            'confusion': {'min': 1.0, 'max': 2.2, 'ma_multiplier': 1.1},
            'joy': {'min': 0.3, 'max': 1.0, 'ma_multiplier': 0.7},
            'calm': {'min': 0.5, 'max': 1.5, 'ma_multiplier': 1.0},
            'hope': {'min': 0.4, 'max': 1.2, 'ma_multiplier': 0.8},
            'gratitude': {'min': 0.6, 'max': 1.8, 'ma_multiplier': 0.9}
        }

        logger.info("🤫 Strategic Silence Calculator initialized")

    def calculate_silence_duration(
        self,
        emotion: str,
        intensity: float,
        ma_weight: float,
        jeong_weight: float,
        conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """
        Calculate optimal silence duration

        Args:
            emotion: Detected primary emotion
            intensity: Emotion intensity (0-1)
            ma_weight: Ma (contemplative space) personality weight
            jeong_weight: Jeong (emotional connection) personality weight
            conversation_context: Recent conversation context

        Returns:
            Silence duration and reasoning
        """
        # Get base duration for emotion
        base_config = self.base_silence_durations.get(emotion, self.base_silence_durations['calm'])

        # Calculate base duration
        base_duration = base_config['min'] + (base_config['max'] - base_config['min']) * intensity

        # Apply Ma multiplier (higher Ma = more silence)
        ma_adjusted_duration = base_duration * (0.5 + ma_weight * base_config['ma_multiplier'])

        # Apply Jeong adjustment (higher Jeong = slightly more silence for connection)
        jeong_adjustment = jeong_weight * 0.3
        final_duration = ma_adjusted_duration + jeong_adjustment

        # Apply conversation context adjustments
        context_adjustment = self._analyze_context_for_silence(conversation_context, emotion, intensity)
        final_duration += context_adjustment

        # Ensure reasonable bounds
        final_duration = max(0.2, min(4.0, final_duration))

        return {
            'silence_duration_seconds': final_duration,
            'emotion': emotion,
            'intensity': intensity,
            'ma_weight': ma_weight,
            'jeong_weight': jeong_weight,
            'base_duration': base_duration,
            'ma_adjusted': ma_adjusted_duration,
            'context_adjustment': context_adjustment,
            'reasoning': self._generate_silence_reasoning(emotion, intensity, ma_weight, final_duration)
        }

    def _analyze_context_for_silence(self, conversation_context: Optional[List[Dict]], emotion: str, intensity: float) -> float:
        """
        Analyze conversation context for silence adjustments

        Args:
            conversation_context: Recent conversation turns
            emotion: Current emotion
            intensity: Current intensity

        Returns:
            Silence duration adjustment (+/- seconds)
        """
        adjustment = 0.0

        if not conversation_context:
            return adjustment

        # Analyze recent turns
        recent_turns = conversation_context[-3:]

        # Increase silence after intense emotional disclosures
        intense_emotions = ['grief', 'sadness', 'anxiety', 'fear']
        intense_turns = sum(1 for turn in recent_turns
                           if turn.get('emotion') in intense_emotions
                           and turn.get('intensity', 0) > 0.7)

        if intense_turns >= 2:
            adjustment += 0.5  # Extra space after multiple intense disclosures

        # Decrease silence for urgent emotions (anger)
        if emotion == 'anger' and intensity > 0.8:
            adjustment -= 0.3

        # Increase silence for repeated similar emotions (deep processing needed)
        current_emotion_count = sum(1 for turn in recent_turns
                                  if turn.get('emotion') == emotion)

        if current_emotion_count >= 2:
            adjustment += 0.4

        return adjustment

    def _generate_silence_reasoning(self, emotion: str, intensity: float, ma_weight: float, duration: float) -> str:
        """
        Generate human-readable reasoning for silence duration

        Args:
            emotion: Detected emotion
            intensity: Emotion intensity
            ma_weight: Ma personality weight
            duration: Calculated silence duration

        Returns:
            Reasoning string
        """
        intensity_desc = "high" if intensity > 0.7 else "moderate" if intensity > 0.4 else "low"
        ma_desc = "strong" if ma_weight > 0.8 else "moderate" if ma_weight > 0.6 else "balanced"

        reasoning = f"Calculated {duration:.1f}s silence for {intensity_desc} intensity {emotion} "
        reasoning += f"with {ma_desc} contemplative presence (Ma={ma_weight:.2f}). "
        reasoning += "This allows space for emotional processing and authentic connection."

        return reasoning


class CompletePersonalitySilenceSystem:
    """
    Complete Phase 5: Personality Computation + Strategic Silence

    Orchestrates the 5-pillar personality system and strategic silence calculation.
    """

    def __init__(self):
        """Initialize complete personality and silence system"""
        self.personality_system = FivePillarPersonality()
        self.silence_calculator = StrategicSilenceCalculator()

        logger.info("🧘 Complete Personality & Silence System initialized")

    def process_emotion_and_context(
        self,
        detected_emotion: str,
        emotion_intensity: float,
        conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """
        Complete Phase 5 processing: personality + strategic silence

        Args:
            detected_emotion: Primary detected emotion
            emotion_intensity: Emotion intensity (0-1)
            conversation_context: Recent conversation context

        Returns:
            Complete personality and silence analysis
        """
        # Phase 5.1: Compute personality vector
        personality_vector = self.personality_system.compute_personality_vector(
            detected_emotion, emotion_intensity, conversation_context
        )

        # Phase 5.2: Calculate strategic silence
        silence_analysis = self.silence_calculator.calculate_silence_duration(
            detected_emotion,
            emotion_intensity,
            personality_vector['pillars']['ma'],
            personality_vector['pillars']['jeong'],
            conversation_context
        )

        return {
            'personality_vector': personality_vector,
            'strategic_silence': silence_analysis,
            'emotion': detected_emotion,
            'intensity': emotion_intensity,
            'processing_timestamp': time.time()
        }


# Global personality system instance
_personality_system = None

def get_personality_system() -> CompletePersonalitySilenceSystem:
    """Get or create global personality and silence system"""
    global _personality_system
    if _personality_system is None:
        _personality_system = CompletePersonalitySilenceSystem()
    return _personality_system


# Test function
def test_personality_silence_system():
    """Test the personality and strategic silence system"""
    print("🧪 TESTING PERSONALITY & SILENCE SYSTEM")
    print("=" * 60)

    system = get_personality_system()

    # Test cases with different emotions and contexts
    test_cases = [
        {
            'emotion': 'grief',
            'intensity': 0.8,
            'context': [
                {'emotion': 'sadness', 'intensity': 0.7},
                {'emotion': 'grief', 'intensity': 0.9}
            ]
        },
        {
            'emotion': 'anxiety',
            'intensity': 0.6,
            'context': None
        },
        {
            'emotion': 'joy',
            'intensity': 0.5,
            'context': [
                {'emotion': 'calm', 'intensity': 0.3}
            ]
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\\nTest Case {i}: {test_case['emotion']} (intensity: {test_case['intensity']})")

        result = system.process_emotion_and_context(
            test_case['emotion'],
            test_case['intensity'],
            test_case['context']
        )

        # Display personality vector
        personality = result['personality_vector']
        pillars = personality['pillars']
        print("5-Pillar Personality:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"   Dominant: {personality['dominant_pillar']}")
        print(".2f")

        # Display strategic silence
        silence = result['strategic_silence']
        print("\nStrategic Silence:")
        print(".1f")
        print(f"   Reasoning: {silence['reasoning']}")

    print("\\n✅ Personality & Silence System test completed")


if __name__ == "__main__":
    test_personality_silence_system()
