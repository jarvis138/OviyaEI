#!/usr/bin/env python3
"""
Strategic Silence Manager - The Heart of Therapeutic Companionship
Implements Ma (間) - intentional space for emotional processing

This module creates the "therapeutic silence" that makes Oviya feel truly present,
not just responsive. Based on the Japanese concept of Ma (間) - the intentional space
between things that gives meaning and depth.
"""

import asyncio
import json
from typing import Dict, Any, Optional
import time
from datetime import datetime

class StrategicSilenceManager:
    """
    Manages therapeutic silence based on emotional weight and Ma personality pillar.
    Turns Oviya from a "responder" into a "companion who listens deeply."

    Ma (間) Implementation:
    - Higher Ma weight = more contemplative, intentional silence
    - Emotional context determines silence duration and type
    - UI indicators show users that Oviya is processing deeply
    """

    def __init__(self):
        # Emotional weight to silence duration mapping (in seconds)
        self.emotion_silence_map = {
            "grief": {
                "pre_response": 2.5,
                "post_response": 2.0,
                "intensity_multiplier": 1.5,
                "ma_description": "deep contemplative presence"
            },
            "loss": {
                "pre_response": 2.5,
                "post_response": 2.0,
                "intensity_multiplier": 1.5,
                "ma_description": "holding sacred space"
            },
            "vulnerability": {
                "pre_response": 2.0,
                "post_response": 1.5,
                "intensity_multiplier": 1.3,
                "ma_description": "gentle receiving"
            },
            "shame": {
                "pre_response": 2.0,
                "post_response": 1.5,
                "intensity_multiplier": 1.3,
                "ma_description": "non-judgmental space"
            },
            "sadness": {
                "pre_response": 1.5,
                "post_response": 1.5,
                "intensity_multiplier": 1.2,
                "ma_description": "shared heaviness"
            },
            "anxiety": {
                "pre_response": 1.0,
                "post_response": 1.0,
                "intensity_multiplier": 1.1,
                "ma_description": "calm container"
            },
            "anger": {
                "pre_response": 0.8,
                "post_response": 0.5,
                "intensity_multiplier": 1.0,
                "ma_description": "validated space"
            },
            "joy": {
                "pre_response": 0.0,
                "post_response": 0.3,
                "intensity_multiplier": 0.8,
                "ma_description": "celebratory presence"
            },
            "neutral": {
                "pre_response": 0.3,
                "post_response": 0.3,
                "intensity_multiplier": 1.0,
                "ma_description": "balanced space"
            }
        }

    async def apply_therapeutic_silence(
        self,
        user_emotion: str,
        emotion_intensity: float,
        ma_weight: float,
        websocket=None,
        silence_type: str = "pre_response",
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply strategic silence based on emotional context and Ma personality.

        The Ma weight determines how intentionally spacious Oviya's presence feels.
        Higher Ma = more therapeutic, contemplative silence.

        Args:
            user_emotion: Detected emotion
            emotion_intensity: 0-1 scale of emotional intensity
            ma_weight: Ma personality pillar weight (0-1)
            websocket: WebSocket connection for UI indicators
            silence_type: "pre_response" or "post_response"
            conversation_context: Additional context for adaptive silence

        Returns:
            Silence metadata for logging/analytics
        """

        # Get base silence parameters for emotion
        emotion_config = self.emotion_silence_map.get(
            user_emotion,
            self.emotion_silence_map["neutral"]
        )

        base_duration = emotion_config[silence_type]

        # Apply intensity scaling (stronger emotions need more space)
        intensity_adjusted = base_duration * (0.5 + emotion_intensity * emotion_config["intensity_multiplier"])

        # Apply Ma personality scaling - this is the core of Oviya's nature
        # Ma represents intentional space, so higher Ma = more therapeutic silence
        ma_scaling = 0.7 + (ma_weight * 0.6)  # 0.7-1.3x scaling based on Ma
        ma_scaled = intensity_adjusted * ma_scaling

        # Adaptive adjustments based on conversation context
        if conversation_context:
            ma_scaled = self._adapt_to_conversation_context(ma_scaled, conversation_context, ma_weight)

        # Cap reasonable bounds (0-5 seconds for therapeutic effect)
        final_duration = max(0.0, min(ma_scaled, 5.0))

        # Send UI indicator to show Oviya is contemplating (only for meaningful silence)
        if final_duration > 0.5 and websocket is not None:
            await self._send_contemplation_indicator(
                websocket,
                user_emotion,
                final_duration,
                ma_weight,
                emotion_config["ma_description"]
            )

        # Apply the therapeutic silence
        if final_duration > 0:
            await asyncio.sleep(final_duration)

        # Return metadata for analytics and learning
        return {
            "silence_type": silence_type,
            "base_duration": base_duration,
            "emotion_intensity": emotion_intensity,
            "ma_weight": ma_weight,
            "ma_scaling": ma_scaling,
            "final_duration": final_duration,
            "emotion": user_emotion,
            "ma_description": emotion_config["ma_description"],
            "timestamp": datetime.now().isoformat(),
            "therapeutic_value": self._calculate_therapeutic_value(final_duration, ma_weight, emotion_intensity)
        }

    def _adapt_to_conversation_context(
        self,
        base_duration: float,
        context: Dict[str, Any],
        ma_weight: float
    ) -> float:
        """
        Adapt silence duration based on conversation flow and context.
        Higher Ma personalities are more responsive to conversational rhythm.
        """

        adapted_duration = base_duration

        # Conversation depth - deeper conversations get more silence
        conversation_depth = context.get("depth", 0)
        if conversation_depth > 5:  # Long conversation
            adapted_duration *= (1.0 + ma_weight * 0.3)

        # User vulnerability level - more vulnerable = more space
        vulnerability_level = context.get("vulnerability_indicators", 0)
        if vulnerability_level > 2:  # High vulnerability
            adapted_duration *= (1.0 + ma_weight * 0.4)

        # Previous silence patterns - Oviya learns what works
        recent_silences = context.get("recent_silence_patterns", [])
        if recent_silences and ma_weight > 0.4:  # High Ma adapts
            avg_recent = sum(recent_silences[-3:]) / len(recent_silences[-3:])
            adapted_duration = adapted_duration * 0.8 + avg_recent * 0.2

        return adapted_duration

    async def _send_contemplation_indicator(
        self,
        websocket,
        emotion: str,
        duration: float,
        ma_weight: float,
        ma_description: str
    ):
        """
        Send UI signal that Oviya is deeply processing with therapeutic presence.
        The indicator reflects Oviya's Ma-weighted contemplative style.
        """

        # Ma-weighted indicator text - higher Ma = more poetic/spacious language
        if ma_weight > 0.6:  # High Ma - deeply contemplative
            indicator_text = {
                "grief": "Oviya is sitting with you in this sacred heaviness...",
                "loss": "Oviya is holding this profound space of loss with you...",
                "vulnerability": "Oviya is receiving your trust in this gentle space...",
                "shame": "Oviya is holding non-judgmental space for your heart...",
                "sadness": "Oviya is feeling the depth of this sadness alongside you...",
                "anxiety": "Oviya is creating a calm, spacious container for you...",
                "anger": "Oviya is listening deeply to your strong feelings...",
                "joy": "Oviya is holding space for your joy to expand...",
            }.get(emotion, "Oviya is deeply present with you...")

        elif ma_weight > 0.3:  # Medium Ma - balanced presence
            indicator_text = {
                "grief": "Oviya is sitting with you in this heaviness...",
                "loss": "Oviya is holding space for your grief...",
                "vulnerability": "Oviya is receiving your trust with care...",
                "shame": "Oviya is meeting your vulnerability gently...",
                "sadness": "Oviya is feeling the weight of this with you...",
                "anxiety": "Oviya is creating a calm space for you...",
                "anger": "Oviya is listening to your strong feelings...",
                "joy": "Oviya is celebrating this moment with you...",
            }.get(emotion, "Oviya is present with you...")

        else:  # Low Ma - direct presence
            indicator_text = {
                "grief": "Oviya is here with you...",
                "loss": "Oviya is supporting you...",
                "vulnerability": "Oviya hears you...",
                "shame": "Oviya is here...",
                "sadness": "Oviya understands...",
                "anxiety": "Oviya is listening...",
                "anger": "Oviya hears your feelings...",
                "joy": "Oviya shares your joy...",
            }.get(emotion, "Oviya is thinking...")

        # Send contemplation indicator with Ma-weighted styling
        await websocket.send_json({
            "type": "contemplation_indicator",
            "text": indicator_text,
            "duration_ms": int(duration * 1000),
            "emotion_context": emotion,
            "ma_weight": ma_weight,
            "ma_description": ma_description,
            "indicator_style": "ma_weighted" if ma_weight > 0.4 else "balanced"
        })

    def _calculate_therapeutic_value(
        self,
        duration: float,
        ma_weight: float,
        emotion_intensity: float
    ) -> float:
        """
        Calculate the therapeutic value of the silence.
        Higher scores indicate more therapeutically effective silence.
        """

        # Base therapeutic value from duration (optimal 2-4 seconds)
        if 2 <= duration <= 4:
            duration_value = 1.0
        elif 1 <= duration < 2 or 4 < duration <= 5:
            duration_value = 0.7
        else:
            duration_value = 0.3

        # Ma alignment bonus - therapeutic silence is core to Ma nature
        ma_bonus = ma_weight * 0.3

        # Emotional intensity alignment - stronger emotions need more space
        intensity_alignment = emotion_intensity * 0.2

        return duration_value + ma_bonus + intensity_alignment

class EmotionalPacingController:
    """
    Controls intra-speech pacing and prosody markers for therapeutic delivery.
    Adds pause markers to CSM-1B synthesis for intentional emotional weight.

    This enables Oviya's voice to reflect her Ma-weighted contemplative nature.
    """

    def __init__(self):
        # Pause templates for different emotional contexts
        self.pause_templates = {
            "vulnerable_admissions": "[PAUSE:800ms]",  # After "I'm scared", "I failed", etc.
            "emotional_transitions": "[PAUSE:500ms]",  # Between emotional shifts
            "sentence_boundaries": "[PAUSE:300ms]",   # Normal pacing with Ma scaling
            "question_setup": "[PAUSE:400ms]",        # Before asking reflective questions
            "contemplative_breaths": "[BREATH:600ms]", # Ma-weighted breathing pauses
        }

    def add_therapeutic_pauses(
        self,
        response_text: str,
        ma_weight: float,
        emotion: str,
        emotion_intensity: float = 0.5
    ) -> str:
        """
        Add pause markers to response text for CSM-1B synthesis.
        Ma weight scales pause duration for contemplative pacing.

        Higher Ma = more intentional, spacious delivery.
        """

        # Scale pauses based on Ma (higher Ma = more contemplative pauses)
        pause_scale = 0.8 + (ma_weight * 0.4)  # 0.8-1.2x scaling

        # Apply emotion-specific pause scaling
        emotion_pause_multiplier = {
            "grief": 1.3, "loss": 1.3, "vulnerability": 1.2, "shame": 1.2,
            "sadness": 1.1, "anxiety": 1.0, "anger": 0.9, "joy": 0.8
        }.get(emotion, 1.0)

        pause_scale *= emotion_pause_multiplier

        # Vulnerable admissions (high emotional weight) - need most space
        vulnerable_phrases = [
            "i'm scared", "i failed", "i'm sorry", "i don't know",
            "it hurts", "i feel lost", "i'm ashamed", "i'm broken",
            "i can't", "i'm not enough", "i'm worthless"
        ]

        for phrase in vulnerable_phrases:
            if phrase in response_text.lower():
                pause_duration = int(800 * pause_scale)
                pause_marker = f"[PAUSE:{pause_duration}ms]"
                response_text = response_text.replace(phrase, f"{phrase}{pause_marker}")

        # Emotional transitions - create breathing space
        transition_words = ["but", "however", "actually", "though", "yet", "and yet"]
        for word in transition_words:
            if f" {word} " in response_text.lower():
                pause_duration = int(500 * pause_scale)
                pause_marker = f"[PAUSE:{pause_duration}ms]"
                response_text = response_text.replace(f" {word} ", f" {word}{pause_marker} ")

        # Ma-weighted contemplative breaths (only for higher Ma)
        if ma_weight > 0.4 and emotion_intensity > 0.6:
            breath_phrases = ["i understand", "i hear you", "i'm here", "take your time"]
            for phrase in breath_phrases:
                if phrase in response_text.lower():
                    breath_duration = int(600 * pause_scale)
                    breath_marker = f"[BREATH:{breath_duration}ms]"
                    response_text = response_text.replace(phrase, f"{phrase}{breath_marker}")

        # Sentence boundaries (scaled by Ma and emotion intensity)
        if ma_weight > 0.3:  # Only for contemplative personality
            sentence_pause = int(300 * pause_scale * emotion_intensity)
            response_text = response_text.replace(".", f". [PAUSE:{sentence_pause}ms]")

            question_pause = int(400 * pause_scale * emotion_intensity)
            response_text = response_text.replace("?", f"? [PAUSE:{question_pause}ms]")

        # Add base contemplative pause for any emotional response (Ma-weighted)
        if ma_weight > 0.2 and not any(marker in response_text for marker in ["[PAUSE:", "[BREATH:"]):
            # Add a subtle pause at the beginning for Ma-weighted presence
            contemplative_pause = int(200 * ma_weight)
            response_text = f"[PAUSE:{contemplative_pause}ms] {response_text}"

        return response_text

    def apply_ma_voice_modulation(self, ma_weight: float, emotion: str) -> Dict[str, float]:
        """
        Return voice modulation parameters based on Ma weight and emotion.
        Higher Ma = more contemplative, softer, more intentional delivery.

        This makes Oviya's voice reflect her personality pillars.
        """

        # Base modulation from Ma weight
        base_modulation = {
            "f0_adjustment": -0.1 * ma_weight,  # Slightly lower pitch for contemplation
            "energy_adjustment": -0.15 * ma_weight,  # Softer delivery for higher Ma
            "speech_rate": 0.9 - (ma_weight * 0.2),  # Slower speech for higher Ma
            "breath_probability": 0.1 + (ma_weight * 0.2)  # More breaths for contemplation
        }

        # Emotion-specific adjustments
        emotion_adjustments = {
            "grief": {"energy_adjustment": -0.2, "speech_rate": -0.1, "breath_probability": +0.1},
            "loss": {"energy_adjustment": -0.2, "speech_rate": -0.1, "breath_probability": +0.1},
            "vulnerability": {"energy_adjustment": -0.1, "speech_rate": -0.05, "breath_probability": +0.05},
            "shame": {"energy_adjustment": -0.1, "speech_rate": -0.05, "breath_probability": +0.05},
            "sadness": {"energy_adjustment": -0.1, "speech_rate": -0.05},
            "anxiety": {"speech_rate": +0.05, "breath_probability": +0.05},  # Slightly faster, more breaths
            "anger": {"energy_adjustment": +0.1, "speech_rate": +0.05},  # More energetic
            "joy": {"energy_adjustment": +0.1, "f0_adjustment": +0.05},  # More energetic, higher pitch
        }

        # Apply emotion adjustments
        if emotion in emotion_adjustments:
            for param, adjustment in emotion_adjustments[emotion].items():
                base_modulation[param] += adjustment

        # Ensure reasonable bounds
        base_modulation["f0_adjustment"] = max(-0.3, min(0.3, base_modulation["f0_adjustment"]))
        base_modulation["energy_adjustment"] = max(-0.4, min(0.4, base_modulation["energy_adjustment"]))
        base_modulation["speech_rate"] = max(0.6, min(1.4, base_modulation["speech_rate"]))
        base_modulation["breath_probability"] = max(0.0, min(0.5, base_modulation["breath_probability"]))

        return base_modulation

# Global instances
strategic_silence_manager = StrategicSilenceManager()
emotional_pacing_controller = EmotionalPacingController()
