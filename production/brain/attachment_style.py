from __future__ import annotations

from typing import Dict


class AttachmentStyleDetector:
    """Heuristic detector for high-level attachment style over time.
    Use aggregate signals (sessions per week, reassurance asks, avoidance).
    """

    def detect(self, user_history: Dict) -> str:
        sessions_per_week = int(user_history.get("sessions_per_week", 3))
        reassurance = int(user_history.get("reassurance_prompts", 0))
        avoidance = float(user_history.get("avoidance_ratio", 0.2))

        if sessions_per_week > 7 and reassurance >= 5:
            return "anxious_preoccupied"
        if avoidance > 0.6:
            return "dismissive_avoidant"
        if 0.3 < avoidance < 0.6 and reassurance >= 3:
            return "fearful_avoidant"
        return "secure"

    def adapt_interaction_style(self, style: str) -> Dict:
        strategies = {
            'anxious_preoccupied': {
                'check_in_frequency': 'proactive',
                'reassurance_level': 'explicit',
                'availability_signal': 'always',
                'validation_intensity': 'high'
            },
            'dismissive_avoidant': {
                'check_in_frequency': 'responsive',
                'reassurance_level': 'subtle',
                'availability_signal': 'stated_once',
                'validation_intensity': 'moderate'
            },
            'fearful_avoidant': {
                'check_in_frequency': 'gentle',
                'reassurance_level': 'consistent',
                'availability_signal': 'patient',
                'validation_intensity': 'steady'
            },
            'secure': {
                'check_in_frequency': 'balanced',
                'reassurance_level': 'natural',
                'availability_signal': 'implicit',
                'validation_intensity': 'appropriate'
            }
        }
        return strategies.get(style, strategies['secure'])




