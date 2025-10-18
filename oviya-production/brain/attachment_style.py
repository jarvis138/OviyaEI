from __future__ import annotations

from typing import Dict


class AttachmentStyleDetector:
    """Heuristic detector for high-level attachment style over time.
    Use aggregate signals (sessions per week, reassurance asks, avoidance).
    """

    def detect(self, user_history: Dict) -> str:
        """
        Infer a user's high-level attachment style from aggregated session and interaction signals.
        
        Parameters:
            user_history (Dict): Mapping of user metrics. Recognized keys:
                - "sessions_per_week" (int-like): number of sessions per week (default 3).
                - "reassurance_prompts" (int-like): count of reassurance prompts (default 0).
                - "avoidance_ratio" (float-like): avoidance signal in range [0,1] (default 0.2).
        
        Returns:
            str: One of:
                - "anxious_preoccupied" if sessions_per_week > 7 and reassurance_prompts >= 5.
                - "dismissive_avoidant" if avoidance_ratio > 0.6.
                - "fearful_avoidant" if 0.3 < avoidance_ratio < 0.6 and reassurance_prompts >= 3.
                - "secure" otherwise.
        """
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
        """
        Selects an interaction strategy dictionary tailored to the given attachment style.
        
        Parameters:
            style (str): Attachment style identifier. Accepted values: 'anxious_preoccupied', 'dismissive_avoidant', 'fearful_avoidant', 'secure'. Unknown values fall back to 'secure'.
        
        Returns:
            Dict: A strategy dictionary with keys:
                - check_in_frequency: how proactively to initiate contact
                - reassurance_level: explicitness of reassurance
                - availability_signal: how availability is signaled
                - validation_intensity: level of emotional validation
        """
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

