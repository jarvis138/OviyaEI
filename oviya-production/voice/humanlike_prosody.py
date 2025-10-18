from __future__ import annotations

import random
from typing import Dict, Tuple


class HumanlikeProsodyEngine:
    """Create timing plans and light prosody hints for humanlike feel."""

    def __init__(self, enable_fillers: bool = True):
        """
        Initialize the HumanlikeProsodyEngine.
        
        Parameters:
            enable_fillers (bool): If True, allow the engine to include occasional filler hints in timing plans (default True).
        """
        self.enable_fillers = enable_fillers

    def enhance(self, text: str, emotion: str, ctx: Dict) -> Tuple[str, Dict]:
        """
        Produce a timing plan with light prosody hints for the given text.
        
        Parameters:
            text (str): The input text to be spoken.
            emotion (str): Emotion label used to determine pre-speech delay.
            ctx (Dict): Additional context (accepted but not used by this implementation).
        
        Returns:
            tuple: A pair (text, timing_plan) where:
                text (str): The original, unchanged input text.
                timing_plan (dict): Timing and prosody hints with keys:
                    - pre_tts_delay_ms (int): Pre-speech delay in milliseconds determined from `emotion`.
                    - insert_breath (bool): `True` if the text length is greater than 120 characters, `False` otherwise.
                    - use_filler (bool): `True` if fillers are enabled and selected by a ~5% random chance, `False` otherwise.
        """
        timing = {
            "pre_tts_delay_ms": self._pre_tts_delay_ms(emotion),
            "insert_breath": len(text) > 120,
            "use_filler": self.enable_fillers and random.random() < 0.05,
        }
        return text, timing

    def _pre_tts_delay_ms(self, emotion: str) -> int:
        """
        Selects a pre-Text-To-Speech delay (in milliseconds) appropriate for the given emotion.
        
        Parameters:
            emotion (str): Emotion label influencing delay. Recognized values:
                - "joyful_excited", "playful": shorter delay
                - "empathetic_sad", "calm_supportive": longer delay
                - any other value: default delay
        
        Returns:
            int: Delay in milliseconds (300 for joyful/playful, 500 for empathetic/calm, 400 otherwise).
        """
        if emotion in ("joyful_excited", "playful"):
            return 300
        if emotion in ("empathetic_sad", "calm_supportive"):
            return 500
        return 400

