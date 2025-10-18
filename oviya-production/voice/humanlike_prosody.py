from __future__ import annotations

import random
from typing import Dict, Tuple


class HumanlikeProsodyEngine:
    """Create timing plans and light prosody hints for humanlike feel."""

    def __init__(self, enable_fillers: bool = True):
        self.enable_fillers = enable_fillers

    def enhance(self, text: str, emotion: str, ctx: Dict) -> Tuple[str, Dict]:
        """Return (possibly adjusted text, timing_plan dict)."""
        timing = {
            "pre_tts_delay_ms": self._pre_tts_delay_ms(emotion),
            "insert_breath": len(text) > 120,
            "use_filler": self.enable_fillers and random.random() < 0.05,
        }
        return text, timing

    def _pre_tts_delay_ms(self, emotion: str) -> int:
        if emotion in ("joyful_excited", "playful"):
            return 300
        if emotion in ("empathetic_sad", "calm_supportive"):
            return 500
        return 400


