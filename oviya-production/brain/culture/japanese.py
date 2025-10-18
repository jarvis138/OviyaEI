from typing import Dict


class JapaneseEmotionalWisdom:
    def __init__(self, min_ms: int = 800, max_ms: int = 2000):
        self.min_ms = min_ms
        self.max_ms = max_ms

    def ma_timing(self, emotional_weight: float) -> Dict:
        if emotional_weight > 0.7:
            return {"pause_before_ms": self.max_ms, "pause_quality": "contemplative_ma"}
        if emotional_weight > 0.4:
            return {"pause_before_ms": 1200, "pause_quality": "respectful_ma"}
        return {"pause_before_ms": self.min_ms, "pause_quality": "natural_ma"}


