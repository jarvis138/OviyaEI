from typing import Dict


class JapaneseEmotionalWisdom:
    def __init__(self, min_ms: int = 800, max_ms: int = 2000):
        """
        Initialize the JapaneseEmotionalWisdom instance with configurable minimum and maximum pause durations.
        
        Parameters:
            min_ms (int): Minimum pause duration in milliseconds (default 800).
            max_ms (int): Maximum pause duration in milliseconds (default 2000).
        """
        self.min_ms = min_ms
        self.max_ms = max_ms

    def ma_timing(self, emotional_weight: float) -> Dict:
        """
        Determine pause timing and quality for a spoken "ma" based on emotional weight.
        
        Parameters:
            emotional_weight (float): Emotional intensity on a 0–1 scale; higher values indicate stronger emotion.
        
        Returns:
            dict: Mapping with keys:
                - "pause_before_ms" (int): Pause duration in milliseconds.
                - "pause_quality" (str): One of "contemplative_ma", "respectful_ma", or "natural_ma" describing the pause character.
        """
        if emotional_weight > 0.7:
            return {"pause_before_ms": self.max_ms, "pause_quality": "contemplative_ma"}
        if emotional_weight > 0.4:
            return {"pause_before_ms": 1200, "pause_quality": "respectful_ma"}
        return {"pause_before_ms": self.min_ms, "pause_quality": "natural_ma"}

