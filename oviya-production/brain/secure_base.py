from __future__ import annotations

from typing import Dict, Optional


class SecureBaseSystem:
    """Detect whether user needs safe haven (comfort) or secure base (encouragement).

    Heuristics only; designed to be lightweight for realtime.
    """

    def __init__(self):
        """
        Initialize the SecureBaseSystem instance without setting any internal state.
        
        This constructor is a no-op and leaves the instance unconfigured.
        """
        pass

    def detect_user_state(self, prosody: Dict, text: str, history: Optional[Dict] = None) -> str:
        """
        Determine which type of interpersonal support the user likely needs based on text and prosodic energy.
        
        Analyzes the provided user text (case-insensitive) and an optional prosody dictionary to classify the user's state as one of three labels: "safe_haven_needed", "exploration_support_needed", or "neutral_presence". The decision is driven by presence of predefined keywords indicating distress or excitement and by the numerical `energy` value in `prosody` (defaults to 0.05 when missing).
        
        Parameters:
            prosody (Dict): Optional prosodic features; if present, the numeric value under the "energy" key is used. If `prosody` is None or missing "energy", energy defaults to 0.05.
            text (str): User-facing text to analyze.
            history (Optional[Dict]): Optional historical context (not used by this heuristic).
        
        Returns:
            str: One of:
                - "safe_haven_needed" if the text or low energy indicates distress,
                - "exploration_support_needed" if the text or high energy indicates excitement/curiosity,
                - "neutral_presence" otherwise.
        """
        t = (text or "").lower()
        energy = float(prosody.get("energy", 0.05)) if prosody else 0.05

        if any(k in t for k in ["afraid", "scared", "worried", "panic", "cry", "hurt", "accident", "lost", "grief", "heartbroken"]):
            return "safe_haven_needed"
        if energy < 0.02:
            return "safe_haven_needed"

        if any(k in t for k in ["excited", "can't wait", "promotion", "won", "launched", "shipped", "curious"]):
            return "exploration_support_needed"
        if energy > 0.08:
            return "exploration_support_needed"

        return "neutral_presence"

