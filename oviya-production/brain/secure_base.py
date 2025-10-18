from __future__ import annotations

from typing import Dict, Optional


class SecureBaseSystem:
    """Detect whether user needs safe haven (comfort) or secure base (encouragement).

    Heuristics only; designed to be lightweight for realtime.
    """

    def __init__(self):
        pass

    def detect_user_state(self, prosody: Dict, text: str, history: Optional[Dict] = None) -> str:
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


