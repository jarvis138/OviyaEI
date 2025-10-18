from typing import Dict, Tuple


class ScandinavianEmotionalWisdom:
    def lagom(self, intensity: float, text: str) -> Tuple[str, str]:
        if intensity > 0.85:
            t = "This is a lot to hold. Let's take it one piece at a time.\n\n" + (text or "")
            return t, "slower_calmer"
        if intensity < 0.15:
            t = "I'm here with you, even in this heaviness.\n\n" + (text or "")
            return t, "warmer_gentler"
        return text or "", "maintain_balance"


