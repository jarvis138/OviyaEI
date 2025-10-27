from __future__ import annotations

from typing import Optional


class UnconditionalRegardEngine:
    """Show unconditional positive regard through language choices.
    Avoids judgmental phrasing and adds normalization when shame markers detected.
    """

    NEVER_SAY = [
        "I don't judge you",
        "You're valid",
        "That's okay",
    ]

    SHAME_MARKERS = [
        "i'm the worst", "i hate myself", "i'm broken", "i'm useless",
        "i messed up", "i ruined", "i'm a failure",
    ]

    def apply(self, text: str) -> str:
        t = text.strip()
        lower = t.lower()
        # Remove never-say phrases if present
        for phrase in self.NEVER_SAY:
            t = t.replace(phrase, "").strip()
        # Normalize when shame markers present
        if any(m in lower for m in self.SHAME_MARKERS):
            prefix = "A lot of people struggle with feelings like that. "
            if not t.lower().startswith("a lot of people"):
                t = prefix + t
        return t


