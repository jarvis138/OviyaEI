from __future__ import annotations

from typing import Dict


class ConsistentPersonaMemory:
    """Ensure responses align with core values and historical tone.
    Lightweight stub: adjusts obvious mismatches.
    """

    def __init__(self):
        self.core_values = {
            "always_validates_first": True,
            "never_judges": True,
            "matches_energy_level": True,
            "remembers_context": True,
        }

    def ensure_consistency(self, text: str, user_id: str) -> str:
        t = text.strip()
        # Validation-first: if starts with advice, prepend a validation phrase
        lowers = t.lower()
        if any(lowers.startswith(x) for x in ["you should", "try ", "do this", "here's what", "first,"]):
            t = "That makes sense. " + t
        return t


