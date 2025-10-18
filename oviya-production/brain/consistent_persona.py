from __future__ import annotations

from typing import Dict


class ConsistentPersonaMemory:
    """Ensure responses align with core values and historical tone.
    Lightweight stub: adjusts obvious mismatches.
    """

    def __init__(self):
        """
        Initialize the ConsistentPersonaMemory instance.
        
        Creates the `core_values` dictionary with boolean flags that guide lightweight persona adjustments:
        - `always_validates_first`: whether responses should begin by validating the user's input.
        - `never_judges`: whether responses should avoid judging the user.
        - `matches_energy_level`: whether responses should adapt to the user's tone/energy.
        - `remembers_context`: whether the persona should preserve conversational context.
        """
        self.core_values = {
            "always_validates_first": True,
            "never_judges": True,
            "matches_energy_level": True,
            "remembers_context": True,
        }

    def ensure_consistency(self, text: str, user_id: str) -> str:
        """
        Ensure response tone aligns with the persona by optionally prepending a brief validation phrase.
        
        If the stripped input begins with a common advisory prefix (for example: "you should", "try ", "do this", "here's what", "first,"), the method prepends "That makes sense. " to the stripped text; otherwise it returns the stripped text unchanged.
        
        Parameters:
            text (str): The input text to normalize and potentially adjust.
            user_id (str): Identifier for the user (accepted for interface compatibility; not used in the current implementation).
        
        Returns:
            str: The trimmed input text, possibly prefixed with a short validation phrase.
        """
        t = text.strip()
        # Validation-first: if starts with advice, prepend a validation phrase
        lowers = t.lower()
        if any(lowers.startswith(x) for x in ["you should", "try ", "do this", "here's what", "first,"]):
            t = "That makes sense. " + t
        return t

