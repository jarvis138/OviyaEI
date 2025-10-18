from typing import Dict, Tuple


class ScandinavianEmotionalWisdom:
    def lagom(self, intensity: float, text: str) -> Tuple[str, str]:
        """
        Return a text adjusted for emotional intensity and a style tag indicating delivery.
        
        Parameters:
            intensity (float): Emotional intensity on a 0.0–1.0 scale.
            text (str): Original text to deliver; falsy values are treated as an empty string.
        
        Returns:
            tuple[str, str]: A pair (message, style_tag). `message` is the original `text` possibly prefixed with a supportive line when intensity is greater than 0.85 or less than 0.15. `style_tag` is one of "slower_calmer", "warmer_gentler", or "maintain_balance" describing the recommended delivery style.
        """
        if intensity > 0.85:
            t = "This is a lot to hold. Let's take it one piece at a time.\n\n" + (text or "")
            return t, "slower_calmer"
        if intensity < 0.15:
            t = "I'm here with you, even in this heaviness.\n\n" + (text or "")
            return t, "warmer_gentler"
        return text or "", "maintain_balance"

