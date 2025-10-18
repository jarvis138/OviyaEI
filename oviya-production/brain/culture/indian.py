import re
from typing import Dict


class IndianEmotionalWisdom:
    HARM_PATTERNS = {
        "comparison": r"others have it worse|at least you",
        "dismissal": r"you're overreacting|it's not that bad",
        "shame": r"you shouldn't feel|that's wrong to think",
        "toxicity": r"they're toxic|cut them off|you deserve better",
    }

    def apply_ahimsa(self, text: str) -> str:
        """
        Replace phrases matching configured harm patterns with a gentle, curious prompt.
        
        Scans the input text for any regular expressions defined in the class attribute HARM_PATTERNS and replaces each match with "Let's stay gentle with this and stay curious." Matching is case-insensitive and applied across all patterns; a falsy input is treated as an empty string.
        
        Parameters:
            text (str): Input string to sanitize; if falsy, treated as "".
        
        Returns:
            str: The input text with harmful phrases replaced by the gentler prompt.
        """
        t = text or ""
        for _, pat in self.HARM_PATTERNS.items():
            if re.search(pat, t, re.IGNORECASE):
                t = re.sub(pat, "Let's stay gentle with this and stay curious.", t, flags=re.IGNORECASE)
        return t

    def sattva_balance(self, intensity: float) -> Dict:
        """
        Map a numeric emotional intensity to a conversational energy profile (energy, tone, and prosody).
        
        Parameters:
            intensity (float): Emotional intensity, typically in the range 0.0–1.0; lower values indicate calmer states and higher values indicate heightened states.
        
        Returns:
            dict: A mapping with keys:
                - "energy" (float): Suggested energy level (0.0–1.0 scale or comparable relative value).
                - "tone" (str): Short descriptor of the conversational tone to adopt.
                - "prosody" (str): Descriptor of speech prosody characteristics.
        """
        if intensity > 0.9:
            return {"energy": 0.6, "tone": "grounding", "prosody": "slower_softer"}
        if intensity < 0.2:
            return {"energy": 0.5, "tone": "gentle_invitation", "prosody": "warmer_softer"}
        return {"energy": intensity, "tone": "steady_presence", "prosody": "balanced"}

