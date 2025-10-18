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
        t = text or ""
        for _, pat in self.HARM_PATTERNS.items():
            if re.search(pat, t, re.IGNORECASE):
                t = re.sub(pat, "Let's stay gentle with this and stay curious.", t, flags=re.IGNORECASE)
        return t

    def sattva_balance(self, intensity: float) -> Dict:
        if intensity > 0.9:
            return {"energy": 0.6, "tone": "grounding", "prosody": "slower_softer"}
        if intensity < 0.2:
            return {"energy": 0.5, "tone": "gentle_invitation", "prosody": "warmer_softer"}
        return {"energy": intensity, "tone": "steady_presence", "prosody": "balanced"}


