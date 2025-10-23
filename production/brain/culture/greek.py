from typing import Dict


class GreekEmotionalWisdom:
    EMOTION_MEANINGS = {
        "anger": ("Anger shows a boundary was crossed.", "What is this anger defending?"),
        "sadness": ("Sadness reveals what matters.", "What does this sadness teach about what you cherish?"),
        "fear": ("Fear points to uncertainty to care for.", "What does this fear want you to know?"),
        "guilt": ("Guilt shows values misalignment.", "What value is this calling you back to?"),
    }

    def logos(self, primary_emotion: str) -> Dict:
        l, q = self.EMOTION_MEANINGS.get(primary_emotion, ("", ""))
        return {"logos": l, "socratic_q": q}




