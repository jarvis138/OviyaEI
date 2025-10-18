from typing import Dict


class GreekEmotionalWisdom:
    EMOTION_MEANINGS = {
        "anger": ("Anger shows a boundary was crossed.", "What is this anger defending?"),
        "sadness": ("Sadness reveals what matters.", "What does this sadness teach about what you cherish?"),
        "fear": ("Fear points to uncertainty to care for.", "What does this fear want you to know?"),
        "guilt": ("Guilt shows values misalignment.", "What value is this calling you back to?"),
    }

    def logos(self, primary_emotion: str) -> Dict:
        """
        Retrieve the logos text and Socratic question associated with a primary emotion.
        
        Parameters:
            primary_emotion (str): The emotion name used to look up entries in EMOTION_MEANINGS.
        
        Returns:
            Dict: A dictionary with keys "logos" (the logos text) and "socratic_q" (the Socratic question). If the emotion is not found, both values are empty strings.
        """
        l, q = self.EMOTION_MEANINGS.get(primary_emotion, ("", ""))
        return {"logos": l, "socratic_q": q}

