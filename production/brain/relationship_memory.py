from __future__ import annotations

from typing import Dict, Optional, List


class RelationshipMemorySystem:
    """Very lightweight episodic/semantic memory for recalls.

    In production, back this with persistent store; here, keep per-session hints.
    """

    def __init__(self):
        self.episodes: List[Dict] = []

    def store_interaction(self, user_input: str, ai_response: str, emotion: str):
        entry = {
            "user": (user_input or "")[:256],
            "ai": (ai_response or "")[:256],
            "emotion": emotion,
        }
        self.episodes.append(entry)
        if len(self.episodes) > 100:
            self.episodes = self.episodes[-100:]

    def proactive_recall(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        # Simple cues
        if "coffee" in t:
            return "Coffee reminds me of your earlier story—how did that turn out?"
        if "interview" in t:
            return "Is this like the interview you mentioned before—what helped then?"
        if "family" in t:
            return "You’ve shared family matters matter a lot—what would support look like now?"
        return None




