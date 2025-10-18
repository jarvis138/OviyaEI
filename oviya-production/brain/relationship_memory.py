from __future__ import annotations

from typing import Dict, Optional, List


class RelationshipMemorySystem:
    """Very lightweight episodic/semantic memory for recalls.

    In production, back this with persistent store; here, keep per-session hints.
    """

    def __init__(self):
        """
        Initialize a per-session episodes store for relationship memory.
        
        Creates an empty list assigned to `self.episodes` to hold interaction entries. Each entry is expected to be a dict with keys "user", "ai", and "emotion"; user/ai text is stored trimmed to 256 characters and the list is intended to be capped at 100 recent entries.
        """
        self.episodes: List[Dict] = []

    def store_interaction(self, user_input: str, ai_response: str, emotion: str):
        """
        Store a user–AI interaction in the session memory.
        
        Creates a memory entry with the provided `user_input`, `ai_response`, and `emotion`, truncating `user_input` and `ai_response` to the first 256 characters (or using an empty string if falsy). Appends the entry to `self.episodes` and keeps only the most recent 100 entries.
        
        Parameters:
        	user_input (str): The user's input text to store.
        	ai_response (str): The AI's response text to store.
        	emotion (str): A short label describing the emotional tone associated with the interaction.
        """
        entry = {
            "user": (user_input or "")[:256],
            "ai": (ai_response or "")[:256],
            "emotion": emotion,
        }
        self.episodes.append(entry)
        if len(self.episodes) > 100:
            self.episodes = self.episodes[-100:]

    def proactive_recall(self, text: str) -> Optional[str]:
        """
        Return a short canned prompt when the input contains one of several cue words.
        
        Scans the provided text for simple cues ("coffee", "interview", "family") and returns a corresponding reminder or prompt for the first matched cue; returns None if no cue is found.
        
        Parameters:
            text (str): Text to scan for cue words.
        
        Returns:
            Optional[str]: A canned prompt for the detected cue, or None if no cue is present.
        """
        t = (text or "").lower()
        # Simple cues
        if "coffee" in t:
            return "Coffee reminds me of your earlier story—how did that turn out?"
        if "interview" in t:
            return "Is this like the interview you mentioned before—what helped then?"
        if "family" in t:
            return "You’ve shared family matters matter a lot—what would support look like now?"
        return None

