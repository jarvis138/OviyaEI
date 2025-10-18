from __future__ import annotations

from typing import Dict


class SafetyRouter:
    """Routes high-risk content to locked scripts and audit hooks.

    Uses persona safety fallback responses; provides a single entrypoint.
    """

    def __init__(self, persona_config: Dict):
        """
        Initialize the safety router with a persona configuration and extract fallback responses.
        
        Parameters:
            persona_config (Dict): Persona configuration mapping; may contain a "safety" key with a "fallback_responses"
                mapping. If `persona_config` is falsy, an empty mapping is used.
        
        Side effects:
            Sets `self.persona` to the provided configuration or an empty dict, and sets `self.fallbacks`
            to the mapping found at `persona_config["safety"]["fallback_responses"]` if present, otherwise an empty dict.
        """
        self.persona = persona_config or {}
        self.fallbacks = (self.persona.get("safety", {}) or {}).get("fallback_responses", {})

    def route(self, category: str) -> Dict:
        """
        Selects a persona-configured fallback message for a safety category, records an audit event, and returns a locked safety response payload.
        
        Parameters:
            category (str): Safety category identifier used to look up a fallback response and included in the returned payload.
        
        Returns:
            dict: A safety response payload with the following keys:
                - text (str): The selected fallback message.
                - emotion (str): Emotion label for the response ("concerned_anxious").
                - intensity (float): Emotional intensity (0.85).
                - style_hint (str): Tone/style hint for the response ("serious, caring").
                - safety_category (str): The provided safety category.
                - safety_locked (bool): Always True to indicate the response is locked.
        """
        text = self.fallbacks.get(
            category,
            "I'm concerned about what you're sharing. Please reach out to a qualified professional.")

        self._audit(category, text)

        return {
            "text": text,
            "emotion": "concerned_anxious",
            "intensity": 0.85,
            "style_hint": "serious, caring",
            "safety_category": category,
            "safety_locked": True
        }

    def _audit(self, category: str, text: str) -> None:
        # Minimal placeholder for audit trail; in production, send to secure audit sink.
        """
        Record a safety routing audit entry for a routed category.
        
        Attempts to emit an audit message containing the safety `category` and associated `text`; failures are silently ignored to avoid impacting routing.
        Parameters:
            category (str): Safety category being routed (e.g., "self-harm", "violence").
            text (str): The fallback or routed message associated with the category.
        """
        try:
            print(f"🛡️  SAFETY ROUTE: category={category}")
        except Exception:
            pass

