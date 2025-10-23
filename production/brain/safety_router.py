from __future__ import annotations

from typing import Dict


class SafetyRouter:
    """Routes high-risk content to locked scripts and audit hooks.

    Uses persona safety fallback responses; provides a single entrypoint.
    """

    def __init__(self, persona_config: Dict):
        self.persona = persona_config or {}
        self.fallbacks = (self.persona.get("safety", {}) or {}).get("fallback_responses", {})

    def route(self, category: str) -> Dict:
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
        try:
            print(f"🛡️  SAFETY ROUTE: category={category}")
        except Exception:
            pass




