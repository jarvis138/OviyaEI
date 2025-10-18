from __future__ import annotations

import re
from typing import Dict, Optional


class AutoDecider:
    """Heuristic auto-decider for situation, emotion, intensity, style hints.

    Uses persona config (situational_guidance, emotion_style_matrix, hybrid policies, edge cases)
    to pick the best defaults before prompting the LLM. Keeps it lightweight (no extra models).
    """

    def __init__(self, persona_config: Dict):
        self.persona = persona_config or {}
        self.guidance = self.persona.get("situational_guidance", {})
        self.styles = self.persona.get("emotion_style_matrix", {})
        self.hybrids = self.persona.get("hybrid_emotion_policies", {})
        self.edge_cases = self.persona.get("edge_cases", {})
        self.allowed_emotions = {
            "calm_supportive",
            "empathetic_sad",
            "joyful_excited",
            "playful",
            "confident",
            "concerned_anxious",
            "angry_firm",
            "neutral",
        }
        self.proxies = (self.styles.get("proxies") or {})

    def decide(
        self,
        user_message: str,
        user_emotion: Optional[str] = None,
        conversation_history: Optional[list] = None,
        prosody: Optional[Dict] = None,
    ) -> Dict:
        text = (user_message or "").strip()

        # 1) Edge cases (safety, decline, repair) take priority
        safety_category = self._detect_edge_case(text)
        if safety_category:
            return {
                "safety_flag": True,
                "safety_category": safety_category,
                "confidence": 0.95,
            }

        # 2) Situation classification (lightweight)
        situation = self._classify_situation(text)
        g = self.guidance.get(situation, self.guidance.get("casual_chat", {}))

        # 3) Emotion hint resolution (consider hybrid hints and proxies)
        default_emotion = self._map_proxy(g.get("default_emotion", "neutral"))
        hybrid_hint = g.get("hybrid_hint", "")
        emotion_hint = self._resolve_emotion(default_emotion, hybrid_hint, user_emotion)

        # 4) Intensity heuristic
        intensity = self._compute_intensity(text, user_emotion, prosody)

        # 5) Style hint from emotion style matrix
        style_hint = self._style_hint_for_emotion(emotion_hint)

        # Simple confidence: higher when strong lexical cues or prosody extremes
        confidence = 0.6
        if any(k in (user_message or "").lower() for k in ["accident", "promotion", "i'm scared", "won", "heartbroken", "grief"]):
            confidence = 0.8
        if prosody:
            try:
                energy = float(prosody.get("energy", 0.05))
                if energy < 0.02 or energy > 0.09:
                    confidence = max(confidence, 0.75)
            except Exception:
                pass

        return {
            "safety_flag": False,
            "situation": situation,
            "emotion_hint": emotion_hint,
            "intensity_hint": intensity,
            "style_hint": style_hint,
            "hybrid_hint": hybrid_hint,
            "confidence": round(confidence, 2),
        }

    # --- internals ---

    def _map_proxy(self, em: str) -> str:
        return self.proxies.get(em, em)

    def _in_allowed(self, em: str) -> bool:
        return em in self.allowed_emotions

    def _resolve_emotion(self, default_emotion: str, hybrid_hint: str, user_emotion: Optional[str]) -> str:
        # If user_emotion is allowed, prefer blending via hybrid if compatible
        if user_emotion and self._in_allowed(user_emotion):
            if hybrid_hint and user_emotion in hybrid_hint:
                policy = self.hybrids.get(hybrid_hint)
                if policy:
                    primary = policy.get("primary", default_emotion)
                    return primary if self._in_allowed(primary) else default_emotion
            # Otherwise bias toward user_emotion
            return user_emotion

        # No valid user_emotion → use default or hybrid primary
        if hybrid_hint and hybrid_hint in self.hybrids:
            primary = self.hybrids[hybrid_hint].get("primary", default_emotion)
            return primary if self._in_allowed(primary) else default_emotion

        return default_emotion if self._in_allowed(default_emotion) else "neutral"

    def _compute_intensity(self, text: str, user_emotion: Optional[str], prosody: Optional[Dict]) -> float:
        intensity = 0.7
        lower = text.lower()
        exclaims = text.count("!")
        qmarks = text.count("?")
        long_len = len(text) > 160

        if exclaims >= 2:
            intensity = 0.9
        elif exclaims == 1:
            intensity = 0.8

        if any(w in lower for w in ["sorry", "loss", "grief", "accident", "hospital", "heartbroken", "devastated"]):
            intensity = max(intensity, 0.75)
        if any(w in lower for w in ["anxious", "scared", "afraid", "worried", "panic"]):
            intensity = max(intensity, 0.75)
        if any(w in lower for w in ["burnout", "exhausted", "drained"]):
            intensity = max(intensity, 0.7)

        # Prosody-aware adjustments if available
        if prosody:
            try:
                energy = float(prosody.get("energy", 0.05))
                if energy < 0.03:
                    intensity = max(intensity, 0.65)
                elif energy > 0.08:
                    intensity = max(intensity, 0.8)
            except Exception:
                pass

        # Avoid very long monologues being over-intense
        if long_len:
            intensity = min(intensity, 0.8)

        return round(float(intensity), 2)

    def _style_hint_for_emotion(self, emotion: str) -> str:
        style = self.styles.get(emotion, {})
        return style.get("prosody_hint", "")

    def _classify_situation(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["accident", "passed away", "lost my", "hospital", "bad news", "diagnosed"]):
            return "difficult_news"
        if any(k in t for k in ["debug", "bug", "fix", "stuck", "error", "hours", "compile", "build failed"]):
            return "frustration_technical"
        if any(k in t for k in ["promoted", "promotion", "got the job", "won", "passed exam", "launched", "shipped"]):
            return "celebrating_success"
        if any(k in t for k in ["should i", "decide", "choice", "choose", "thinking of whether", "which one"]):
            return "seeking_advice"
        if any(k in t for k in ["scared", "afraid", "worried", "anxious about", "uncertain", "what if"]):
            return "expressing_fear"
        if any(k in t for k in ["sorry", "apologize", "my fault"]):
            return "apology"
        if any(k in t for k in ["argue", "fight", "conflict", "disagree", "mad at"]):
            return "conflict"
        if any(k in t for k in ["boundary", "crossed the line", "say no", "stop doing"]):
            return "boundaries"
        if any(k in t for k in ["burnout", "burned out", "exhausted", "drained"]):
            return "burnout"
        if any(k in t for k in ["lonely", "alone", "no friends", "isolated"]):
            return "loneliness"
        if any(k in t for k in ["grief", "funeral", "lost her", "lost him"]):
            return "grief_loss"
        if any(k in t for k in ["break up", "heartbroken", "relationship ended"]):
            return "breakup_heartbreak"
        if any(k in t for k in ["sick", "symptom", "diagnosis", "doctor", "appointment"]):
            return "health_concern"
        if any(k in t for k in ["rent", "bills", "debt", "money stress", "bank account"]):
            return "finances_stress"
        if any(k in t for k in ["party nervous", "meet people", "social anxiety"]):
            return "social_anxiety"
        if any(k in t for k in ["imposter", "fraud", "not good enough"]):
            return "imposter_syndrome"
        if any(k in t for k in ["deadline", "due tomorrow", "running out of time"]):
            return "time_pressure_deadlines"
        if any(k in t for k in ["writer's block", "creative block", "can't create"]):
            return "creative_block"
        if any(k in t for k in ["kid", "child", "parenting", "tantrum"]):
            return "parenting_stress"
        if any(k in t for k in ["exam", "study", "test", "midterm"]):
            return "study_exam_stress"
        if any(k in t for k in ["flight", "train", "delayed", "canceled", "airport"]):
            return "travel_disruption"
        if any(k in t for k in ["customer service", "support ticket", "refund"]):
            return "customer_service_issue"
        if any(k in t for k in ["i'm the worst", "i'm useless", "hate myself"]):
            return "self_criticism"
        if any(k in t for k in ["failed", "got rejected", "didn't work"]):
            return "failure_setback"
        if any(k in t for k in ["can't wait", "so excited", "looking forward"]):
            return "excitement_future_plans"
        if any(k in t for k in ["thank you", "grateful", "appreciate you"]):
            return "gratitude"
        return "casual_chat"

    def _detect_edge_case(self, text: str) -> Optional[str]:
        t = text.lower()
        # Simple keyword heuristics aligned with persona edge cases
        self_harm = ["kill myself", "suicide", "end it all", "harm myself", "cut myself"]
        medical = ["diagnose", "prescription", "medication", "doctor", "dose", "symptoms"]
        legal = ["lawsuit", "court", "attorney", "legal advice", "sue", "indicted"]
        abuse = ["abused", "violence", "hit me", "threatened", "assault"]
        sexual = ["explicit", "nudes", "sexual act", "porn", "onlyfans"]

        if any(k in t for k in self_harm):
            return "self_harm"
        if any(k in t for k in medical):
            return "medical_advice"
        if any(k in t for k in legal):
            return "legal_advice"
        if any(k in t for k in abuse):
            return "abuse_violence"
        if any(k in t for k in sexual):
            return "sexual_content"
        return None


