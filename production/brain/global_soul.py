from typing import Dict
from .culture.japanese import JapaneseEmotionalWisdom
from .culture.indian import IndianEmotionalWisdom
from .culture.korean import KoreanEmotionalWisdom
from .culture.greek import GreekEmotionalWisdom
from .culture.scandinavian import ScandinavianEmotionalWisdom


class OviyaGlobalSoul:
    def __init__(self, persona_config: Dict):
        self.jp = JapaneseEmotionalWisdom()
        self.ind = IndianEmotionalWisdom()
        self.kr = KoreanEmotionalWisdom()
        self.gr = GreekEmotionalWisdom()
        self.sc = ScandinavianEmotionalWisdom()
        self.flags = (persona_config.get("feature_flags") or {})
        self.weights = (persona_config.get("cultural_weights") or {})

    def plan(self, user_id: str, ctx: Dict) -> Dict:
        ma = self.jp.ma_timing(ctx.get("emotional_weight", 0.6))
        jeong = self.kr.update_jeong(
            user_id,
            int(ctx.get("session_seconds", 0)),
            float(ctx.get("vulnerability", 0.0)),
            bool(ctx.get("regular_checkin", False))
        )
        woori = self.kr.woori_shifts(jeong)
        sattva = self.ind.sattva_balance(ctx.get("intensity", 0.6))
        text = ctx.get("draft", "")
        text, lagom = self.sc.lagom(ctx.get("intensity", 0.6), text)
        meaning = {}
        if ctx.get("needs_meaning") and ctx.get("validated_first"):
            meaning = self.gr.logos(ctx.get("primary_emotion", ""))
        return {
            "ma": ma,
            "jeong_depth": jeong,
            "woori": woori,
            "sattva": sattva,
            "lagom": lagom,
            "meaning": meaning,
            "ahimsa_text": self.ind.apply_ahimsa(text),
        }




