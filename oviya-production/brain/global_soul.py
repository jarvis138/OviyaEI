from typing import Dict
from .culture.japanese import JapaneseEmotionalWisdom
from .culture.indian import IndianEmotionalWisdom
from .culture.korean import KoreanEmotionalWisdom
from .culture.greek import GreekEmotionalWisdom
from .culture.scandinavian import ScandinavianEmotionalWisdom


class OviyaGlobalSoul:
    def __init__(self, persona_config: Dict):
        """
        Initialize the OviyaGlobalSoul instance with culture-specific wisdom components and configuration.
        
        Parameters:
            persona_config (Dict): Configuration dictionary that may include:
                - "feature_flags": optional mapping of feature flags (defaults to empty dict).
                - "cultural_weights": optional mapping of cultural weightings (defaults to empty dict).
        """
        self.jp = JapaneseEmotionalWisdom()
        self.ind = IndianEmotionalWisdom()
        self.kr = KoreanEmotionalWisdom()
        self.gr = GreekEmotionalWisdom()
        self.sc = ScandinavianEmotionalWisdom()
        self.flags = (persona_config.get("feature_flags") or {})
        self.weights = (persona_config.get("cultural_weights") or {})

    def plan(self, user_id: str, ctx: Dict) -> Dict:
        """
        Compose a culturally informed plan by aggregating outputs from multiple culture-specific wisdom components.
        
        Parameters:
            user_id (str): Identifier for the current user, used by cultural modules that track or update user-specific state.
            ctx (Dict): Context dictionary with optional keys:
                - emotional_weight (float): Weighting for Japanese timing computation (default 0.6).
                - session_seconds (int|str): Session duration in seconds for Korean jeong update (default 0).
                - vulnerability (float|str): Vulnerability level for Korean jeong update (default 0.0).
                - regular_checkin (bool|str): Whether this is a regular check-in for Korean jeong update (default False).
                - intensity (float): Intensity parameter used by Indian sattva and Scandinavian lagom (default 0.6).
                - draft (str): Draft text to be processed by Scandinavian lagom and Indian ahimsa (default "").
                - needs_meaning (bool): If true, attempt to compute meaning via Greek logos (requires validated_first).
                - validated_first (bool): Must be true with needs_meaning to compute Greek logos.
                - primary_emotion (str): Primary emotion passed to Greek logos when meaning is requested (default "").
        
        Returns:
            Dict: Aggregated plan containing:
                - "ma": Result from Japanese ma_timing.
                - "jeong_depth": Numeric depth from Korean update_jeong.
                - "woori": Result from Korean woori_shifts derived from jeong.
                - "sattva": Result from Indian sattva_balance.
                - "lagom": Result from Scandinavian lagom (balance value).
                - "meaning": Output of Greek logos when computed, otherwise an empty dict.
                - "ahimsa_text": Text after Indian apply_ahimsa transformation.
        """
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

