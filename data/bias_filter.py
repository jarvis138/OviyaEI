from typing import Dict, Tuple, List
import re


class CulturalBiasFilter:
    STEREOTYPE_PATTERNS = {
        'ja_jp': [r'\bbowing\b', r'\bsamurai\b', r'\bsushi\b', r'\bpolite\b.*\ball\b', r'\brobots?\b'],
        'hi_in': [r'\bcurry\b', r'\byoga\b', r'\bspiritual\b.*\ball\b', r'\bcall center\b', r'\btech support\b'],
        'ko_kr': [r'\bkpop\b', r'\bplastic surgery\b', r'\bworkaholics?\b'],
        'el_gr': [r'\blazy\b', r'\bdebt\b', r'\bzeus\b.*\bmentioned\b'],
        'sv_se': [r'\bbland\b', r'\bcold\b']
    }

    def __init__(self, llm_judge_fn=None):
        self.llm_judge = llm_judge_fn

    def check_pattern_based(self, text: str, culture: str) -> Tuple[bool, List[str]]:
        pats = self.STEREOTYPE_PATTERNS.get(culture, [])
        text_lower = (text or '').lower()
        matches = [p for p in pats if re.search(p, text_lower)]
        return len(matches) > 0, matches

    def filter_sample(self, sample: Dict, threshold: float = 0.4, use_llm: bool = False) -> Tuple[bool, Dict]:
        text = sample.get('response', '')
        culture = sample.get('culture', '')
        has_pattern, patterns = self.check_pattern_based(text, culture)
        meta = {'pattern_matched': has_pattern, 'matched_patterns': patterns, 'llm_score': None}
        if has_pattern:
            return False, meta
        return True, meta


