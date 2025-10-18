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
        """
        Initialize the CulturalBiasFilter with an optional custom judge function.
        
        Parameters:
            llm_judge_fn (callable | None): Optional function used to score or judge text for bias; stored as `self.llm_judge`. If None, no LLM-based judgment is used.
        """
        self.llm_judge = llm_judge_fn

    def check_pattern_based(self, text: str, culture: str) -> Tuple[bool, List[str]]:
        """
        Determine whether the given text matches any stereotype regex patterns for the specified culture.
        
        Parameters:
            text (str): The input text to check; None is treated as an empty string.
            culture (str): Culture code key used to select regex patterns (e.g., 'ja_jp').
        
        Returns:
            (bool, List[str]): `True` if at least one pattern matched, `False` otherwise; second element is the list of matching regex patterns.
        """
        pats = self.STEREOTYPE_PATTERNS.get(culture, [])
        text_lower = (text or '').lower()
        matches = [p for p in pats if re.search(p, text_lower)]
        return len(matches) > 0, matches

    def filter_sample(self, sample: Dict, threshold: float = 0.4, use_llm: bool = False) -> Tuple[bool, Dict]:
        """
        Filter a single sample by detecting culture-specific stereotype patterns in its response.
        
        Parameters:
            sample (Dict): Input record expected to contain a 'response' string and a 'culture' code; missing keys default to empty string.
            threshold (float): Ignored by this implementation; present for API compatibility.
            use_llm (bool): Ignored by this implementation; present for API compatibility.
        
        Returns:
            Tuple[bool, Dict]: A pair where the first element is `True` if the sample passes the filter (no pattern matched) and `False` if it is filtered out (pattern matched). The second element is a meta dictionary with keys:
                - 'pattern_matched' (bool): whether any stereotype pattern matched the response.
                - 'matched_patterns' (List[str]): the regex patterns that matched.
                - 'llm_score' (None): placeholder for an optional LLM-based score (always `None` in this implementation).
        """
        text = sample.get('response', '')
        culture = sample.get('culture', '')
        has_pattern, patterns = self.check_pattern_based(text, culture)
        meta = {'pattern_matched': has_pattern, 'matched_patterns': patterns, 'llm_score': None}
        if has_pattern:
            return False, meta
        return True, meta

