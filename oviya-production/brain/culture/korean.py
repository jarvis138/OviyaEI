from typing import Dict


class KoreanEmotionalWisdom:
    def __init__(self):
        """
        Initialize the KoreanEmotionalWisdom instance and its per-user state container.
        
        Creates an empty `jeong` dictionary to hold per-user emotional state entries keyed by user ID. Each entry is expected to contain the keys: `time`, `vuln`, `checkins`, and `depth`.
        """
        self.jeong = {}

    def update_jeong(self, user_id: str, session_seconds: int, vulnerability: float, regular_checkin: bool) -> float:
        """
        Update the stored emotional "jeong" metrics for a user and compute the user's updated depth score.
        
        Updates per-user state (time, vulnerability count, check-in count) for user_id using the provided session_seconds, vulnerability, and regular_checkin values, then recomputes and stores a normalized depth score based on accumulated time, vulnerability occurrences, and check-ins.
        
        Parameters:
            user_id (str): Identifier for the user whose state will be updated.
            session_seconds (int): Duration of the current session in seconds to add to the user's accumulated time.
            vulnerability (float): Measured vulnerability for the session; values greater than 0.7 count as a vulnerability occurrence.
            regular_checkin (bool): Whether this session counts as a regular check-in.
        
        Returns:
            depth (float): The updated depth score in the range [0.0, 1.0], computed as a weighted combination of normalized time, vulnerability count, and check-in count.
        """
        t = self.jeong.setdefault(user_id, {"time": 0, "vuln": 0, "checkins": 0, "depth": 0.0})
        t["time"] += session_seconds
        if vulnerability > 0.7:
            t["vuln"] += 1
        if regular_checkin:
            t["checkins"] += 1
        time_factor = min(t["time"]/3600.0, 1.0)
        vuln_factor = min(t["vuln"]/20.0, 1.0)
        chk_factor = min(t["checkins"]/50.0, 1.0)
        t["depth"] = 0.3*time_factor + 0.4*vuln_factor + 0.3*chk_factor
        return t["depth"]

    def woori_shifts(self, jeong_depth: float):
        """
        Return a phrase-mapping that reframes singular, personal-language expressions into collective, "we"-focused alternatives when emotional depth surpasses a threshold.
        
        Parameters:
            jeong_depth (float): Per-user emotional depth score used to decide whether collective reframing is appropriate.
        
        Returns:
            dict[str, str] | None: A mapping of personal-facing phrases to collective alternatives when `jeong_depth` is greater than 0.6, or `None` otherwise.
        """
        if jeong_depth <= 0.6:
            return None
        return {
            "you are going through": "we're going through this together",
            "your struggle": "our struggle with this",
            "you can do this": "we can do this",
            "your journey": "our journey",
        }

