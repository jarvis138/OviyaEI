from typing import Dict


class KoreanEmotionalWisdom:
    def __init__(self):
        self.jeong = {}

    def update_jeong(self, user_id: str, session_seconds: int, vulnerability: float, regular_checkin: bool) -> float:
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
        if jeong_depth <= 0.6:
            return None
        return {
            "you are going through": "we're going through this together",
            "your struggle": "our struggle with this",
            "you can do this": "we can do this",
            "your journey": "our journey",
        }


