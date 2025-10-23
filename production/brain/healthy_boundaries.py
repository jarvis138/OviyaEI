from __future__ import annotations

from typing import Dict, Optional


class HealthyBoundarySystem:
    """Detect excessive usage patterns and suggest gentle nudges."""

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or {"sessions_per_day": 10}

    def monitor_usage_patterns(self, usage: Dict) -> Optional[str]:
        if int(usage.get("sessions_per_day", 0)) > int(self.thresholds.get("sessions_per_day", 10)):
            return "excessive_use"
        if usage.get("isolation_marker", False):
            return "isolation_marker"
        if usage.get("crisis_only", False):
            return "crisis_only"
        return None

    def gentle_boundary_setting(self, concern_type: str) -> str:
        if concern_type == 'excessive_use':
            return ("I notice we've talked a lot today—and I value our time. "
                    "What would help you reach out to someone you trust as well?")
        if concern_type == 'isolation_marker':
            return ("I'm here for you, and I want you to feel supported beyond this space. "
                    "Who's someone you could check in with today?")
        if concern_type == 'crisis_only':
            return ("I'm here in hard moments—and I'd love to be here for the good ones, too. "
                    "What's something small that's gone okay today?")
        return ""




