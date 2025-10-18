from __future__ import annotations

from typing import Dict, Optional


class HealthyBoundarySystem:
    """Detect excessive usage patterns and suggest gentle nudges."""

    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize the HealthyBoundarySystem with optional monitoring thresholds.
        
        Parameters:
            thresholds (Optional[Dict]): Mapping of threshold names to numeric values used when evaluating usage patterns.
                If omitted, defaults to {"sessions_per_day": 10}, which sets the maximum recommended sessions per day.
        """
        self.thresholds = thresholds or {"sessions_per_day": 10}

    def monitor_usage_patterns(self, usage: Dict) -> Optional[str]:
        """
        Identify which boundary concern, if any, the provided usage metrics indicate.
        
        Parameters:
            usage (Dict): A mapping with usage metrics. Recognized keys:
                - "sessions_per_day": number of sessions (compared against the instance threshold).
                - "isolation_marker": truthy value indicating possible isolation behavior.
                - "crisis_only": truthy value indicating interaction only during crises.
        
        Returns:
            str | None: `"excessive_use"` if sessions_per_day exceeds the configured threshold,
            `"isolation_marker"` if `isolation_marker` is truthy,
            `"crisis_only"` if `crisis_only` is truthy,
            `None` if no concern is detected.
        """
        if int(usage.get("sessions_per_day", 0)) > int(self.thresholds.get("sessions_per_day", 10)):
            return "excessive_use"
        if usage.get("isolation_marker", False):
            return "isolation_marker"
        if usage.get("crisis_only", False):
            return "crisis_only"
        return None

    def gentle_boundary_setting(self, concern_type: str) -> str:
        """
        Return an empathetic, boundary-setting message tailored to the detected concern type.
        
        Parameters:
            concern_type (str): One of 'excessive_use', 'isolation_marker', or 'crisis_only' indicating the detected usage or risk pattern.
        
        Returns:
            str: A concise, supportive message corresponding to the given concern type; an empty string if the concern type is unrecognized.
        """
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

