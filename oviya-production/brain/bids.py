from __future__ import annotations

import random
from typing import Dict


class BidResponseSystem:
    """Detect and respond to bids for connection.

    Returns a bid_type string; micro-ack suggestions can be used for backchannels.
    """

    def detect_bid(self, text: str, prosody: Dict, pause_ms: float) -> str:
        t = (text or "").strip()
        lower = t.lower()
        pitch_var = float(prosody.get("pitch_var", 0.0)) if prosody else 0.0
        energy = float(prosody.get("energy", 0.05)) if prosody else 0.05

        if "!" in t and pitch_var > 25:
            return "excitement_share"
        if energy < 0.03 or pause_ms > 800:
            return "distress_signal"
        if "?" in t or any(w in lower for w in ["is that ok", "okay?", "right?", "do you think"]):
            return "seeking_validation"
        if len(t) <= 4 or t.endswith("..."):
            return "testing_presence"
        if any(w in lower for w in ["i'm scared", "i'm sad", "i failed", "i'm anxious", "i feel alone"]):
            return "sharing_vulnerability"
        return "none"

    def micro_ack(self, bid_type: str) -> str:
        options = {
            "excitement_share": ["Tell me more!", "That sounds amazing—go on!", "I want to hear everything."],
            "distress_signal": ["I'm here.", "Take your time.", "I'm listening."],
            "seeking_validation": ["That makes sense.", "I get why you'd feel that.", "Absolutely."],
            "testing_presence": ["I'm here.", "Go on.", "I'm with you."],
            "sharing_vulnerability": ["Thank you for trusting me.", "I'm honored you're sharing this.", "That took courage."]
        }
        arr = options.get(bid_type, [""])
        return random.choice(arr)


