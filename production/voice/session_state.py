import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Timeouts (tunable)
NO_AUDIO_TIMEOUT = 30.0    # seconds (idle listening)
STUCK_TTS_TIMEOUT = 5.0    # seconds
DEAD_CONN_TIMEOUT = 120.0  # seconds (hard cleanup)


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: str = "idle"   # idle, listening, stt_partial, llm_streaming, tts_streaming, waiting
    last_activity_ts: float = field(default_factory=time.monotonic)
    active_tasks: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    cancel_flags: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_activity_ts = time.monotonic()

    def set_state(self, new_state: str) -> None:
        self.state = new_state
        self.touch()


SESSIONS: Dict[str, SessionState] = {}


def create_session(session_id: Optional[str] = None) -> SessionState:
    if session_id and session_id in SESSIONS:
        s = SESSIONS[session_id]
    else:
        s = SessionState(session_id or None)  # type: ignore[arg-type]
        SESSIONS[s.session_id] = s
    s.touch()
    return s


def get_session(session_id: str) -> Optional[SessionState]:
    return SESSIONS.get(session_id)


def cleanup_sessions() -> None:
    now = time.monotonic()
    to_delete = []
    for sid, s in list(SESSIONS.items()):
        if now - s.last_activity_ts > DEAD_CONN_TIMEOUT:
            to_delete.append(sid)
    for sid in to_delete:
        SESSIONS.pop(sid, None)


