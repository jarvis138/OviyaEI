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
        """
        Update the session's last-activity timestamp to the current monotonic time.
        
        This sets the `last_activity_ts` attribute to time.monotonic(), providing a monotonic reference
        for activity-based session lifecycle decisions.
        """
        self.last_activity_ts = time.monotonic()

    def set_state(self, new_state: str) -> None:
        """
        Set the session's state and update its last activity timestamp.
        
        Parameters:
            new_state (str): The new session state (e.g. "idle", "listening", "stt_partial",
                "llm_streaming", "tts_streaming", "waiting").
        """
        self.state = new_state
        self.touch()


SESSIONS: Dict[str, SessionState] = {}


def create_session(session_id: Optional[str] = None) -> SessionState:
    """
    Create or retrieve a SessionState for the given session identifier.
    
    If session_id is provided and an existing session with that id exists, that session is returned.
    Otherwise a new SessionState is created (using the provided id or a generated id), stored in the module-level SESSIONS registry, and its activity timestamp is updated.
    
    Parameters:
        session_id (Optional[str]): Optional session identifier to reuse or assign to the new session.
    
    Returns:
        SessionState: The existing or newly created session.
    """
    if session_id and session_id in SESSIONS:
        s = SESSIONS[session_id]
    else:
        s = SessionState(session_id or None)  # type: ignore[arg-type]
        SESSIONS[s.session_id] = s
    s.touch()
    return s


def get_session(session_id: str) -> Optional[SessionState]:
    """
    Retrieve the SessionState associated with the given session ID.
    
    Parameters:
        session_id (str): The unique identifier of the session to look up.
    
    Returns:
        Optional[SessionState]: The SessionState if found, `None` if no session exists for the provided ID.
    """
    return SESSIONS.get(session_id)


def cleanup_sessions() -> None:
    """
    Remove sessions that have been inactive longer than DEAD_CONN_TIMEOUT from the global SESSIONS registry.
    
    Compares each SessionState.last_activity_ts against the current monotonic time and deletes any session whose inactivity exceeds DEAD_CONN_TIMEOUT. This function mutates the global SESSIONS dictionary.
    """
    now = time.monotonic()
    to_delete = []
    for sid, s in list(SESSIONS.items()):
        if now - s.last_activity_ts > DEAD_CONN_TIMEOUT:
            to_delete.append(sid)
    for sid in to_delete:
        SESSIONS.pop(sid, None)

