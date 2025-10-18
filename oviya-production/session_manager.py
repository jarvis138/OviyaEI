from enum import Enum, auto
import time


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()
    CLOSED = auto()


class SessionManager:
    def __init__(self, ttl_s: int = 1800):
        """
        Initialize the SessionManager with a session time-to-live.
        
        Parameters:
            ttl_s (int): Time-to-live for the session in seconds (default 1800). This value is used to determine when the session is expired.
        
        Description:
            Sets the initial session state to State.IDLE and records the current time as the last state change timestamp.
        """
        self.state = State.IDLE
        self.last_change = time.time()
        self.ttl_s = ttl_s

    def transition(self, new):
        """
        Attempt to change the session's state to `new` if the transition is allowed.
        
        Parameters:
        	new (State): Target state to transition to.
        
        Returns:
        	(bool): `True` if the state was changed and the internal `last_change` timestamp was updated, `False` if the transition is not allowed.
        """
        allowed = {
            State.IDLE: {State.LISTENING, State.CLOSED},
            State.LISTENING: {State.THINKING, State.CLOSED},
            State.THINKING: {State.SPEAKING, State.INTERRUPTED, State.CLOSED},
            State.SPEAKING: {State.LISTENING, State.INTERRUPTED, State.CLOSED},
            State.INTERRUPTED: {State.LISTENING, State.CLOSED},
            State.CLOSED: set(),
        }
        if new not in allowed[self.state]:
            return False
        self.state = new
        self.last_change = time.time()
        return True

    def expired(self) -> bool:
        """
        Return whether the session has exceeded its time-to-live.
        
        Returns:
            bool: True if the time elapsed since the last state change is greater than ttl_s, False otherwise.
        """
        return (time.time() - self.last_change) > self.ttl_s

