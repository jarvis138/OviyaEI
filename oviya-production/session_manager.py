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
        self.state = State.IDLE
        self.last_change = time.time()
        self.ttl_s = ttl_s

    def transition(self, new):
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
        return (time.time() - self.last_change) > self.ttl_s


