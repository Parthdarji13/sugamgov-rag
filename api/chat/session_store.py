"""
api/chat/session_store.py
=========================
Thread-safe in-memory session store for SugamGov AI conversational chat.

Maintains bounded conversational history (last 10 messages) per session.
Designed with a clean interface to easily replace in-memory storage with MongoDB
in future phases.
"""

import time
import uuid
import threading
from typing import Dict, Optional, List

from api.chat.models import ChatMessage, SessionData

DEFAULT_MAX_HISTORY_MESSAGES = 10


class InMemorySessionStore:
    """
    Thread-safe in-memory session store for multi-turn conversations.
    """
    def __init__(self, max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES):
        self.max_history_messages = max_history_messages
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.RLock()

    def create_session(self, session_id: Optional[str] = None) -> SessionData:
        """
        Creates a new conversation session.
        If session_id is not provided, generates a unique UUID4.
        """
        with self._lock:
            sid = session_id.strip() if session_id and session_id.strip() else f"sess_{uuid.uuid4().hex[:16]}"
            session = SessionData(
                session_id=sid,
                messages=[],
                created_at=time.time(),
                updated_at=time.time(),
            )
            self._sessions[sid] = session
            return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Retrieves a session by session_id. Returns None if not found.
        """
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> SessionData:
        """
        Appends a user message and assistant answer to the session's history.
        Enforces the bounded history limit (keeping the most recent N messages).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = self.create_session(session_id)

            now = time.time()
            session.messages.append(ChatMessage(role="user", content=user_message, timestamp=now))
            session.messages.append(ChatMessage(role="assistant", content=assistant_message, timestamp=now))

            # Enforce bounded history
            if len(session.messages) > self.max_history_messages:
                session.messages = session.messages[-self.max_history_messages:]

            session.updated_at = now
            return session

    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a session from memory. Returns True if deleted, False if not found.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        """Returns all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def clear_all(self) -> None:
        """Clears all sessions (used for test isolation)."""
        with self._lock:
            self._sessions.clear()


# Global singleton instance for the FastAPI application
_session_store: Optional[InMemorySessionStore] = None


def get_session_store() -> InMemorySessionStore:
    """Returns the singleton instance of InMemorySessionStore."""
    global _session_store
    if _session_store is None:
        _session_store = InMemorySessionStore()
    return _session_store
