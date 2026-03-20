"""
Working Memory module.

Stores the most recent dialogue turns as a sliding context window.
Based on Baddeley (1992) working-memory model: a limited-capacity store
for active, immediately accessible information.

Objective 2a: Working memory for recent conversation context.
"""

from collections import deque
from typing import List, Dict


class WorkingMemory:
    """
    Maintains a sliding window of the most recent conversation turns.

    Each turn is stored as a dict with 'role' and 'content' keys (matching
    the LLaMA chat-message format).  When the window is full the oldest turn
    is evicted automatically.

    Args:
        max_turns (int): Maximum number of dialogue turns to keep in memory.
                         The proposal requires at least 10 turns (Objective 1).
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._buffer: deque = deque(maxlen=max_turns)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Add a single dialogue turn to working memory.

        Args:
            role    : 'user' or 'assistant'
            content : Text of the turn.
        """
        self._buffer.append({"role": role, "content": content})

    def clear(self) -> None:
        """Remove all turns from working memory."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_context(self) -> List[Dict[str, str]]:
        """Return the current window as a list of message dicts."""
        return list(self._buffer)

    def get_last_n(self, n: int) -> List[Dict[str, str]]:
        """Return the *n* most recent turns."""
        turns = list(self._buffer)
        return turns[-n:] if n < len(turns) else turns

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(max_turns={self.max_turns}, "
            f"current_turns={len(self._buffer)})"
        )
