"""
Working Memory Module
=====================

Maintains the most recent N turn-pairs in a sliding-window deque.
When the window is full, the oldest turns are evicted and returned
to the caller for consolidation into episodic memory.

Reference: Baddeley, A.D. (1992). Working memory. Science, 255(5044), 556-559.
"""

import time
from collections import deque
from typing import Dict, List, Optional


class WorkingMemory:
    """
    Sliding-window working memory that holds the most recent conversation turns.

    Each entry is a dict with keys:
        role      — "user" | "assistant"
        content   — turn text
        turn_id   — monotonic counter across the full dialogue
        timestamp — Unix float

    When ``len(self._turns) > max_turns * 2`` (pairs × 2 for user+assistant),
    the oldest entry is evicted and returned to the caller so it can be
    consolidated into episodic memory.
    """

    def __init__(self, max_turns: int = 6):
        """
        Args:
            max_turns: Maximum number of user/assistant *pairs* to keep
                       in the window. Actual message count is max_turns × 2.
        """
        self.max_turns = max_turns
        self._turns: deque = deque()
        self._turn_counter: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> List[Dict]:
        """
        Append a new turn and evict the oldest messages if over capacity.

        Args:
            role:    "user" or "assistant"
            content: Turn text

        Returns:
            List of evicted turn dicts (may be empty).
        """
        self._turn_counter += 1
        self._turns.append(
            {
                "role": role,
                "content": content,
                "turn_id": self._turn_counter,
                "timestamp": time.time(),
            }
        )

        evicted: List[Dict] = []
        while len(self._turns) > self.max_turns * 2:
            evicted.append(self._turns.popleft())

        return evicted

    def get_turns(self) -> List[Dict]:
        """Return all current turns in chronological order (full dicts)."""
        return list(self._turns)

    def get_context_messages(self) -> List[Dict]:
        """Return turns as HuggingFace-style ``{"role": ..., "content": ...}`` dicts."""
        return [{"role": t["role"], "content": t["content"]} for t in self._turns]

    def clear(self):
        """Flush all turns (e.g. at session boundary)."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turn_count(self) -> int:
        """Total turns ever added (not reset by clear())."""
        return self._turn_counter

    @property
    def pair_count(self) -> int:
        """Number of complete user/assistant pairs currently in window."""
        return len(self._turns) // 2
