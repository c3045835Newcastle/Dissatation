"""
Memory Controller module.

The memory controller is the coordination layer that decides *what*
information should be stored in each memory tier and *when* consolidation
from working memory into episodic / semantic memory should occur.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory


# ---------------------------------------------------------------------------
# Patterns used to extract facts for semantic memory
# ---------------------------------------------------------------------------

# Detects statements of the form "my name is …" / "I am called …"
# Each word in the captured name must start with an uppercase letter to
# prevent lowercase connectors (e.g. "and", "or") from being included.
# Supports hyphenated names (O'Brien, McDonald) and multi-word names.
_NAME_PATTERNS = [
    re.compile(r"[Mm]y name is ([A-Z][a-zA-Z'-]+(?: [A-Z][a-zA-Z'-]+)*)"),
    re.compile(r"[Ii](?:'m| am) called ([A-Z][a-zA-Z'-]+(?: [A-Z][a-zA-Z'-]+)*)"),
    re.compile(r"[Cc]all me ([A-Z][a-zA-Z'-]+(?: [A-Z][a-zA-Z'-]+)*)"),
]

# Detects preference statements: "I like/love/prefer/enjoy …"
_PREFERENCE_PATTERNS = [
    re.compile(
        r"i (?:like|love|prefer|enjoy|hate|dislike) ([^.!?\n]{3,60})",
        re.IGNORECASE,
    )
]

# Detects goal / objective statements: "I want/need/am trying to …"
_GOAL_PATTERNS = [
    re.compile(
        r"i (?:want|need|am trying) to ([^.!?\n]{3,80})",
        re.IGNORECASE,
    )
]

# Detects occupation / study statements: "I am a/an …" / "I work as …"
_OCCUPATION_PATTERNS = [
    re.compile(r"[Ii](?:'m| am) (?:a|an) ([a-z]+(?: [a-z]+){0,3})"),
    re.compile(r"[Ii] work as (?:a|an) ([a-z]+(?: [a-z]+){0,3})"),
    re.compile(r"[Ii](?:'m| am) studying ([^.!?\n]{3,60})"),
]


class MemoryController:
    """
    Coordinates the three memory tiers.

    Responsibilities:
      1. Add new turns to working memory.
      2. Extract personal facts from user messages → semantic memory.
      3. Consolidate working-memory turns into episodic episodes when
         a configurable threshold is reached.
      4. Build the enriched context string for the language model.

    Args:
        working_memory  : WorkingMemory instance.
        episodic_memory : EpisodicMemory instance.
        semantic_memory : SemanticMemory instance.
        consolidation_interval (int): Consolidate every N user turns.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        consolidation_interval: int = 5,
    ):
        self.wm = working_memory
        self.em = episodic_memory
        self.sm = semantic_memory
        self.consolidation_interval = consolidation_interval
        self._turn_counter: int = 0
        self._session_id: str = "default"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, session_id: str) -> None:
        """Set the current session identifier (used when storing episodes)."""
        self._session_id = session_id

    def process_user_turn(self, user_message: str) -> None:
        """Process a user turn: update working memory and extract semantic facts.

        Args:
            user_message: The raw user message text.
        """
        self.wm.add_turn("user", user_message)
        self._extract_and_store_facts(user_message)
        self._turn_counter += 1

        # Trigger consolidation at regular intervals
        if self._turn_counter % self.consolidation_interval == 0:
            self._consolidate()

    def process_assistant_turn(self, assistant_message: str) -> None:
        """Add the assistant response to working memory.

        Args:
            assistant_message: The raw assistant response text.
        """
        self.wm.add_turn("assistant", assistant_message)

    def build_context(self, current_query: str) -> str:
        """Build the enriched prompt context for the language model.

        Combines:
          • Semantic memory (user facts, always included if non-empty)
          • Retrieved episodic memories relevant to the current query
          • Recent working-memory turns (formatted as conversation history)

        Args:
            current_query: The current user query (used for episodic retrieval).

        Returns:
            A context string to prepend to the model prompt.
        """
        parts: List[str] = []

        # 1. Semantic memory (persistent user facts)
        semantic_ctx = self.sm.format_for_context()
        if semantic_ctx:
            parts.append(semantic_ctx)

        # 2. Episodic memory (relevant past interactions)
        episodes = self.em.retrieve(current_query)
        if episodes:
            ep_lines = ["[Episodic Memory – relevant past interactions]"]
            for ep in episodes:
                ep_lines.append(
                    f"  Session {ep['session_id']}, turn {ep['turn']}: {ep['summary']}"
                )
            parts.append("\n".join(ep_lines))

        return "\n\n".join(parts)

    def get_working_context(self) -> List[Dict[str, str]]:
        """Return the current working-memory message list."""
        return self.wm.get_context()

    def save_all(self, base_path: str) -> None:
        """Persist episodic and semantic memories to *base_path*."""
        self.em.save(os.path.join(base_path, "episodic"))
        self.sm.save(os.path.join(base_path, "semantic_memory.json"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_store_facts(self, text: str) -> None:
        """Run pattern matching to extract personal facts from *text*."""
        for pattern in _NAME_PATTERNS:
            m = pattern.search(text)
            if m:
                self.sm.store_fact("name", m.group(1).strip(), "personal")

        for pattern in _PREFERENCE_PATTERNS:
            m = pattern.search(text)
            if m:
                preference = m.group(1).strip().rstrip(".,;")
                # Use first 40 chars as the key for deduplication
                key = "preference_" + re.sub(r"\W+", "_", preference[:40]).strip("_")
                self.sm.store_fact(key, preference, "preferences")

        for pattern in _GOAL_PATTERNS:
            m = pattern.search(text)
            if m:
                goal = m.group(1).strip().rstrip(".,;")
                key = "goal_" + re.sub(r"\W+", "_", goal[:40]).strip("_")
                self.sm.store_fact(key, goal, "goals")

        for pattern in _OCCUPATION_PATTERNS:
            m = pattern.search(text)
            if m:
                occupation = m.group(1).strip().rstrip(".,;")
                self.sm.store_fact("occupation", occupation, "personal")

    def _consolidate(self) -> None:
        """Consolidate the current working-memory window into episodic memory."""
        turns = self.wm.get_context()
        if not turns:
            return

        # Build a raw text block from the turns
        text_parts = [
            f"{t['role'].capitalize()}: {t['content']}" for t in turns
        ]
        raw_text = "\n".join(text_parts)

        # Produce a simple extractive summary (first user message + length hint)
        user_msgs = [t["content"] for t in turns if t["role"] == "user"]
        summary = (
            f"[{len(turns)} turns] " + (user_msgs[0][:120] if user_msgs else "")
        )

        self.em.add_episode(
            text=raw_text,
            summary=summary,
            session_id=self._session_id,
            turn=self._turn_counter,
        )
