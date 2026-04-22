"""
Memory Controller
=================

Orchestrates all three memory layers (Working, Episodic, Semantic) and
exposes a single high-level API that the dialogue system calls:

1.  ``add_user_turn(text)``      — record user message, extract facts, evict old
                                   working-memory turns into episodic store.
2.  ``add_assistant_turn(text)`` — record assistant reply, evict if needed.
3.  ``build_context(query)``     — compose the enriched message list to inject
                                   into the model: semantic facts + retrieved
                                   episodic memories + recent working memory.
4.  ``reset_session()``          — flush working memory for a new conversation
                                   session while preserving episodic/semantic.
5.  ``reset_all()``              — wipe all memory layers (between eval runs).
"""

from typing import Dict, List, Optional, Tuple

from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory


class MemoryController:
    """
    Central memory controller for the hierarchical memory architecture.

    Parameters
    ----------
    working_memory_turns : int
        Maximum turn-*pairs* retained in working memory before eviction.
    episodic_top_k : int
        Maximum number of episodic entries retrieved per query.
    episodic_min_score : float
        Minimum cosine similarity for an episodic entry to be included.
    persist_path : str or None
        Directory for on-disk persistence of episodic and semantic memory.
        ``None`` = in-memory only (default, suitable for evaluation runs).
    """

    def __init__(
        self,
        working_memory_turns: int = 6,
        episodic_top_k: int = 3,
        episodic_min_score: float = 0.25,
        persist_path: Optional[str] = None,
    ):
        self.working_memory = WorkingMemory(max_turns=working_memory_turns)
        self.episodic_memory = EpisodicMemory(persist_path=persist_path)
        self.semantic_memory = SemanticMemory(persist_path=persist_path)

        self.episodic_top_k = episodic_top_k
        self.episodic_min_score = episodic_min_score
        self._total_turns: int = 0

    # ------------------------------------------------------------------
    # Recording turns
    # ------------------------------------------------------------------

    def add_user_turn(self, text: str) -> List[Dict]:
        """
        Record a user turn.

        Side effects:
        - Extracts facts into semantic memory.
        - Adds turn to working memory; consolidates evicted turns into episodic.

        Returns:
            List of evicted turn dicts (may be empty).
        """
        self._total_turns += 1
        self.semantic_memory.extract_and_store(text, source_turn_id=self._total_turns)
        evicted = self.working_memory.add_turn("user", text)
        if evicted:
            self.episodic_memory.add_turns(evicted)
        return evicted

    def add_assistant_turn(self, text: str) -> List[Dict]:
        """
        Record an assistant turn.

        Returns:
            List of evicted turn dicts (may be empty).
        """
        evicted = self.working_memory.add_turn("assistant", text)
        if evicted:
            self.episodic_memory.add_turns(evicted)
        return evicted

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def build_context(
        self,
        user_query: str,
        system_prompt: str = "",
    ) -> Tuple[List[Dict], int]:
        """
        Compose the enriched message list for the model.

        The result is a list of ``{"role": ..., "content": ...}`` dicts that
        the caller should pass (after appending the current user message) to
        the tokeniser's chat template.

        Layout
        ------
        1. System turn: ``system_prompt`` + ``[User Profile & Persistent Facts]``
           (only present if either has content)
        2. System turn: ``[Relevant past conversation excerpts …]``
           (only present if episodic retrieval found results)
        3. Working-memory turns (user/assistant, in order)

        Returns:
            (messages, n_episodic_hits)
        """
        messages: List[Dict] = []

        # --- 1. Semantic preamble ---
        semantic_str = self.semantic_memory.format_for_context()
        preamble_parts: List[str] = []
        if system_prompt:
            preamble_parts.append(system_prompt)
        if semantic_str:
            preamble_parts.append(semantic_str)
        if preamble_parts:
            messages.append({"role": "system", "content": "\n\n".join(preamble_parts)})

        # --- 2. Episodic retrieval ---
        retrieved = self.episodic_memory.query(
            user_query,
            top_k=self.episodic_top_k,
            min_score=self.episodic_min_score,
        )
        if retrieved:
            lines = ["[Relevant past conversation excerpts retrieved from episodic memory]"]
            for item in retrieved:
                # Truncate long entries to avoid inflating context unnecessarily
                excerpt = item["text"][:250]
                lines.append(f"  [{item['role']}] (turn {item['turn_id']}) {excerpt}")
            messages.append({"role": "system", "content": "\n".join(lines)})

        # --- 3. Recent working memory ---
        messages.extend(self.working_memory.get_context_messages())

        return messages, len(retrieved)

    # ------------------------------------------------------------------
    # Session / reset management
    # ------------------------------------------------------------------

    def reset_session(self):
        """
        Start a new conversation session.
        Clears working memory; episodic and semantic memory persist.
        """
        self.working_memory.clear()

    def reset_all(self):
        """
        Fully reset all memory layers.
        Use between independent evaluation scenarios to prevent leakage.
        """
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self._total_turns = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict:
        """Return a summary of current memory state for diagnostics."""
        return {
            "working_memory_messages": len(self.working_memory),
            "episodic_memory_entries": len(self.episodic_memory),
            "semantic_memory_facts": len(self.semantic_memory),
            "total_turns": self._total_turns,
        }
