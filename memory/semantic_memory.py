"""
Semantic Memory Module
======================

Stores persistent, structured facts about the user and session context as
key-value pairs.  Facts are automatically extracted from user utterances via
lightweight regex heuristics and optionally persisted to disk as JSON.

Semantic memory persists **across sessions**, unlike episodic memory which
is reset at ``new_session()`` boundaries.

Reference: Baddeley, A.D. (1992). Working memory. Science, 255(5044), 556-559.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SemanticMemory:
    """
    Persistent key-value store for structured user facts.

    Facts are grouped by *category* (e.g. ``"user_name"``, ``"preference"``)
    and stored as ``{category: {key: value}}`` nested dicts.

    Example::

        sm = SemanticMemory()
        sm.extract_and_store("My name is Alice and I am a PhD student.")
        print(sm.format_for_context())
        # [User Profile & Persistent Facts]
        #   - user_name: alice
        #   - user_role: a phd student
    """

    # Heuristic regex patterns for automatic fact extraction.
    # Each entry: (compiled pattern, category_label)
    _FACT_PATTERNS: List[Tuple] = [
        (re.compile(r"(?:my name is|i(?:'m| am) called)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)", re.I), "user_name"),
        (re.compile(r"i(?:'m| am)\s+(\d{1,3})\s+years?\s+old", re.I), "user_age"),
        (re.compile(r"i(?:'m| am)\s+a(?:n)?\s+([\w\s]{3,40}?)(?:\.|,|\s+and\s|$)", re.I), "user_role"),
        (re.compile(r"i\s+work\s+as\s+(?:a(?:n)?\s+)?([\w\s]{3,40}?)(?:\.|,|$)", re.I), "user_occupation"),
        (re.compile(r"i\s+prefer\s+([\w\s]{2,60}?)(?:\.|,|$)", re.I), "preference"),
        (re.compile(r"i\s+like\s+([\w\s]{2,40}?)(?:\.|,| and|$)", re.I), "preference"),
        (re.compile(r"i\s+(?:don'?t\s+like|dislike|hate)\s+([\w\s]{2,40}?)(?:\.|,|$)", re.I), "dislike"),
        (re.compile(r"my\s+(?:favourite|favorite)\s+[\w]+\s+is\s+([\w\s]{2,40}?)(?:\.|,|$)", re.I), "favourite"),
        # Injected facts ("The code/password/key is X")
        (re.compile(r"(?:the\s+)?(?:code|password|key|access code)\s+(?:is|=)\s+(\w+)", re.I), "injected_code"),
        (re.compile(r"(?:the\s+)?(?:deadline|due date)\s+is\s+([\w\s,]+?)(?:\.|$)", re.I), "deadline"),
    ]

    def __init__(self, persist_path: Optional[str] = None):
        """
        Args:
            persist_path: Directory for on-disk persistence.  ``None`` = in-memory only.
        """
        self._facts: Dict[str, Dict[str, str]] = {}
        self.persist_path: Optional[Path] = Path(persist_path) if persist_path else None

        if self.persist_path and (self.persist_path / "semantic_memory.json").exists():
            self._load()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def store(self, category: str, key: str, value: str):
        """Manually store a fact."""
        if category not in self._facts:
            self._facts[category] = {}
        self._facts[category][key] = value
        if self.persist_path:
            self._save()

    def extract_and_store(self, text: str, source_turn_id: int = 0) -> List[Tuple[str, str, str]]:
        """
        Heuristically extract facts from a user utterance and store them.

        Args:
            text:           The user message to analyse.
            source_turn_id: Turn number for generating unique fact keys.

        Returns:
            List of ``(category, key, value)`` tuples for the extracted facts.
        """
        # Common words that should never be the end of an extracted value
        _TRAILING_NOISE = re.compile(
            r"\s+(?:and|or|but|the|a|an|is|was|are|were|have|had|so)\s*$", re.I
        )

        extracted: List[Tuple[str, str, str]] = []
        for pattern, category in self._FACT_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip().lower()
                # Strip trailing function words (e.g. "alice and" → "alice")
                value = _TRAILING_NOISE.sub("", value).strip()
                if 2 <= len(value) <= 80:
                    key = f"{category}_{source_turn_id}"
                    self.store(category, key, value)
                    extracted.append((category, key, value))
        return extracted

    def get_all_facts(self) -> Dict[str, Dict[str, str]]:
        """Return a copy of all stored facts."""
        return {k: dict(v) for k, v in self._facts.items()}

    def get_category(self, category: str) -> Dict[str, str]:
        """Return all facts in a given category."""
        return dict(self._facts.get(category, {}))

    def format_for_context(self) -> str:
        """
        Format all stored facts into a compact context string suitable for
        prepending to model prompts.

        Returns empty string if no facts are stored.
        """
        if not self._facts:
            return ""
        lines = ["[User Profile & Persistent Facts]"]
        for category, entries in self._facts.items():
            for key, value in entries.items():
                lines.append(f"  - {category}: {value}")
        return "\n".join(lines)

    def clear(self):
        """Remove all stored facts."""
        self._facts.clear()
        if self.persist_path:
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        self.persist_path.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path / "semantic_memory.json", "w") as fh:
            json.dump(self._facts, fh, indent=2)

    def _load(self):
        with open(self.persist_path / "semantic_memory.json") as fh:
            self._facts = json.load(fh)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(v) for v in self._facts.values())
