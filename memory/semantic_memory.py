"""
Semantic Memory module.

Stores persistent user facts and domain knowledge as structured key-value
data.  Inspired by the semantic memory component of Baddeley's (1992)
cognitive memory model: long-term storage of general world knowledge and
personal facts that should persist across sessions.

Objective 2c: Semantic memory for storing persistent user information.
"""

import json
import os
from typing import Any, Dict, List, Optional


class SemanticMemory:
    """
    Key-value store for persistent user facts.

    Facts are organised into *categories* (e.g. ``'personal'``,
    ``'preferences'``, ``'goals'``) to allow targeted retrieval.  The store
    is backed by a JSON file so that information persists across sessions.

    Args:
        storage_path (str): Path to the JSON file used for persistence.
                            Pass ``None`` to keep memory in-process only.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._facts: Dict[str, Dict[str, Any]] = {}

        if storage_path and os.path.exists(storage_path):
            self._load(storage_path)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def store_fact(self, key: str, value: Any, category: str = "general") -> None:
        """Store or update a fact.

        Args:
            key      : Fact identifier (e.g. ``'user_name'``).
            value    : The fact value.
            category : Logical grouping (e.g. ``'personal'``, ``'preferences'``).
        """
        if category not in self._facts:
            self._facts[category] = {}
        self._facts[category][key] = value

    def remove_fact(self, key: str, category: str = "general") -> bool:
        """Remove a fact.  Returns ``True`` if the fact existed."""
        if category in self._facts and key in self._facts[category]:
            del self._facts[category][key]
            return True
        return False

    def clear_category(self, category: str) -> None:
        """Remove all facts in *category*."""
        self._facts.pop(category, None)

    def clear(self) -> None:
        """Remove all stored facts."""
        self._facts.clear()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_fact(self, key: str, category: str = "general") -> Optional[Any]:
        """Retrieve a single fact by *key* and *category*."""
        return self._facts.get(category, {}).get(key)

    def get_category(self, category: str) -> Dict[str, Any]:
        """Return all facts in *category* as a dict."""
        return dict(self._facts.get(category, {}))

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Return all stored facts organised by category."""
        return {cat: dict(facts) for cat, facts in self._facts.items()}

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Return facts whose key or string value contains *query* (case-insensitive).

        Returns a list of dicts: ``[{'category': ..., 'key': ..., 'value': ...}]``
        """
        query_lower = query.lower()
        results = []
        for category, facts in self._facts.items():
            for key, value in facts.items():
                if query_lower in key.lower() or query_lower in str(value).lower():
                    results.append({"category": category, "key": key, "value": value})
        return results

    def format_for_context(self) -> str:
        """Return a human-readable summary suitable for injection into a prompt."""
        if not self._facts:
            return ""
        lines = ["[Semantic Memory – known facts about the user]"]
        for category, facts in self._facts.items():
            if facts:
                lines.append(f"  {category.capitalize()}:")
                for key, value in facts.items():
                    lines.append(f"    • {key}: {value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Write the fact store to *path* (JSON)."""
        target = path or self.storage_path
        if not target:
            return
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w") as fh:
            json.dump(self._facts, fh, indent=2)

    def _load(self, path: str) -> None:
        with open(path) as fh:
            self._facts = json.load(fh)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(facts) for facts in self._facts.values())

    def __repr__(self) -> str:
        n_categories = len(self._facts)
        n_facts = len(self)
        return f"SemanticMemory(categories={n_categories}, facts={n_facts})"
