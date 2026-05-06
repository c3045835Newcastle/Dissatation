"""
Episodic Memory module.

Stores past interaction summaries and retrieves them via FAISS vector
similarity search (Johnson et al., 2017).  A lightweight sentence-transformer
(all-MiniLM-L6-v2) provides the embeddings.
"""

import json
import os
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class EpisodicMemory:
    """
    Episodic memory backed by a FAISS flat L2 index.

    Each stored episode contains:
      - text   : the original interaction text (user + assistant turn)
      - summary: a short summary written by the memory controller
      - session: the session identifier
      - turn   : the dialogue-turn index within that session

    When *faiss* or *sentence-transformers* are not installed the class
    degrades gracefully to linear keyword search so that the rest of the
    system remains importable and testable.

    Args:
        embedding_model (str): Sentence-transformer model name.
        top_k           (int): Default number of episodes to retrieve.
        storage_path   (str): Optional path to persist the index on disk.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = 5,
        storage_path: Optional[str] = None,
    ):
        self.top_k = top_k
        self.storage_path = storage_path
        self._episodes: List[Dict] = []

        # Embedding model
        if _ST_AVAILABLE:
            try:
                self._encoder = SentenceTransformer(embedding_model)
                self._dim = self._encoder.get_sentence_embedding_dimension()
            except Exception:
                self._encoder = None
                self._dim = self.EMBEDDING_DIM
        else:
            self._encoder = None
            self._dim = self.EMBEDDING_DIM

        # FAISS index
        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatL2(self._dim)
        else:
            self._index = None

        # Load from disk if available
        if storage_path:
            self._load(storage_path)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_episode(
        self,
        text: str,
        summary: str,
        session_id: str = "default",
        turn: int = 0,
    ) -> None:
        """Store a new episode and update the FAISS index.

        Args:
            text      : Raw interaction text.
            summary   : Short summary of the interaction.
            session_id: Identifier of the current dialogue session.
            turn      : Turn index within the session.
        """
        episode = {
            "text": text,
            "summary": summary,
            "session_id": session_id,
            "turn": turn,
        }
        self._episodes.append(episode)

        if self._encoder is not None and self._index is not None:
            vec = self._encode(summary)
            self._index.add(vec)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict]:
        """Retrieve the *top_k* most relevant episodes for *query*.

        Args:
            query : The query string (usually the current user message).
            top_k : Number of results to return; defaults to self.top_k.

        Returns:
            List of episode dicts ordered by relevance (most relevant first).
        """
        k = top_k or self.top_k
        if not self._episodes:
            return []

        k = min(k, len(self._episodes))

        if self._encoder is not None and self._index is not None:
            vec = self._encode(query)
            _distances, indices = self._index.search(vec, k)  # type: ignore
            return [self._episodes[i] for i in indices[0] if i < len(self._episodes)]

        # Fallback: simple keyword matching
        query_lower = query.lower()
        scored = []
        for ep in self._episodes:
            score = sum(
                1 for word in query_lower.split()
                if word in ep["summary"].lower() or word in ep["text"].lower()
            )
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist episodes and FAISS index to *path* (or self.storage_path)."""
        target = path or self.storage_path
        if not target:
            return
        os.makedirs(target, exist_ok=True)

        with open(os.path.join(target, "episodes.json"), "w") as fh:
            json.dump(self._episodes, fh, indent=2)

        if self._index is not None and _FAISS_AVAILABLE:
            faiss.write_index(
                self._index, os.path.join(target, "faiss.index")
            )

    def _load(self, path: str) -> None:
        episodes_path = os.path.join(path, "episodes.json")
        index_path = os.path.join(path, "faiss.index")

        if os.path.exists(episodes_path):
            with open(episodes_path) as fh:
                self._episodes = json.load(fh)

        if _FAISS_AVAILABLE and os.path.exists(index_path):
            self._index = faiss.read_index(index_path)
        elif _FAISS_AVAILABLE and self._episodes:
            # Rebuild index from stored episodes
            self._index = faiss.IndexFlatL2(self._dim)
            if self._encoder is not None:
                summaries = [ep["summary"] for ep in self._episodes]
                vecs = self._encoder.encode(
                    summaries, normalize_embeddings=False, show_progress_bar=False
                ).astype(np.float32)
                self._index.add(vecs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        """Encode *text* to a (1, dim) float32 numpy array."""
        vec = self._encoder.encode(
            [text], normalize_embeddings=False, show_progress_bar=False
        )
        return vec.astype(np.float32)

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(episodes={len(self._episodes)}, "
            f"faiss={'yes' if _FAISS_AVAILABLE else 'no (fallback)'})"
        )
