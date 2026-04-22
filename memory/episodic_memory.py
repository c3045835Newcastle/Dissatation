"""
Episodic Memory Module
======================

Stores past conversation turns as dense vector embeddings for semantic
retrieval.  Uses FAISS IndexFlatIP (inner-product / cosine similarity after
L2 normalisation) when available; falls back to NumPy brute-force search if
the ``faiss`` package is not installed.

Embeddings are computed with ``sentence-transformers/all-MiniLM-L6-v2``
(~90 MB) when sentence-transformers is available; otherwise a fallback
TF-IDF-style bag-of-words is used so the module remains importable without
the optional dependency.

Reference: Lewis, P. et al. (2020). Retrieval-augmented generation for
knowledge-intensive NLP tasks. NeurIPS 2020.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional heavy dependencies
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lightweight fallback vectoriser (TF bag-of-words, no external deps)
# ---------------------------------------------------------------------------

class _BagOfWordsVectoriser:
    """Minimal bag-of-words vectoriser used when sentence-transformers is absent."""

    DIM = 256

    def __init__(self):
        self._vocab: Dict[str, int] = {}

    def _hash_token(self, token: str) -> int:
        return hash(token) % self.DIM

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        vecs = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for tok in tokens:
                vecs[i, self._hash_token(tok)] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Vector-store episodic memory backed by FAISS (or NumPy fallback).

    Each stored entry has metadata::

        {
          "idx":       int,   # position in the flat index
          "text":      str,   # turn text
          "role":      str,   # "user" | "assistant"
          "turn_id":   int,   # original turn number
          "timestamp": float, # Unix timestamp
          "topic":     str,   # optional topic label
        }

    The index uses cosine similarity (inner product after normalisation).
    """

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM_ST = 384
    EMBEDDING_DIM_BOW = _BagOfWordsVectoriser.DIM

    def __init__(
        self,
        persist_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Args:
            persist_path:    Directory for on-disk persistence.  ``None`` = in-memory only.
            embedding_model: SentenceTransformer model name (default: all-MiniLM-L6-v2).
        """
        self.persist_path: Optional[Path] = Path(persist_path) if persist_path else None
        self.embedding_model_name = embedding_model or self.EMBEDDING_MODEL

        self._encoder = None
        self._dim: int = 0
        self._index = None                   # FAISS index or None
        self._embeddings: List[np.ndarray] = []  # fallback list for NumPy search
        self._metadata: List[Dict] = []

        self._init_encoder()
        self._init_index()

        if self.persist_path and (self.persist_path / "episodic_index.faiss").exists():
            self._load()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_encoder(self):
        if _ST_AVAILABLE:
            try:
                self._encoder = SentenceTransformer(
                    self.embedding_model_name,
                    local_files_only=False,
                )
                self._dim = self.EMBEDDING_DIM_ST
                return
            except Exception:
                # Network unavailable or model not cached — fall through to BOW
                pass
        self._encoder = _BagOfWordsVectoriser()
        self._dim = self.EMBEDDING_DIM_BOW

    def _init_index(self):
        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self._dim)
            self._embeddings = []
        else:
            self._index = None
            self._embeddings = []

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embedding matrix of shape (len(texts), dim)."""
        vecs = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.array(vecs, dtype=np.float32)
        # Normalise manually if the encoder didn't do it (e.g. BOW fallback)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        role: str,
        turn_id: int,
        timestamp: Optional[float] = None,
        topic: str = "",
    ) -> int:
        """
        Add a single turn entry.

        Returns:
            Index of the stored entry.
        """
        vec = self._embed([text])  # (1, dim)

        if _FAISS_AVAILABLE and self._index is not None:
            self._index.add(vec)
        else:
            self._embeddings.append(vec[0])

        idx = len(self._metadata)
        self._metadata.append(
            {
                "idx": idx,
                "text": text,
                "role": role,
                "turn_id": turn_id,
                "timestamp": timestamp or time.time(),
                "topic": topic,
            }
        )

        if self.persist_path:
            self._save()

        return idx

    def add_turns(self, turns: List[Dict]):
        """Add multiple evicted working-memory turns in batch."""
        for turn in turns:
            self.add(
                text=turn["content"],
                role=turn["role"],
                turn_id=turn.get("turn_id", 0),
                timestamp=turn.get("timestamp"),
            )

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        min_score: float = 0.25,
    ) -> List[Dict]:
        """
        Retrieve the ``top_k`` most semantically relevant stored turns.

        Args:
            query_text: Current user message used as the search query.
            top_k:      Maximum number of results to return.
            min_score:  Minimum cosine similarity threshold (0–1).

        Returns:
            List of metadata dicts, each augmented with ``"retrieval_score"``,
            sorted by descending similarity.
        """
        if not self._metadata:
            return []

        q_vec = self._embed([query_text])
        effective_k = min(top_k, len(self._metadata))

        if _FAISS_AVAILABLE and self._index is not None:
            scores_arr, idx_arr = self._index.search(q_vec, effective_k)
            scores = scores_arr[0].tolist()
            indices = idx_arr[0].tolist()
        else:
            # NumPy cosine similarity
            emb_matrix = np.stack(self._embeddings)  # (n, dim)
            sims = emb_matrix @ q_vec[0]             # (n,)
            top_indices = np.argsort(sims)[::-1][:effective_k]
            indices = top_indices.tolist()
            scores = sims[top_indices].tolist()

        results: List[Dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            if score < min_score:
                continue
            meta = dict(self._metadata[idx])
            meta["retrieval_score"] = float(score)
            results.append(meta)

        return sorted(results, key=lambda x: x["retrieval_score"], reverse=True)

    def clear(self):
        """Remove all stored entries."""
        self._init_index()
        self._metadata.clear()
        self._embeddings.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        self.persist_path.mkdir(parents=True, exist_ok=True)
        if _FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, str(self.persist_path / "episodic_index.faiss"))
        elif self._embeddings:
            np.save(str(self.persist_path / "episodic_embeddings.npy"), np.array(self._embeddings))
        with open(self.persist_path / "episodic_metadata.json", "w") as fh:
            json.dump(self._metadata, fh, indent=2)

    def _load(self):
        if _FAISS_AVAILABLE:
            idx_path = self.persist_path / "episodic_index.faiss"
            if idx_path.exists():
                self._index = faiss.read_index(str(idx_path))
        else:
            emb_path = self.persist_path / "episodic_embeddings.npy"
            if emb_path.exists():
                self._embeddings = list(np.load(str(emb_path)))

        meta_path = self.persist_path / "episodic_metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                self._metadata = json.load(fh)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._metadata)

    @property
    def using_faiss(self) -> bool:
        return _FAISS_AVAILABLE and self._index is not None

    @property
    def using_sentence_transformers(self) -> bool:
        return _ST_AVAILABLE
