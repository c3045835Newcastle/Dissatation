"""
Hierarchical Memory Architecture for LLM Dialogue Systems.

Implements a three-tier memory system:
  - Working memory  : recent conversation context (sliding window)
  - Episodic memory : past interactions retrieved via FAISS vector search
  - Semantic memory : persistent user facts stored as structured data

References:
  Baddeley (1992)  – cognitive model of human memory
  Lewis et al. (2020) – retrieval-augmented generation
  Johnson et al. (2017) – FAISS billion-scale similarity search
"""

from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .memory_controller import MemoryController

__all__ = [
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "MemoryController",
]
