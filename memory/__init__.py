"""
memory — Hierarchical Memory Architecture for LLM Dialogue Systems
===================================================================

Three-layer memory system inspired by Baddeley (1992) cognitive memory model:

  WorkingMemory    — sliding-window recent context (last N turn-pairs)
  EpisodicMemory   — past interactions stored as dense vector embeddings (FAISS)
  SemanticMemory   — persistent structured facts about the user
  MemoryController — orchestrates all three layers

Usage::

    from memory.memory_controller import MemoryController

    mc = MemoryController()
    mc.add_user_turn("My name is Alice and I'm a PhD student.")
    context_messages, n_episodic = mc.build_context("What should I study next?")
"""

from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.memory_controller import MemoryController

__all__ = ["WorkingMemory", "EpisodicMemory", "SemanticMemory", "MemoryController"]
