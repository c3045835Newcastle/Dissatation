"""
Hierarchical Memory Dialogue Pipeline.

Integrates the three-tier memory architecture (working / episodic / semantic)
with the base Llama 3.1 8B model to produce a dialogue system that maintains
long-term coherence across sessions.

References:
  Lewis et al. (2020) – retrieval-augmented generation pattern.
"""

import os
from typing import Dict, List, Optional

import config
from memory import EpisodicMemory, MemoryController, SemanticMemory, WorkingMemory


class HierarchicalMemoryPipeline:
    """
    Dialogue pipeline with hierarchical memory.

    The pipeline wraps the base language model and enriches every request
    with relevant context assembled by the MemoryController:

      1. Semantic memory  → user facts always prepended to the system prompt
      2. Episodic memory  → top-k relevant past interactions retrieved via FAISS
      3. Working memory   → the sliding window of the current conversation

    The model itself is loaded lazily to keep the class lightweight for
    offline testing.

    Args:
        session_id        (str) : Identifier for the current dialogue session.
        memory_base_path  (str) : Root directory for persisting memory stores.
        max_working_turns (int) : Sliding-window size for working memory.
        episodic_top_k    (int) : Episodes retrieved per query.
        load_model        (bool): Whether to load the LLM immediately.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant with a persistent memory. "
        "Use any user facts and past interactions provided below to give "
        "coherent, consistent, and accurate responses across multiple sessions."
    )

    def __init__(
        self,
        session_id: str = "session_1",
        memory_base_path: str = "./memory_store",
        max_working_turns: int = config.WORKING_MEMORY_MAX_TURNS,
        episodic_top_k: int = config.EPISODIC_TOP_K,
        load_model: bool = False,
    ):
        self.session_id = session_id
        self.memory_base_path = memory_base_path

        # Instantiate the three memory tiers
        episodic_path = os.path.join(memory_base_path, "episodic")
        semantic_path = os.path.join(memory_base_path, "semantic_memory.json")

        self.working_memory = WorkingMemory(max_turns=max_working_turns)
        self.episodic_memory = EpisodicMemory(
            top_k=episodic_top_k,
            storage_path=episodic_path if os.path.isdir(episodic_path) else None,
        )
        self.semantic_memory = SemanticMemory(
            storage_path=semantic_path if os.path.exists(semantic_path) else None,
        )

        # Memory controller orchestrates all three tiers
        self.controller = MemoryController(
            working_memory=self.working_memory,
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            consolidation_interval=config.MEMORY_CONSOLIDATION_INTERVAL,
        )
        self.controller.set_session(session_id)

        # LLM (loaded on demand)
        self._model = None
        if load_model:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process one user turn and return the assistant response.

        Args:
            user_message: The user's input text.

        Returns:
            The assistant's response string.
        """
        # Step 1 – update working memory and extract semantic facts
        self.controller.process_user_turn(user_message)

        # Step 2 – build enriched context from all memory tiers
        memory_context = self.controller.build_context(user_message)

        # Step 3 – assemble message list for the LLM
        messages = self._build_messages(user_message, memory_context)

        # Step 4 – generate response (or use stub when model not loaded)
        response = self._generate(messages)

        # Step 5 – update working memory with assistant response
        self.controller.process_assistant_turn(response)

        return response

    def save_memory(self) -> None:
        """Persist all memory stores to disk."""
        os.makedirs(self.memory_base_path, exist_ok=True)
        self.controller.save_all(self.memory_base_path)

    def get_memory_summary(self) -> Dict:
        """Return a summary of all current memory contents."""
        return {
            "session_id": self.session_id,
            "working_memory_turns": len(self.working_memory),
            "episodic_episodes": len(self.episodic_memory),
            "semantic_facts": len(self.semantic_memory),
            "semantic_details": self.semantic_memory.get_all(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self, user_message: str, memory_context: str
    ) -> List[Dict[str, str]]:
        """Assemble the message list sent to the LLM.

        Layout:
          [system prompt + memory context]
          ... working-memory turns (all except the latest user turn) ...
          [latest user turn]
        """
        system_content = self.SYSTEM_PROMPT
        if memory_context:
            system_content += "\n\n" + memory_context

        messages = [{"role": "system", "content": system_content}]

        # Working memory contains the full history including the just-added
        # user message; we include all but the last entry here, then append
        # the user message explicitly.
        wm_turns = self.working_memory.get_context()
        if len(wm_turns) > 1:
            messages.extend(wm_turns[:-1])

        messages.append({"role": "user", "content": user_message})
        return messages

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the loaded model (or return a stub)."""
        if self._model is None:
            return (
                "[Model not loaded – call HierarchicalMemoryPipeline("
                "load_model=True) to enable LLM responses]"
            )
        return self._model.chat(messages)

    def _load_model(self) -> None:
        """Lazily load the base Llama model."""
        from llama_base_model import BaseLlama31Model
        print("Loading Llama 3.1 8B model for hierarchical memory pipeline…")
        self._model = BaseLlama31Model()
        print("Model loaded.")
