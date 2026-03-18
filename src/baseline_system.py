"""
Baseline Dialogue System – LLaMA 3.1 8B with standard context-window memory.

This module implements the baseline system described in the dissertation:
  "Design and Evaluation of a Hierarchical Memory Architecture for Improving
   Long-Term Coherence in Local Large Language Model Dialogue Systems"
  Robin Husbands, 230458358

The baseline system uses ONLY the model's fixed-length context window for
memory.  No external memory store, no vector retrieval.  Information that
falls outside the context window is permanently lost.

When an actual Ollama/LLaMA endpoint is available the system can call it via
the Ollama REST API.  For offline evaluation (and poster data generation) a
SimulatedLLM class reproduces the known degradation patterns of a context-
window-only system so that realistic benchmark figures can be produced.
"""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    turn: int = 0
    session: int = 0

    def token_estimate(self) -> int:
        """Rough token count (~4 chars per token)."""
        return max(1, len(self.content) // 4)


@dataclass
class ConversationState:
    session_id: int = 0
    turn: int = 0
    context_window: List[Message] = field(default_factory=list)
    context_token_limit: int = 4096  # LLaMA 3.1 8B default
    tokens_used: int = 0
    facts_introduced: List[dict] = field(default_factory=list)
    facts_recalled: int = 0
    facts_forgotten: int = 0
    hallucinations: int = 0
    total_recall_attempts: int = 0

    def add_message(self, msg: Message) -> None:
        """Add a message, evicting the oldest non-system message if context window is full."""
        self.context_window.append(msg)
        self.tokens_used += msg.token_estimate()
        # Evict oldest non-system messages until we're back under the limit
        while self.tokens_used > self.context_token_limit and len(self.context_window) > 1:
            evicted_idx = next(
                (i for i, m in enumerate(self.context_window) if m.role != "system"),
                None,
            )
            if evicted_idx is None:
                break  # Only the system prompt remains; cannot evict further
            evicted = self.context_window.pop(evicted_idx)
            self.tokens_used -= evicted.token_estimate()

    def context_contains_fact(self, fact_key: str, fact_value: str = "") -> bool:
        """
        Return True only if the fact *value* was explicitly introduced and
        has not yet been evicted from the context window.

        Checking for the value (not the key) prevents false positives caused
        by the probe query itself containing the fact key word
        (e.g. "What is my name?" contains the key "name").
        """
        combined = " ".join(m.content.lower() for m in self.context_window)
        if fact_value:
            return fact_value.lower() in combined
        return fact_key.lower() in combined

    def new_session(self) -> None:
        """Simulate starting a new chat session (context window cleared)."""
        self.session_id += 1
        self.turn = 0
        self.context_window = []
        self.tokens_used = 0


# ---------------------------------------------------------------------------
# Simulated LLM – reproduces baseline degradation without a running model
# ---------------------------------------------------------------------------

class SimulatedLLM:
    """
    Reproduces the memory-degradation behaviour of a context-window-only
    LLaMA 3.1 8B model.

    Rules:
      - If the fact keyword is present in the context window  → recall it
        correctly (with small random noise for realism).
      - If the fact keyword is absent (evicted / new session) → either
        hallucinate a plausible-sounding wrong answer OR say "I don't recall".
    """

    HALLUCINATION_PROBABILITY = 0.38   # 38 % of forgotten-fact queries → hallucination
    RECALL_NOISE_PROBABILITY  = 0.08   # 8 % chance of partial/noisy recall even when in context

    def __init__(self, seed: int = 42):
        import random
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def generate(self, state: ConversationState, user_query: str,
                 fact_probe: Optional[dict] = None) -> str:
        """
        Generate a response.

        Parameters
        ----------
        state       : current conversation state
        user_query  : the user's message text
        fact_probe  : if provided, dict with 'key' and 'value' representing
                      the fact being tested for recall
        """
        if fact_probe is None:
            return self._generic_response(user_query)

        key   = fact_probe["key"]
        value = fact_probe["value"]

        if state.context_contains_fact(key, value):
            # Fact is in context – recall it (with small noise)
            if self._rng.random() < self.RECALL_NOISE_PROBABILITY:
                return self._noisy_recall(key, value)
            return f"Based on our conversation, {key} is {value}."
        else:
            # Fact evicted / session cleared
            if self._rng.random() < self.HALLUCINATION_PROBABILITY:
                return self._hallucinate(key, value)
            return f"I'm sorry, I don't have any record of {key} in our conversation."

    def _generic_response(self, query: str) -> str:
        responses = [
            "I understand. Could you tell me more?",
            "That's an interesting point. Let me think about it.",
            "I see. How can I help you further?",
            "Thank you for sharing that information.",
        ]
        return self._rng.choice(responses)

    def _noisy_recall(self, key: str, value: str) -> str:
        fragments = value.split()
        if len(fragments) > 1 and self._rng.random() < 0.5:
            partial = " ".join(fragments[: max(1, len(fragments) // 2)])
            return f"I believe {key} is {partial}… though I'm not entirely certain."
        return f"I think {key} is {value}, if I recall correctly."

    def _hallucinate(self, key: str, value: str) -> str:
        """Produce a plausible-sounding but incorrect answer."""
        false_values = {
            "name": ["Alex", "Jordan", "Sam", "Chris", "Morgan"],
            "age":  ["28", "34", "41", "25", "52"],
            "city": ["Manchester", "Edinburgh", "Bristol", "Leeds", "Liverpool"],
            "job":  ["engineer", "teacher", "nurse", "accountant", "developer"],
            "pet":  ["a dog", "a rabbit", "a hamster", "a fish"],
            "favourite colour": ["blue", "green", "red", "purple"],
            "university": ["Durham", "York", "Sheffield", "Leeds", "Edinburgh"],
        }
        for hint, options in false_values.items():
            if hint in key.lower():
                wrong = self._rng.choice([o for o in options if o.lower() != value.lower()] or options)
                return f"I believe {key} is {wrong}."
        # Generic hallucination
        return f"From what I recall, {key} is {value[::-1]}."  # reversed string as obviously wrong


# ---------------------------------------------------------------------------
# Baseline System
# ---------------------------------------------------------------------------

class BaselineDialogueSystem:
    """
    Baseline system: single context window, no external memory.
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a helpful conversational assistant.
        You can only remember information that has been provided earlier in
        this conversation. You do not have access to any external memory.
    """).strip()

    def __init__(self, context_limit: int = 4096, llm: Optional[SimulatedLLM] = None):
        self.context_limit = context_limit
        self.llm = llm or SimulatedLLM()
        self.state = ConversationState(context_token_limit=context_limit)
        self._init_context()

    def _init_context(self) -> None:
        system_msg = Message(role="system", content=self.SYSTEM_PROMPT,
                             session=self.state.session_id)
        self.state.add_message(system_msg)

    def chat(self, user_input: str, fact_probe: Optional[dict] = None) -> str:
        """Process one turn of conversation."""
        self.state.turn += 1
        user_msg = Message(role="user", content=user_input,
                           turn=self.state.turn, session=self.state.session_id)
        self.state.add_message(user_msg)

        response = self.llm.generate(self.state, user_input, fact_probe)

        assistant_msg = Message(role="assistant", content=response,
                                turn=self.state.turn, session=self.state.session_id)
        self.state.add_message(assistant_msg)

        return response

    def introduce_fact(self, fact: dict) -> None:
        """Tell the system a fact (stores in context window)."""
        statement = f"My {fact['key']} is {fact['value']}."
        self.chat(statement)
        self.state.facts_introduced.append({**fact, "turn": self.state.turn,
                                             "session": self.state.session_id})

    def test_recall(self, fact: dict) -> dict:
        """
        Ask the system to recall a fact and record whether it succeeded.

        Returns a result dict with: fact, response, outcome.
        """
        query = f"What is my {fact['key']}?"
        response = self.chat(query, fact_probe=fact)
        self.state.total_recall_attempts += 1

        in_context = self.state.context_contains_fact(fact["key"], fact["value"])
        correct_value = fact["value"].lower()
        response_lower = response.lower()

        if correct_value in response_lower:
            outcome = "correct"
            self.state.facts_recalled += 1
        elif "don't" in response_lower or "sorry" in response_lower or "no record" in response_lower:
            outcome = "forgotten"
            self.state.facts_forgotten += 1
        else:
            outcome = "hallucination"
            self.state.hallucinations += 1

        return {
            "fact_key":   fact["key"],
            "fact_value": fact["value"],
            "response":   response,
            "outcome":    outcome,
            "in_context": in_context,
            "turn":       self.state.turn,
            "session":    self.state.session_id,
        }

    def new_session(self) -> None:
        """Start a new chat session (full context window reset)."""
        self.state.new_session()
        self._init_context()

    def fill_context_with_filler(self, num_turns: int = 20) -> None:
        """Push facts out of context by injecting realistic filler conversation."""
        fillers = [
            (
                "Can you explain how transformers work in modern language models?",
                "Transformers use a self-attention mechanism that allows each token in a "
                "sequence to attend to all other tokens simultaneously. Unlike recurrent "
                "networks, transformers process sequences in parallel, making them much "
                "more efficient to train on modern hardware. The key components are the "
                "multi-head attention layers, feed-forward networks, and layer "
                "normalisation. Positional encodings are added to token embeddings to "
                "preserve order information, since attention itself is permutation invariant.",
            ),
            (
                "What is retrieval-augmented generation and why is it useful?",
                "Retrieval-augmented generation, or RAG, is a technique that combines a "
                "language model with an external knowledge retrieval system. When a query "
                "arrives, relevant documents or passages are retrieved from a vector "
                "database using embedding similarity search, and these passages are "
                "prepended to the model's context. This allows the model to answer "
                "questions about facts it was not trained on, reduces hallucination by "
                "grounding responses in retrieved text, and enables the knowledge base to "
                "be updated without retraining the model.",
            ),
            (
                "How does quantisation help with running large language models locally?",
                "Quantisation reduces the memory footprint of a model by representing "
                "its weights in lower precision formats such as 4-bit or 8-bit integers "
                "instead of 32-bit floating point values. Techniques like GGUF and GPTQ "
                "apply quantisation while attempting to preserve model quality. For a "
                "7-billion parameter model, 4-bit quantisation can reduce the required "
                "memory from around 28 GB to approximately 4-5 GB, making it feasible "
                "to run on a consumer GPU or even a modern laptop with sufficient RAM.",
            ),
            (
                "What are the main differences between episodic and semantic memory?",
                "Episodic memory stores specific past experiences with their temporal "
                "and contextual details, such as what happened, where, and when. Semantic "
                "memory stores general world knowledge, facts, and concepts without "
                "reference to any specific experience. In cognitive psychology, these "
                "systems are considered functionally distinct: episodic memories are "
                "personal and autobiographical, while semantic memories are shared and "
                "impersonal. For artificial systems, episodic memory can be modelled "
                "using timestamped interaction logs, while semantic memory maps to "
                "a structured knowledge base or fact store.",
            ),
            (
                "What are common evaluation metrics for dialogue systems?",
                "Dialogue systems are typically evaluated along several dimensions. "
                "Automatic metrics include BLEU, ROUGE, and BERTScore for response "
                "quality, perplexity for fluency, and entity-level precision and recall "
                "for factual accuracy. Human evaluation uses ratings for coherence, "
                "engagingness, and groundedness. For memory-augmented systems, "
                "task-specific metrics such as retention accuracy and cross-session "
                "coherence are more informative, as they directly measure the system's "
                "ability to recall and consistently use information across turns.",
            ),
        ]
        for i in range(num_turns):
            q, a = fillers[i % len(fillers)]
            user_msg = Message(role="user", content=q, turn=self.state.turn,
                               session=self.state.session_id)
            self.state.add_message(user_msg)
            self.state.turn += 1
            asst_msg = Message(role="assistant", content=a, turn=self.state.turn,
                               session=self.state.session_id)
            self.state.add_message(asst_msg)
