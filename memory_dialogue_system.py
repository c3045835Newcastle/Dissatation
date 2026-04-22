"""
Post-Version Dialogue System: Hierarchical Memory Architecture
==============================================================

This module implements the enhanced (post) dialogue system described in the
CSC3094 dissertation proposal.  It augments the Base Llama 3.1 8B model with
a three-tier hierarchical memory architecture:

  Working Memory   — sliding-window of recent turn-pairs (default: last 6)
  Episodic Memory  — past interactions indexed as FAISS vector embeddings
  Semantic Memory  — persistent key-value facts extracted from the dialogue

A ``MemoryController`` coordinates all three layers.  On each turn:

  1.  Relevant past turns are retrieved from episodic memory via semantic
      similarity to the current user query.
  2.  Persistent user facts (name, preferences, stated constraints) are
      prepended to the prompt as a structured context block.
  3.  The current sliding-window context (working memory) follows.
  4.  When the working-memory window fills, evicted turns are automatically
      consolidated into the episodic vector store.

Usage
-----
::

    from memory_dialogue_system import HierarchicalMemoryDialogueSystem

    system = HierarchicalMemoryDialogueSystem(precision="int4")
    result = system.chat("My name is Alice and I am a PhD student.")
    print(result["response"])

    # Start a new session (episodic + semantic memory persist across sessions)
    system.new_session()
    result = system.chat("What is my name?")
    # The system should recall "Alice" from episodic / semantic memory.

Compare to the pre (baseline) system
-------------------------------------
The pre-system is ``BaseLlama31Model`` from ``llama_base_model.py``.  It has
no external memory — all retention depends solely on tokens remaining inside
the context window.  Once turns fall outside the window, information is lost.
"""

import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from memory.memory_controller import MemoryController


class HierarchicalMemoryDialogueSystem:
    """
    Enhanced Llama 3.1 8B dialogue system with hierarchical memory.

    Parameters
    ----------
    working_memory_turns : int
        Number of recent turn-pairs kept in the sliding context window.
        Older turns are evicted to episodic memory.  Default: 6.
    episodic_top_k : int
        Maximum number of episodic entries retrieved per query.  Default: 3.
    episodic_min_score : float
        Minimum cosine similarity for an episodic entry to be shown.
    persist_path : str or None
        Directory for cross-session memory persistence.  ``None`` = in-memory.
    precision : str
        Model loading precision: ``"fp16"``, ``"int8"``, or ``"int4"``.
        Default ``"int4"`` (safest choice for 16 GB VRAM / 16 GB RAM systems).
    system_prompt : str or None
        Override the default system instruction prepended to every prompt.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, accurate assistant. "
        "Use any [User Profile] or [Relevant past conversation excerpts] "
        "provided above to give consistent, coherent, and factually accurate responses. "
        "Never contradict facts you have previously stated."
    )

    def __init__(
        self,
        working_memory_turns: int = 6,
        episodic_top_k: int = 3,
        episodic_min_score: float = 0.25,
        persist_path: Optional[str] = None,
        precision: str = "int4",
        system_prompt: Optional[str] = None,
    ):
        self.precision = precision
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self.memory = MemoryController(
            working_memory_turns=working_memory_turns,
            episodic_top_k=episodic_top_k,
            episodic_min_score=episodic_min_score,
            persist_path=persist_path,
        )

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_model(precision)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, precision: str):
        """Load the Llama 3.1 8B model in the requested numeric precision."""
        print(f"[HierarchicalMemory] Loading {config.MODEL_NAME} ({precision.upper()}) …")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            cache_dir=config.MODEL_CACHE_DIR,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict = {
            "cache_dir": config.MODEL_CACHE_DIR,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if precision == "int4":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_cfg
            model_kwargs["torch_dtype"] = torch.float16
        elif precision == "int8":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
        else:  # fp16
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, **model_kwargs)
        self.model.eval()
        print("[HierarchicalMemory] Model ready.")

    # ------------------------------------------------------------------
    # Core dialogue turn
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> Dict:
        """
        Process one conversational turn with full hierarchical memory support.

        Steps
        -----
        1. Add user turn to working memory (triggers semantic extraction and
           episodic consolidation of any evicted turns).
        2. Build enriched context: semantic preamble + episodic retrieval +
           recent working-memory turns.
        3. Append current user message and tokenise via chat template.
        4. Generate with the LLM (greedy by default for reproducibility).
        5. Add assistant response to working memory.

        Parameters
        ----------
        user_message :    The user's utterance.
        max_new_tokens :  Maximum tokens to generate.
        temperature :     Sampling temperature (0 = greedy).
        do_sample :       Whether to use nucleus sampling.

        Returns
        -------
        dict with keys:
            ``response``          — generated assistant text (str)
            ``episodic_hits``     — number of episodic entries retrieved (int)
            ``memory_stats``      — working/episodic/semantic sizes (dict)
            ``generation_time_s`` — wall-clock generation time in seconds (float)
            ``prompt_tokens``     — number of prompt tokens sent to model (int)
        """
        t_start = time.perf_counter()

        # 1. Record user turn (also extracts facts + evicts if needed)
        self.memory.add_user_turn(user_message)

        # 2. Build enriched context (does NOT include the current user turn)
        context_messages, n_episodic = self.memory.build_context(
            user_query=user_message,
            system_prompt=self.system_prompt,
        )

        # Append current user message
        context_messages.append({"role": "user", "content": user_message})

        # 3. Tokenise
        try:
            prompt = self.tokenizer.apply_chat_template(
                context_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback formatting if chat template is unavailable
            parts = []
            for msg in context_messages:
                role_label = {
                    "system": "System",
                    "user": "User",
                    "assistant": "Assistant",
                }.get(msg["role"], msg["role"].capitalize())
                parts.append(f"{role_label}: {msg['content']}")
            prompt = "\n".join(parts) + "\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_prompt_tokens = inputs["input_ids"].shape[1]

        # 4. Generate
        gen_kwargs: Dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = config.TOP_P
            gen_kwargs["top_k"] = config.TOP_K

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_token_ids = output_ids[0][n_prompt_tokens:]
        response_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        # 5. Record assistant turn
        self.memory.add_assistant_turn(response_text)

        return {
            "response": response_text,
            "episodic_hits": n_episodic,
            "memory_stats": dict(self.memory.stats),
            "generation_time_s": round(time.perf_counter() - t_start, 3),
            "prompt_tokens": n_prompt_tokens,
        }

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def new_session(self):
        """
        Start a new conversation session.

        Working memory is cleared; episodic and semantic memory persist.
        Call this to simulate a user returning after a break — the system
        retains knowledge of past interactions without the raw transcript.
        """
        self.memory.reset_session()

    def reset(self):
        """
        Fully reset all memory layers.

        Use between independent evaluation scenarios to prevent cross-scenario
        memory leakage.
        """
        self.memory.reset_all()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def memory_stats(self) -> Dict:
        """Current sizes of all three memory layers."""
        return self.memory.stats

    def describe(self) -> str:
        """Return a one-line description of the system configuration."""
        return (
            f"HierarchicalMemoryDialogueSystem("
            f"precision={self.precision}, "
            f"working_turns={self.memory.working_memory.max_turns}, "
            f"episodic_top_k={self.memory.episodic_top_k}, "
            f"min_score={self.memory.episodic_min_score})"
        )
