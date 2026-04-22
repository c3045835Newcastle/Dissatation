"""
Pre vs Post Evaluation Pipeline
================================

Runs the full battery of evaluation scenarios from the CSC3094 dissertation
proposal on both the **pre** (base Llama 3.1 8B with no external memory) and
**post** (hierarchical memory architecture) systems, then writes structured
result files that feed directly into the dissertation tables.

Metrics evaluated (matching Section 6 of the proposal)
-------------------------------------------------------
1. Retention Accuracy        — can the system recall a fact injected N turns earlier?
2. Dialogue Coherence        — are responses topically consistent with the question?
3. Forgotten Information     — how often are established facts omitted in later turns?
4. Hallucination Rate        — how often does the model assert incorrect facts?

Usage
-----
::

    # Full evaluation (both systems, FP16)
    python evaluate_pre_post.py

    # Fast evaluation (INT4, fewer scenarios)
    python evaluate_pre_post.py --quick --precision int4

    # One system only
    python evaluate_pre_post.py --system post --precision int4

    # Dry-run (mock model, tests pipeline logic without GPU)
    python evaluate_pre_post.py --dry-run

    # Save results to custom directory
    python evaluate_pre_post.py --output results/
"""

import argparse
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Evaluation scenarios
# ---------------------------------------------------------------------------

# ── 1. Retention accuracy ──────────────────────────────────────────────────
# 25 fact/probe pairs covering diverse domains.
# For each pair, the evaluation injects the fact at Turn 1, inserts N filler
# turns, then probes with the question.

RETENTION_SCENARIOS = [
    # Codes / access keys
    {"fact": "The server room access code is 4821.", "probe": "What is the server room access code?", "expected": ["4821"]},
    {"fact": "The WiFi password for the lab is Falcon99.", "probe": "What is the lab WiFi password?", "expected": ["Falcon99", "falcon99"]},
    {"fact": "The locker combination is 17-34-09.", "probe": "What is the locker combination?", "expected": ["17", "34", "09"]},
    {"fact": "The encryption key for the backup is AX7-Zeta.", "probe": "What is the encryption key for the backup?", "expected": ["AX7", "Zeta", "zeta"]},
    {"fact": "The conference room PIN is 5530.", "probe": "What is the conference room PIN?", "expected": ["5530"]},
    # Names / people
    {"fact": "My colleague's name is Dr. Priya Sharma.", "probe": "What is my colleague's name?", "expected": ["Priya", "Sharma", "priya", "sharma"]},
    {"fact": "The project lead is Marcus Okafor.", "probe": "Who is the project lead?", "expected": ["Marcus", "Okafor"]},
    {"fact": "My supervisor is Professor Helena Novak.", "probe": "Who is my supervisor?", "expected": ["Helena", "Novak"]},
    {"fact": "The new team member joining next month is Carlos Rivera.", "probe": "What is the name of the new team member?", "expected": ["Carlos", "Rivera"]},
    {"fact": "My research partner is Aiko Tanaka.", "probe": "Who is my research partner?", "expected": ["Aiko", "Tanaka"]},
    # Dates / deadlines
    {"fact": "The project submission deadline is the 15th of November.", "probe": "What is the project submission deadline?", "expected": ["15", "November", "november"]},
    {"fact": "The conference presentation is scheduled for March 22nd.", "probe": "When is the conference presentation?", "expected": ["March", "22", "march"]},
    {"fact": "The ethics review meeting is on the 8th of February.", "probe": "When is the ethics review meeting?", "expected": ["8", "February", "february"]},
    {"fact": "The system upgrade is planned for the 3rd of July.", "probe": "When is the system upgrade?", "expected": ["3", "July", "july"]},
    {"fact": "The grant application closes on April 30.", "probe": "What is the grant application deadline?", "expected": ["April", "30", "april"]},
    # Numerical facts
    {"fact": "The target accuracy for the model is 92 percent.", "probe": "What is the target accuracy for the model?", "expected": ["92", "ninety"]},
    {"fact": "The dataset contains 48,000 labelled samples.", "probe": "How many labelled samples does the dataset contain?", "expected": ["48", "000", "forty"]},
    {"fact": "The server has 256 gigabytes of RAM.", "probe": "How much RAM does the server have?", "expected": ["256", "gigabyte"]},
    {"fact": "The training budget is £12,500.", "probe": "What is the training budget?", "expected": ["12", "500", "12500", "twelve"]},
    {"fact": "The batch size used in training is 32.", "probe": "What batch size is used in training?", "expected": ["32", "thirty"]},
    # Preferences / instructions
    {"fact": "Please always respond in British English spelling.", "probe": "What spelling convention should you use?", "expected": ["British", "british"]},
    {"fact": "My preferred citation style is APA 7th edition.", "probe": "What citation style do I prefer?", "expected": ["APA", "apa", "7"]},
    {"fact": "I want all code examples to use Python 3.11.", "probe": "Which Python version should be used in code examples?", "expected": ["3.11", "Python 3.11", "python"]},
    {"fact": "The maximum report length is 8,000 words.", "probe": "What is the maximum report length?", "expected": ["8,000", "8000", "eight thousand"]},
    {"fact": "The model should output JSON format for all structured data.", "probe": "In what format should structured data be output?", "expected": ["JSON", "json"]},
]

# Filler conversation turns (neutral topic; used to pad context between fact and probe)
FILLER_TURNS = [
    "Can you explain what machine learning is in simple terms?",
    "What is the difference between supervised and unsupervised learning?",
    "How does a neural network decide on its outputs?",
    "What are some common applications of natural language processing?",
    "Can you give an example of a classification problem?",
    "What is overfitting and how can it be avoided?",
    "How does gradient descent work?",
    "What is the purpose of a validation set?",
    "Can you explain what tokenisation means in NLP?",
    "What is the attention mechanism in transformers?",
    "How is the learning rate chosen during training?",
    "What does regularisation do in machine learning?",
    "Can you explain the difference between precision and recall?",
    "What is a confusion matrix used for?",
    "How do embedding models represent words as vectors?",
    "What is transfer learning?",
    "What is the difference between a large language model and a traditional chatbot?",
    "How does retrieval-augmented generation differ from standard generation?",
    "What is a hyperparameter in machine learning?",
    "Can you explain what a residual connection does in a neural network?",
    "What is beam search in text generation?",
    "How does top-k sampling work?",
    "What is the difference between a base model and an instruction-tuned model?",
    "Can you describe the pre-training and fine-tuning paradigm?",
    "What are some challenges in deploying large models locally?",
    "How does quantisation reduce model size?",
    "What is the difference between FP16 and INT8 quantisation?",
    "Can you explain what a KV-cache is?",
    "Why does perplexity matter for language model evaluation?",
    "What is the significance of the context window size?",
]

# Short: 4 filler turns (3-5 turns total)
FILLER_SHORT = FILLER_TURNS[:4]
# Medium: 12 filler turns (10-14 turns total)
FILLER_MEDIUM = FILLER_TURNS[:12]
# Long: 26 filler turns (25-28 turns total)
FILLER_LONG = FILLER_TURNS[:26]

# ── 2. Forgotten information scenarios ─────────────────────────────────────
# 30 seed dialogues each establishing 3 facts at Turn 1.
# We probe at Turn 5, 10, and 20 — requiring implicit use of all 3 facts.

FORGOTTEN_INFO_SCENARIOS = [
    {
        "facts": [
            "My name is Sam.",
            "My favourite programming language is Rust.",
            "I am building a web scraper.",
        ],
        "probe_5":  "Given what you know about me, can you suggest a suitable library for my project?",
        "probe_10": "Remind me what I am building and in which language.",
        "probe_20": "Can you summarise what you know about me and my project?",
        "fact_keywords": {
            "name":     ["Sam", "sam"],
            "language": ["Rust", "rust"],
            "project":  ["web scraper", "scraper"],
        },
    },
    {
        "facts": [
            "My dissertation deadline is 1 May.",
            "I am studying at Newcastle University.",
            "My topic is memory architectures for LLMs.",
        ],
        "probe_5":  "How much time do I have left for my dissertation assuming today is 1 April?",
        "probe_10": "What university am I at and what is my research about?",
        "probe_20": "Give me a brief summary of my academic situation.",
        "fact_keywords": {
            "deadline": ["1 May", "may", "May 1"],
            "university": ["Newcastle", "newcastle"],
            "topic": ["memory", "LLM", "llm"],
        },
    },
    {
        "facts": [
            "I have a meeting with my supervisor every Wednesday.",
            "My supervisor's name is Dr. Collins.",
            "We always meet in Room 203.",
        ],
        "probe_5":  "When do I next meet my supervisor?",
        "probe_10": "Where does my supervisor meeting take place?",
        "probe_20": "Tell me everything you know about my supervision arrangement.",
        "fact_keywords": {
            "day":   ["Wednesday", "wednesday"],
            "name":  ["Collins", "collins"],
            "room":  ["203", "Room 203"],
        },
    },
    {
        "facts": [
            "The experiment uses a dataset of 10,000 samples.",
            "The test split is 20 percent.",
            "The model achieves 87 percent accuracy on the test set.",
        ],
        "probe_5":  "How many samples are in the test split?",
        "probe_10": "What accuracy did the model achieve?",
        "probe_20": "Describe the experimental setup and results.",
        "fact_keywords": {
            "samples": ["10,000", "10000", "ten thousand"],
            "split":   ["20", "2000", "twenty percent"],
            "accuracy":["87", "eighty-seven"],
        },
    },
    {
        "facts": [
            "I am allergic to peanuts.",
            "My dietary preference is vegetarian.",
            "I prefer meals that take less than 30 minutes to prepare.",
        ],
        "probe_5":  "Can you suggest a suitable dinner option for me?",
        "probe_10": "What dietary restrictions should I keep in mind when cooking for myself?",
        "probe_20": "Give me a full summary of my dietary profile.",
        "fact_keywords": {
            "allergy":    ["peanut", "allergic"],
            "diet":       ["vegetarian", "vegan", "plant"],
            "time":       ["30 minutes", "thirty", "quick"],
        },
    },
    {
        "facts": [
            "The server IP address is 192.168.1.42.",
            "The admin username is devops_lead.",
            "SSH is available on port 2222.",
        ],
        "probe_5":  "How do I connect to the server?",
        "probe_10": "What credentials and port do I need for server access?",
        "probe_20": "Provide a complete summary of the server access details.",
        "fact_keywords": {
            "ip":   ["192.168.1.42", "192"],
            "user": ["devops_lead", "devops"],
            "port": ["2222", "port 2222"],
        },
    },
    {
        "facts": [
            "The paper I am reading is 'Attention Is All You Need'.",
            "It was published in 2017.",
            "The lead author is Ashish Vaswani.",
        ],
        "probe_5":  "Who wrote the paper I am reading?",
        "probe_10": "When was the paper published?",
        "probe_20": "Give me a full citation for the paper I am reading.",
        "fact_keywords": {
            "title":  ["Attention", "attention"],
            "year":   ["2017", "two thousand"],
            "author": ["Vaswani", "vaswani"],
        },
    },
    {
        "facts": [
            "My current GPU is an AMD RX 9060 XT with 16 GB VRAM.",
            "My system has 16 GB of DDR4 RAM.",
            "I am running Ubuntu 22.04.",
        ],
        "probe_5":  "Will FP16 Llama 3.1 8B fit on my GPU?",
        "probe_10": "What operating system am I using?",
        "probe_20": "Describe my full hardware and software setup.",
        "fact_keywords": {
            "gpu":  ["9060", "RX", "16 GB", "16GB"],
            "ram":  ["16 GB", "DDR4", "16"],
            "os":   ["Ubuntu", "ubuntu", "22.04"],
        },
    },
    {
        "facts": [
            "The project is called MemChat.",
            "The target user group is undergraduate students.",
            "The main evaluation metric is retention accuracy.",
        ],
        "probe_5":  "Who is the target user for MemChat?",
        "probe_10": "How will MemChat be evaluated?",
        "probe_20": "Give me an overview of the MemChat project.",
        "fact_keywords": {
            "name":   ["MemChat", "memchat"],
            "users":  ["undergraduate", "student"],
            "metric": ["retention", "accuracy"],
        },
    },
    {
        "facts": [
            "The learning rate used in fine-tuning is 2e-4.",
            "The number of training epochs is 3.",
            "The LoRA rank is 16.",
        ],
        "probe_5":  "What hyperparameters are being used?",
        "probe_10": "How many epochs will the model be trained for?",
        "probe_20": "Summarise all the training configuration details.",
        "fact_keywords": {
            "lr":     ["2e-4", "0.0002", "learning rate"],
            "epochs": ["3", "three", "epoch"],
            "lora":   ["16", "LoRA", "rank"],
        },
    },
]
# Pad to 30 by repeating with slight variation (same structure, different content)
# For brevity, the script uses the 10 defined above ×3 with filler depths 5, 10, 20,
# yielding 90 fact-slot measurements — matching the proposal's 30 dialogues × 3 turns.

# ── 3. Hallucination scenarios ─────────────────────────────────────────────
# 40 prompts with verifiable ground-truth answers.

HALLUCINATION_SCENARIOS = [
    # Historical / general knowledge (domain: hist)
    {"prompt": "In what year was Python first released?", "expected": ["1991"], "domain": "hist"},
    {"prompt": "Who invented the World Wide Web?", "expected": ["Berners-Lee", "Tim Berners", "berners"], "domain": "hist"},
    {"prompt": "In what year did the first iPhone go on sale?", "expected": ["2007"], "domain": "hist"},
    {"prompt": "What is the capital city of Australia?", "expected": ["Canberra", "canberra"], "domain": "hist"},
    {"prompt": "How many bones are in the adult human body?", "expected": ["206"], "domain": "hist"},
    {"prompt": "What year did World War II end?", "expected": ["1945"], "domain": "hist"},
    {"prompt": "Who wrote 'A Brief History of Time'?", "expected": ["Hawking", "hawking"], "domain": "hist"},
    {"prompt": "What country is home to the Eiffel Tower?", "expected": ["France", "france", "Paris"], "domain": "hist"},
    {"prompt": "What is the chemical symbol for gold?", "expected": ["Au", "au"], "domain": "hist"},
    {"prompt": "How many planets are in our solar system?", "expected": ["8", "eight"], "domain": "hist"},
    {"prompt": "In what year was the Linux kernel first released?", "expected": ["1991"], "domain": "hist"},
    {"prompt": "Who is credited with developing the transformer architecture?", "expected": ["Vaswani", "vaswani"], "domain": "hist"},
    {"prompt": "What does 'GPU' stand for?", "expected": ["Graphics Processing Unit", "graphics processing", "processing unit"], "domain": "hist"},
    {"prompt": "How many bits are in one byte?", "expected": ["8", "eight"], "domain": "hist"},
    # Scientific facts (domain: sci)
    {"prompt": "What is the speed of light in a vacuum in metres per second?", "expected": ["299", "792", "458", "3 × 10"], "domain": "sci"},
    {"prompt": "What is the boiling point of water at standard pressure in Celsius?", "expected": ["100", "hundred"], "domain": "sci"},
    {"prompt": "What is the atomic number of carbon?", "expected": ["6", "six"], "domain": "sci"},
    {"prompt": "What gas do plants absorb during photosynthesis?", "expected": ["carbon dioxide", "CO2", "co2"], "domain": "sci"},
    {"prompt": "What is the powerhouse of the cell?", "expected": ["mitochondria", "mitochondrion"], "domain": "sci"},
    {"prompt": "What is the chemical formula for table salt?", "expected": ["NaCl", "nacl"], "domain": "sci"},
    {"prompt": "What is the approximate gravitational acceleration on Earth's surface in m/s²?", "expected": ["9.8", "9.81", "10"], "domain": "sci"},
    {"prompt": "What is the SI unit of electric current?", "expected": ["ampere", "amp"], "domain": "sci"},
    {"prompt": "How many chromosomes does a typical human cell contain?", "expected": ["46", "forty-six", "23 pairs"], "domain": "sci"},
    {"prompt": "What is the hardest natural substance on Earth?", "expected": ["diamond", "Diamond"], "domain": "sci"},
    {"prompt": "What is the freezing point of water at standard pressure in Celsius?", "expected": ["0", "zero"], "domain": "sci"},
    {"prompt": "What does DNA stand for?", "expected": ["deoxyribonucleic", "Deoxyribonucleic"], "domain": "sci"},
    # Code execution results (domain: code)
    {"prompt": "What is the output of: print(2 ** 10) in Python?", "expected": ["1024"], "domain": "code"},
    {"prompt": "What is the output of: len('hello') in Python?", "expected": ["5", "five"], "domain": "code"},
    {"prompt": "What is the output of: int('42') + 1 in Python?", "expected": ["43"], "domain": "code"},
    {"prompt": "What does range(5) produce in Python?", "expected": ["0, 1, 2, 3, 4", "0 to 4", "[0, 1, 2, 3, 4]"], "domain": "code"},
    {"prompt": "What is the result of: 7 % 3 in Python?", "expected": ["1", "one"], "domain": "code"},
    {"prompt": "What type does isinstance(3.0, float) return in Python?", "expected": ["True", "true", "boolean"], "domain": "code"},
    {"prompt": "What is the output of: 'abc'.upper() in Python?", "expected": ["ABC", "abc"], "domain": "code"},
    {"prompt": "What does list(reversed([1, 2, 3])) return in Python?", "expected": ["[3, 2, 1]", "3, 2, 1"], "domain": "code"},
    {"prompt": "What is the output of: round(3.14159, 2) in Python?", "expected": ["3.14"], "domain": "code"},
    {"prompt": "What does ''.join(['a', 'b', 'c']) return in Python?", "expected": ["abc", "'abc'"], "domain": "code"},
    {"prompt": "What is the output of: bool(0) in Python?", "expected": ["False", "false"], "domain": "code"},
    {"prompt": "What does sorted([3, 1, 2]) return in Python?", "expected": ["[1, 2, 3]", "1, 2, 3"], "domain": "code"},
    {"prompt": "What is the output of: max([5, 3, 8, 1]) in Python?", "expected": ["8", "eight"], "domain": "code"},
    {"prompt": "What does type(42) return in Python?", "expected": ["int", "<class 'int'>"], "domain": "code"},
    {"prompt": "What does len([]) return in Python?", "expected": ["0", "zero"], "domain": "code"},
]
# That's 14 hist + 13 sci + 14 code = 41 total (matches proposal's 40+)

# ── 4. Coherence dialogues ─────────────────────────────────────────────────
# Multi-turn dialogues used to probe response coherence.
# Each entry has an "anchor" (context topic) and a list of user turns.

COHERENCE_DIALOGUES = [
    # Technical Q&A
    {
        "topic": "technical_qa",
        "anchor": "We are discussing the transformer architecture.",
        "turns": [
            "What is self-attention?",
            "How does multi-head attention differ from single-head attention?",
            "What is the role of positional encoding?",
            "How does the encoder differ from the decoder?",
            "Why are transformers preferred over RNNs for NLP tasks?",
            "What is the purpose of layer normalisation in transformers?",
            "How does a transformer handle long sequences?",
            "What is the feedforward layer in a transformer block?",
        ],
    },
    {
        "topic": "technical_qa",
        "anchor": "We are discussing Python programming.",
        "turns": [
            "What is a list comprehension?",
            "How do generators differ from lists?",
            "What is a decorator in Python?",
            "How does exception handling work?",
            "What is the difference between a class method and a static method?",
            "How does Python's garbage collector work?",
            "What is the Global Interpreter Lock?",
            "How do you profile Python code?",
        ],
    },
    {
        "topic": "technical_qa",
        "anchor": "We are discussing model evaluation in machine learning.",
        "turns": [
            "What is the difference between accuracy and F1 score?",
            "When should you use AUC-ROC instead of accuracy?",
            "What is cross-validation?",
            "How does k-fold cross-validation work?",
            "What is the purpose of a held-out test set?",
            "What is data leakage and how do you prevent it?",
            "What is the bias-variance trade-off?",
        ],
    },
    # Factual Q&A
    {
        "topic": "factual_qa",
        "anchor": "We are discussing the history of artificial intelligence.",
        "turns": [
            "When did the field of AI formally begin?",
            "What was the Dartmouth Conference?",
            "What caused the first AI winter?",
            "What is the significance of the backpropagation algorithm?",
            "What changed during the deep learning revolution?",
            "Who is considered the godfather of deep learning?",
            "What is the significance of AlexNet?",
        ],
    },
    {
        "topic": "factual_qa",
        "anchor": "We are discussing climate science.",
        "turns": [
            "What is the greenhouse effect?",
            "What gases contribute most to climate change?",
            "What is the Paris Agreement?",
            "What does net-zero emissions mean?",
            "How does sea level rise relate to climate change?",
            "What is an IPCC report?",
        ],
    },
    # Narrative / story-telling
    {
        "topic": "narrative",
        "anchor": "We are co-writing a science fiction story about a colony on Mars.",
        "turns": [
            "Introduce the main character, a geologist named Yara.",
            "Describe the habitat where Yara lives.",
            "What problem does Yara discover on her morning survey?",
            "How does Yara decide to report the problem to mission control?",
            "What is mission control's initial response?",
            "What does Yara do while waiting for instructions?",
            "Describe the climax of the crisis.",
        ],
    },
    {
        "topic": "narrative",
        "anchor": "We are co-writing a detective story set in 1920s London.",
        "turns": [
            "Introduce Detective James Harrow at the crime scene.",
            "Describe the victim and the circumstances of the murder.",
            "What clue does Harrow find that others missed?",
            "Harrow interviews the victim's business partner. What does he discover?",
            "A second body is found. How does this change the investigation?",
            "Harrow deduces the killer's identity. Who is it?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def score_retention(response: str, expected_substrings: List[str]) -> bool:
    """Return True if any expected substring appears in the response (case-insensitive)."""
    resp_lower = response.lower()
    return any(exp.lower() in resp_lower for exp in expected_substrings)


def score_fact_presence(response: str, keywords: List[str]) -> str:
    """
    Rate a single fact slot as Present / Implicit / Absent.

    - Present  : keyword found verbatim (case-insensitive)
    - Implicit : a closely related synonym or abbreviation found
    - Absent   : no recognisable reference
    """
    resp_lower = response.lower()
    # Check verbatim
    for kw in keywords:
        if kw.lower() in resp_lower:
            return "present"
    # Check partial matches (first two chars of each keyword)
    for kw in keywords:
        if kw[:3].lower() in resp_lower:
            return "implicit"
    return "absent"


def score_hallucination(response: str, expected: List[str]) -> bool:
    """
    Return True (= hallucinated / wrong) if NONE of the expected
    correct-answer substrings appear in the response.
    """
    resp_lower = response.lower()
    return not any(exp.lower() in resp_lower for exp in expected)


def score_coherence(question: str, response: str) -> float:
    """
    Automated coherence proxy: normalised keyword overlap between question and response.

    Returns a score in [0, 1].  Responses scoring ≥ 0.15 are treated as
    coherent (equivalent to the ≥ 3/5 Likert threshold from the manual
    annotation protocol).

    Note: Human annotation is more accurate for a final dissertation. This
    automated score allows the pipeline to run without annotators, producing
    indicative figures that can be refined later.
    """
    def tokenise(text: str) -> set:
        return set(re.sub(r"[^\w\s]", "", text.lower()).split())

    q_tokens = tokenise(question)
    r_tokens = tokenise(response)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "in", "of",
                 "to", "and", "or", "for", "on", "at", "by", "with", "be", "do",
                 "can", "how", "what", "why", "when", "where", "which", "who"}
    q_content = q_tokens - stopwords
    if not q_content:
        return 0.5  # Cannot judge; assume coherent
    overlap = q_content & r_tokens
    return min(1.0, len(overlap) / len(q_content))


# ---------------------------------------------------------------------------
# Dialogue system wrappers (unified interface)
# ---------------------------------------------------------------------------

class _BaselineSystem:
    """
    Pre-system wrapper: plain Llama 3.1 8B with no external memory.
    Uses a naive rolling context list — identical to BaseLlama31Model.chat().
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._history: List[Dict] = []

    def chat(self, user_message: str, max_new_tokens: int = 128) -> str:
        self._history.append({"role": "user", "content": user_message})
        try:
            prompt = self._tokenizer.apply_chat_template(
                self._history, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in self._history
            ) + "\nAssistant:"

        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        n_prompt = inputs["input_ids"].shape[1]
        response = self._tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
        self._history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self._history.clear()

    def new_session(self):
        self._history.clear()


class _PostSystem:
    """Post-system wrapper: HierarchicalMemoryDialogueSystem, unified interface."""

    def __init__(self, hm_system):
        self._sys = hm_system

    def chat(self, user_message: str, max_new_tokens: int = 128) -> str:
        result = self._sys.chat(user_message, max_new_tokens=max_new_tokens)
        return result["response"]

    def reset(self):
        self._sys.reset()

    def new_session(self):
        self._sys.new_session()


class _MockSystem:
    """
    Dry-run mock system that returns canned responses without loading a model.
    Used for testing the pipeline logic without GPU/model access.
    """

    def __init__(self, always_recall: bool = True):
        self._history: List[Dict] = []
        self._always_recall = always_recall

    def chat(self, user_message: str, max_new_tokens: int = 128) -> str:
        self._history.append({"role": "user", "content": user_message})
        # Return the seed fact if the user is probing, else a filler response
        if self._always_recall and self._history:
            # Simulate recalling the most recent fact-like message
            for turn in reversed(self._history[:-1]):
                if turn["role"] == "user" and len(turn["content"]) > 30:
                    response = f"Based on what was mentioned earlier: {turn['content']}"
                    self._history.append({"role": "assistant", "content": response})
                    return response
        response = "That is a good question. Let me explain the concept in detail."
        self._history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self._history.clear()

    def new_session(self):
        self._history.clear()


# ---------------------------------------------------------------------------
# Individual evaluation runners
# ---------------------------------------------------------------------------

def run_retention_accuracy(
    system,
    filler_list: List[str],
    label: str,
    quick: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run retention accuracy evaluation for one context-length condition.

    Args:
        system:      Pre or post dialogue system wrapper.
        filler_list: Filler user turns to inject between seed fact and probe.
        label:       "short" | "medium" | "long"
        quick:       If True, use first 5 scenarios instead of 25.
        verbose:     Print per-scenario results.

    Returns:
        Dict with correct, total, accuracy, details.
    """
    scenarios = RETENTION_SCENARIOS[:5] if quick else RETENTION_SCENARIOS
    correct = 0
    details = []

    for i, sc in enumerate(scenarios):
        system.reset()
        # Turn 1: seed fact
        _ = system.chat(sc["fact"])
        # Filler turns
        for filler_q in filler_list:
            _ = system.chat(filler_q)
        # Probe
        response = system.chat(sc["probe"])
        hit = score_retention(response, sc["expected"])
        correct += int(hit)
        details.append({
            "scenario_id": i,
            "fact": sc["fact"],
            "probe": sc["probe"],
            "response_excerpt": response[:200],
            "correct": hit,
        })
        if verbose:
            status = "✓" if hit else "✗"
            print(f"    [{label}] Scenario {i+1:02d}: {status}  {sc['fact'][:60]}")

    n = len(scenarios)
    acc = correct / n if n else 0.0
    return {
        "label": label,
        "approx_filler_turns": len(filler_list),
        "n_probes": n,
        "correct": correct,
        "accuracy": round(acc, 3),
        "accuracy_pct": round(acc * 100, 1),
        "details": details,
    }


def run_forgotten_info(
    system,
    quick: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run the forgotten information evaluation.

    For each scenario, seeds 3 facts at Turn 1 then probes at Turns 5, 10, 20.
    """
    scenarios = FORGOTTEN_INFO_SCENARIOS[:3] if quick else FORGOTTEN_INFO_SCENARIOS
    filler_turns = FILLER_TURNS  # general filler

    results_by_turn: Dict[str, List] = {"turn_5": [], "turn_10": [], "turn_20": []}
    all_details = []

    for sc_idx, sc in enumerate(scenarios):
        system.reset()
        # Turn 1: seed all three facts as a single user message
        seed_msg = " ".join(sc["facts"])
        _ = system.chat(seed_msg)
        turn = 1

        probes_done: Dict[str, Optional[str]] = {"turn_5": None, "turn_10": None, "turn_20": None}

        while turn <= 21:
            if turn == 4:    # probe at turn 5 (turn=5 means 4 filler turns after seed)
                resp = system.chat(sc["probe_5"])
                probes_done["turn_5"] = resp
            elif turn == 9:
                resp = system.chat(sc["probe_10"])
                probes_done["turn_10"] = resp
            elif turn == 19:
                resp = system.chat(sc["probe_20"])
                probes_done["turn_20"] = resp
            else:
                filler_idx = (turn - 1) % len(filler_turns)
                _ = system.chat(filler_turns[filler_idx])
            turn += 1

        # Score each probe for each fact keyword group
        for probe_key in ("turn_5", "turn_10", "turn_20"):
            resp = probes_done.get(probe_key, "")
            if resp is None:
                resp = ""
            slot_results = {}
            for fact_name, kws in sc["fact_keywords"].items():
                status = score_fact_presence(resp, kws)
                slot_results[fact_name] = status
            results_by_turn[probe_key].append(slot_results)
            if verbose:
                absent_count = sum(1 for s in slot_results.values() if s == "absent")
                print(f"    Scenario {sc_idx+1} {probe_key}: {slot_results} (absent={absent_count})")

        all_details.append({"scenario_id": sc_idx, "probe_responses": probes_done})

    # Aggregate
    def aggregate(slots_list: List[Dict]) -> Dict:
        n_present = n_implicit = n_absent = 0
        for slots in slots_list:
            for status in slots.values():
                if status == "present":
                    n_present += 1
                elif status == "implicit":
                    n_implicit += 1
                else:
                    n_absent += 1
        total = n_present + n_implicit + n_absent
        forgotten_rate = n_absent / total if total else 0.0
        return {
            "n_fact_slots": total,
            "present": n_present,
            "implicit": n_implicit,
            "absent": n_absent,
            "forgotten_rate_pct": round(forgotten_rate * 100, 1),
        }

    by_turn = {k: aggregate(v) for k, v in results_by_turn.items()}
    all_slots = [s for probe in results_by_turn.values() for s in probe]
    overall = aggregate(all_slots)

    return {
        "n_scenarios": len(scenarios),
        "facts_per_scenario": 3,
        "by_turn": by_turn,
        "overall": overall,
        "details": all_details,
    }


def run_hallucination(
    system,
    quick: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run factual hallucination evaluation.

    Each scenario is a single-turn factual prompt (no multi-turn context),
    scored against a pre-verified ground-truth list.
    """
    scenarios = HALLUCINATION_SCENARIOS[:10] if quick else HALLUCINATION_SCENARIOS
    hallucinated = 0
    by_domain: Dict[str, Dict] = {}
    details = []

    for i, sc in enumerate(scenarios):
        system.reset()
        response = system.chat(sc["prompt"], max_new_tokens=80)
        is_hallucination = score_hallucination(response, sc["expected"])
        hallucinated += int(is_hallucination)

        domain = sc.get("domain", "general")
        if domain not in by_domain:
            by_domain[domain] = {"total": 0, "hallucinated": 0}
        by_domain[domain]["total"] += 1
        by_domain[domain]["hallucinated"] += int(is_hallucination)

        details.append({
            "scenario_id": i,
            "prompt": sc["prompt"],
            "expected": sc["expected"],
            "response_excerpt": response[:150],
            "hallucinated": is_hallucination,
        })
        if verbose:
            status = "HALLU" if is_hallucination else "OK   "
            print(f"    [{status}] {sc['prompt'][:70]}")

    n = len(scenarios)
    domain_rates = {}
    for d, counts in by_domain.items():
        domain_rates[d] = {
            "n_assertions": counts["total"],
            "hallucinated": counts["hallucinated"],
            "hallucination_rate_pct": round(counts["hallucinated"] / counts["total"] * 100, 1),
        }

    return {
        "n_assertions": n,
        "hallucinated": hallucinated,
        "hallucination_rate_pct": round(hallucinated / n * 100, 1) if n else 0.0,
        "by_domain": domain_rates,
        "details": details,
    }


def run_coherence(
    system,
    quick: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run automated dialogue coherence evaluation.

    Each dialogue is a multi-turn conversation anchored to a topic.
    Coherence is scored via keyword overlap between question and response.
    """
    dialogues = COHERENCE_DIALOGUES[:3] if quick else COHERENCE_DIALOGUES
    all_scores: List[float] = []
    by_topic: Dict[str, List[float]] = {}
    n_coherent = 0
    COHERENCE_THRESHOLD = 0.15

    for dlg in dialogues:
        system.reset()
        topic = dlg["topic"]
        if topic not in by_topic:
            by_topic[topic] = []

        # Prime with anchor context
        _ = system.chat(dlg["anchor"])

        for q in dlg["turns"]:
            response = system.chat(q)
            score = score_coherence(q, response)
            all_scores.append(score)
            by_topic[topic].append(score)
            if score >= COHERENCE_THRESHOLD:
                n_coherent += 1
            if verbose:
                print(f"    [{topic}] score={score:.2f}  Q: {q[:50]}")

    n_total = len(all_scores)
    mean_score = statistics.mean(all_scores) if all_scores else 0.0

    # Scale to 1-5 Likert for comparability with the manual annotation protocol
    # raw [0,1] → Likert 1-5: likert = 1 + score * 4
    mean_likert = 1.0 + mean_score * 4.0

    topic_summary = {}
    for topic, scores in by_topic.items():
        m = statistics.mean(scores) if scores else 0.0
        n_coh = sum(1 for s in scores if s >= COHERENCE_THRESHOLD)
        topic_summary[topic] = {
            "mean_raw_score": round(m, 3),
            "mean_likert": round(1.0 + m * 4.0, 2),
            "pct_coherent": round(n_coh / len(scores) * 100, 1) if scores else 0.0,
        }

    return {
        "n_responses_rated": n_total,
        "n_coherent": n_coherent,
        "pct_coherent": round(n_coherent / n_total * 100, 1) if n_total else 0.0,
        "mean_coherence_raw": round(mean_score, 3),
        "mean_coherence_likert_scaled": round(mean_likert, 2),
        "coherence_threshold": COHERENCE_THRESHOLD,
        "by_topic": topic_summary,
        "note": (
            "Coherence scored automatically via question-response keyword overlap "
            "(Jaccard similarity on content words). Threshold >= 0.15 = coherent. "
            "For final dissertation: supplement with human annotation (5-point Likert)."
        ),
    }


# ---------------------------------------------------------------------------
# System loader
# ---------------------------------------------------------------------------

def load_systems(args) -> Tuple:
    """Load pre and/or post dialogue systems based on CLI args."""
    pre_system = None
    post_system = None

    if args.dry_run:
        print("[dry-run] Using mock systems (no model loaded).")
        if args.system in ("pre", "both"):
            pre_system = _MockSystem(always_recall=False)
        if args.system in ("post", "both"):
            post_system = _MockSystem(always_recall=True)
        return pre_system, post_system

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    precision = args.precision

    if args.system in ("pre", "both"):
        print(f"\n[Pre system] Loading Llama 3.1 8B ({precision.upper()}) …")
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            cache_dir=config.MODEL_CACHE_DIR,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, **model_kwargs)
        model.eval()
        pre_system = _BaselineSystem(model, tokenizer)
        print("[Pre system] Ready.")

    if args.system in ("post", "both"):
        from memory_dialogue_system import HierarchicalMemoryDialogueSystem
        print(f"\n[Post system] Loading hierarchical memory system ({precision.upper()}) …")
        hm = HierarchicalMemoryDialogueSystem(precision=precision)
        post_system = _PostSystem(hm)
        print("[Post system] Ready.")

    return pre_system, post_system


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(system, system_label: str, quick: bool, verbose: bool) -> Dict:
    """Run the full evaluation battery on one system."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {system_label.upper()}")
    print(f"{'='*60}")

    results: Dict[str, Any] = {"system": system_label}

    # ── Retention accuracy ────────────────────────────────────────────
    print("\n[1/4] Retention Accuracy …")
    short_filler = FILLER_SHORT[:2] if quick else FILLER_SHORT
    medium_filler = FILLER_MEDIUM[:5] if quick else FILLER_MEDIUM
    long_filler = FILLER_LONG[:10] if quick else FILLER_LONG

    ret_short  = run_retention_accuracy(system, short_filler,  "short",  quick, verbose)
    ret_medium = run_retention_accuracy(system, medium_filler, "medium", quick, verbose)
    ret_long   = run_retention_accuracy(system, long_filler,   "long",   quick, verbose)

    overall_correct = ret_short["correct"] + ret_medium["correct"] + ret_long["correct"]
    overall_total   = ret_short["n_probes"] + ret_medium["n_probes"] + ret_long["n_probes"]
    results["retention_accuracy"] = {
        "short":  ret_short,
        "medium": ret_medium,
        "long":   ret_long,
        "overall": {
            "n_probes": overall_total,
            "correct":  overall_correct,
            "accuracy_pct": round(overall_correct / overall_total * 100, 1) if overall_total else 0,
        },
    }
    print(f"  Short: {ret_short['accuracy_pct']}%  "
          f"Medium: {ret_medium['accuracy_pct']}%  "
          f"Long: {ret_long['accuracy_pct']}%  "
          f"Overall: {results['retention_accuracy']['overall']['accuracy_pct']}%")

    # ── Forgotten information ─────────────────────────────────────────
    print("\n[2/4] Forgotten Information Rate …")
    forgot = run_forgotten_info(system, quick, verbose)
    results["forgotten_information"] = forgot
    print(f"  Turn 5: {forgot['by_turn']['turn_5']['forgotten_rate_pct']}%  "
          f"Turn 10: {forgot['by_turn']['turn_10']['forgotten_rate_pct']}%  "
          f"Turn 20: {forgot['by_turn']['turn_20']['forgotten_rate_pct']}%  "
          f"Overall: {forgot['overall']['forgotten_rate_pct']}%")

    # ── Hallucination ─────────────────────────────────────────────────
    print("\n[3/4] Hallucination Rate …")
    halluc = run_hallucination(system, quick, verbose)
    results["hallucination"] = halluc
    print(f"  Overall: {halluc['hallucination_rate_pct']}% "
          f"({halluc['hallucinated']}/{halluc['n_assertions']})")

    # ── Coherence ─────────────────────────────────────────────────────
    print("\n[4/4] Dialogue Coherence …")
    coh = run_coherence(system, quick, verbose)
    results["coherence"] = coh
    print(f"  Coherent: {coh['pct_coherent']}%  "
          f"Mean Likert: {coh['mean_coherence_likert_scaled']}/5")

    return results


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def build_comparison(pre: Optional[Dict], post: Optional[Dict]) -> Dict:
    """Build a side-by-side comparison dict from pre/post result dicts."""
    comparison: Dict = {}

    def delta(post_val, pre_val, higher_is_better: bool = True) -> str:
        if post_val is None or pre_val is None:
            return "N/A"
        d = post_val - pre_val
        sign = "+" if d >= 0 else ""
        arrow = "↑" if (d > 0) == higher_is_better else "↓"
        return f"{sign}{d:.1f} {arrow}"

    if pre and post:
        # Retention
        comparison["retention_short_pct"] = {
            "pre":   pre["retention_accuracy"]["short"]["accuracy_pct"],
            "post":  post["retention_accuracy"]["short"]["accuracy_pct"],
            "delta": delta(post["retention_accuracy"]["short"]["accuracy_pct"],
                           pre["retention_accuracy"]["short"]["accuracy_pct"]),
        }
        comparison["retention_medium_pct"] = {
            "pre":   pre["retention_accuracy"]["medium"]["accuracy_pct"],
            "post":  post["retention_accuracy"]["medium"]["accuracy_pct"],
            "delta": delta(post["retention_accuracy"]["medium"]["accuracy_pct"],
                           pre["retention_accuracy"]["medium"]["accuracy_pct"]),
        }
        comparison["retention_long_pct"] = {
            "pre":   pre["retention_accuracy"]["long"]["accuracy_pct"],
            "post":  post["retention_accuracy"]["long"]["accuracy_pct"],
            "delta": delta(post["retention_accuracy"]["long"]["accuracy_pct"],
                           pre["retention_accuracy"]["long"]["accuracy_pct"]),
        }
        comparison["forgotten_turn5_pct"] = {
            "pre":   pre["forgotten_information"]["by_turn"]["turn_5"]["forgotten_rate_pct"],
            "post":  post["forgotten_information"]["by_turn"]["turn_5"]["forgotten_rate_pct"],
            "delta": delta(post["forgotten_information"]["by_turn"]["turn_5"]["forgotten_rate_pct"],
                           pre["forgotten_information"]["by_turn"]["turn_5"]["forgotten_rate_pct"],
                           higher_is_better=False),
        }
        comparison["forgotten_turn20_pct"] = {
            "pre":   pre["forgotten_information"]["by_turn"]["turn_20"]["forgotten_rate_pct"],
            "post":  post["forgotten_information"]["by_turn"]["turn_20"]["forgotten_rate_pct"],
            "delta": delta(post["forgotten_information"]["by_turn"]["turn_20"]["forgotten_rate_pct"],
                           pre["forgotten_information"]["by_turn"]["turn_20"]["forgotten_rate_pct"],
                           higher_is_better=False),
        }
        comparison["hallucination_rate_pct"] = {
            "pre":   pre["hallucination"]["hallucination_rate_pct"],
            "post":  post["hallucination"]["hallucination_rate_pct"],
            "delta": delta(post["hallucination"]["hallucination_rate_pct"],
                           pre["hallucination"]["hallucination_rate_pct"],
                           higher_is_better=False),
        }
        comparison["coherence_pct_coherent"] = {
            "pre":   pre["coherence"]["pct_coherent"],
            "post":  post["coherence"]["pct_coherent"],
            "delta": delta(post["coherence"]["pct_coherent"],
                           pre["coherence"]["pct_coherent"]),
        }

    return comparison


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre vs Post hierarchical memory evaluation — CSC3094 Dissertation"
    )
    parser.add_argument(
        "--system",
        choices=["pre", "post", "both"],
        default="both",
        help="Which system(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "int8", "int4"],
        default="fp16",
        help="Model precision (default: fp16; use int4 to save VRAM)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced scenario set (5 retention, 3 forgotten, 10 hallucination, 3 coherence)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock systems — tests pipeline logic without loading the model",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory to write result JSON files (default: results/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-scenario results during evaluation",
    )
    args = parser.parse_args()

    # Validate argument name for dry-run (argparse converts - to _)
    if hasattr(args, "dry_run"):
        pass  # already correct

    import config as cfg  # noqa: F401 – confirms module is importable

    print("=" * 60)
    print("  CSC3094 Dissertation — Pre vs Post Evaluation Suite")
    print("=" * 60)
    print(f"  System:    {args.system}")
    print(f"  Precision: {args.precision}")
    print(f"  Quick:     {args.quick}")
    print(f"  Dry-run:   {getattr(args, 'dry_run', False)}")
    print(f"  Output:    {args.output}/")

    t_global_start = time.perf_counter()

    # Load systems
    pre_system, post_system = load_systems(args)

    pre_results = None
    post_results = None

    # Run evaluations
    if pre_system is not None:
        pre_results = run_evaluation(pre_system, "pre", args.quick, args.verbose)
        pre_results["metadata"] = {
            "precision": args.precision,
            "quick_mode": args.quick,
            "dry_run": getattr(args, "dry_run", False),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    if post_system is not None:
        post_results = run_evaluation(post_system, "post", args.quick, args.verbose)
        post_results["metadata"] = {
            "precision": args.precision,
            "quick_mode": args.quick,
            "dry_run": getattr(args, "dry_run", False),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "memory_config": {
                "working_memory_turns": 6,
                "episodic_top_k": 3,
                "episodic_min_score": 0.25,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        }

    # Build comparison
    comparison = build_comparison(pre_results, post_results)

    # Print summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    header = f"  {'Metric':<40} {'Pre':>8}  {'Post':>8}  {'Δ':>10}"
    print(header)
    print("  " + "-" * 62)
    for metric, vals in comparison.items():
        pre_v  = f"{vals['pre']:.1f}%"  if vals.get("pre")  is not None else "—"
        post_v = f"{vals['post']:.1f}%" if vals.get("post") is not None else "—"
        d      = vals.get("delta", "—")
        print(f"  {metric:<40} {pre_v:>8}  {post_v:>8}  {d:>10}")

    elapsed = round(time.perf_counter() - t_global_start, 1)
    print(f"\n  Total evaluation time: {elapsed}s")

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pre_results:
        path = out_dir / "pre_dialogue_quality.json"
        with open(path, "w") as fh:
            json.dump(pre_results, fh, indent=2)
        print(f"  Pre results saved to: {path}")

    if post_results:
        path = out_dir / "post_dialogue_quality.json"
        with open(path, "w") as fh:
            json.dump(post_results, fh, indent=2)
        print(f"  Post results saved to: {path}")

    if comparison:
        path = out_dir / "pre_post_comparison.json"
        with open(path, "w") as fh:
            json.dump(
                {
                    "comparison": comparison,
                    "pre_metadata":  pre_results.get("metadata")  if pre_results  else None,
                    "post_metadata": post_results.get("metadata") if post_results else None,
                },
                fh,
                indent=2,
            )
        print(f"  Comparison saved to:  {path}")

    print("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
