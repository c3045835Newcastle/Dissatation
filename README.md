# Dissertation – Hierarchical Memory Architecture for LLM Dialogue Systems

**Title:** Design and Evaluation of a Hierarchical Memory Architecture for Improving
Long-Term Coherence in Local Large Language Model Dialogue Systems

**Author:** Robin Husbands (230458358)

---

## Overview

This repository implements the full system described in the CSC3094 dissertation proposal.
It extends a locally-deployed Llama 3.1 8B baseline with a **hierarchical memory
architecture** inspired by Baddeley's (1992) cognitive memory model and evaluated
using retrieval-augmented generation techniques (Lewis et al., 2020).

---

## Dissertation Objectives → Implementation

| # | Proposal Objective | Implementation |
|---|-------------------|----------------|
| 1 | Baseline dialogue system (10+ turns, no persistent memory) | `llama_base_model.py`, `inference.py` |
| 2 | Hierarchical memory: working / episodic / semantic | `memory/` package |
| 3 | Memory controller for storage & consolidation | `memory/memory_controller.py` |
| 4 | Integrate memory into dialogue pipeline | `dialogue_pipeline.py` |
| 5 | 5 controlled multi-session evaluation scenarios | `evaluation/evaluation_scenarios.py` |
| 6 | Evaluation metrics (retention, coherence, hallucination…) | `evaluation/metrics.py` |
| 7 | Analysis & results | `results/`, `BENCHMARK_RESULTS.md` |

---

## Project Structure

```
.
├── config.py                        # Model + memory configuration
├── llama_base_model.py              # Baseline Llama 3.1 8B wrapper (Obj. 1)
├── inference.py                     # Interactive baseline inference
├── dialogue_pipeline.py             # Hierarchical memory pipeline (Obj. 4)
│
├── memory/                          # Hierarchical memory modules (Obj. 2 & 3)
│   ├── __init__.py
│   ├── working_memory.py            # Sliding-window recent-context store (Obj. 2a)
│   ├── episodic_memory.py           # FAISS vector-search past-interaction store (Obj. 2b)
│   ├── semantic_memory.py           # Persistent user-fact store (Obj. 2c)
│   └── memory_controller.py        # Storage & consolidation controller (Obj. 3)
│
├── evaluation/                      # Evaluation framework (Obj. 5 & 6)
│   ├── __init__.py
│   ├── evaluation_scenarios.py     # 5 controlled multi-session scenarios (Obj. 5)
│   └── metrics.py                  # Retention, coherence, hallucination metrics (Obj. 6)
│
├── benchmark.py                     # Throughput & perplexity benchmarking
├── examples.py                      # Usage examples
├── test_setup.py                    # Setup validation tests
├── requirements.txt                 # Python dependencies
└── CSC3094 Dissertation Proposal (1).pdf
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
huggingface-cli login   # required for Llama model access
```

### Run baseline (Objective 1)

```bash
python inference.py
```

### Run hierarchical memory pipeline (Objectives 2–4)

```python
from dialogue_pipeline import HierarchicalMemoryPipeline

pipeline = HierarchicalMemoryPipeline(
    session_id="session_1",
    load_model=True,          # loads Llama 3.1 8B
)

response = pipeline.chat("My name is Alice and I am a software engineer.")
print(response)

# Later in a new session
pipeline2 = HierarchicalMemoryPipeline(
    session_id="session_2",
    load_model=True,
)
response = pipeline2.chat("What is my name?")
print(response)   # Should recall 'Alice' from semantic memory

pipeline.save_memory()
```

### Run evaluation scenarios (Objectives 5–6)

```python
from dialogue_pipeline import HierarchicalMemoryPipeline
from evaluation import EvaluationScenarios, EvaluationRunner

pipeline = HierarchicalMemoryPipeline(load_model=True)

for scenario in EvaluationScenarios.get_all():
    runner = EvaluationRunner(pipeline, scenario, verbose=True)
    metrics = runner.run()
    report = metrics.compute()
    print(metrics.format_report(report))
```

### Validate setup (no LLM needed)

```bash
python test_setup.py
```

---

## Memory Architecture

### Working Memory (`memory/working_memory.py`)
A deque-based sliding window of the most recent dialogue turns.
Keeps the last `WORKING_MEMORY_MAX_TURNS` (default 20) turns active in
context, satisfying the ≥ 10-turn requirement of Objective 1.

### Episodic Memory (`memory/episodic_memory.py`)
Past interaction summaries stored in a FAISS flat L2 index (Johnson et al., 2017).
A `sentence-transformers` model (`all-MiniLM-L6-v2`) provides 384-dimensional
embeddings.  The top-K most relevant episodes are retrieved at each turn via
cosine similarity and injected into the model context (retrieval-augmented
generation, Lewis et al., 2020).  Degrades gracefully to keyword matching when
`faiss-cpu` is not installed.

### Semantic Memory (`memory/semantic_memory.py`)
A JSON-backed key–value store for persistent user facts (name, occupation,
preferences, goals).  Organised into categories (`personal`, `preferences`,
`goals`) and persisted across sessions.  Inspired by the semantic/episodic
memory distinction in Baddeley (1992).

### Memory Controller (`memory/memory_controller.py`)
Coordinates the three tiers:
- Extracts personal facts from user messages via regex patterns → semantic memory.
- Consolidates working-memory windows into episodic episodes every N turns.
- Builds the enriched context string prepended to every model request.

---

## Evaluation Metrics (Objective 6)

| Metric | Description |
|--------|-------------|
| Retention Accuracy | % of expected facts recalled in model response |
| Dialogue Coherence | Consistency of responses across sessions |
| Memory Retrieval Consistency | % of relevant past episodes retrieved |
| Forgotten Information Rate | Avg. number of unrecalled facts per turn |
| Error Detection Rate | Fraction of turns where factual errors were flagged |
| Hallucination Frequency | Avg. number of fabricated facts per turn |

---

## Configuration (`config.py`)

Key memory parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `WORKING_MEMORY_MAX_TURNS` | 20 | Sliding window size (≥ 10 per Obj. 1) |
| `EPISODIC_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for FAISS |
| `EPISODIC_TOP_K` | 5 | Episodes retrieved per query |
| `MEMORY_CONSOLIDATION_INTERVAL` | 5 | Working → episodic consolidation frequency |
| `EVALUATION_NUM_SCENARIOS` | 5 | Evaluation scenarios (≥ 5 per Obj. 5) |

---

## References

- Baddeley, A.D., 1992. Working memory. *Science*, 255(5044), pp.556–559.
- Johnson, J., Douze, M. and Jégou, H., 2017. Billion-scale similarity search with GPUs. *IEEE Trans. Big Data*, 7(3).
- Lewis, P. et al., 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*, 33.
- Touvron, H. et al., 2023. LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*.
- Vaswani, A. et al., 2017. Attention is all you need. *NeurIPS*, 30.