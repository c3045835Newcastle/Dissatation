# Dissertation – Hierarchical Memory Architecture for LLM Dialogue Systems

**Title:** Design and Evaluation of a Hierarchical Memory Architecture for Improving
Long-Term Coherence in Local Large Language Model Dialogue Systems

**Author:** Robin Husbands (230458358)

---

## Overview

This repository implements a hierarchical memory architecture for locally-deployed
Llama 3.1 8B dialogue systems. It extends the baseline model with a **hierarchical
memory architecture** inspired by Baddeley's (1992) cognitive memory model and
evaluated using retrieval-augmented generation techniques (Lewis et al., 2020).

---

## Project Structure

```
.
├── config.py                        # Model + memory configuration
├── llama_base_model.py              # Baseline Llama 3.1 8B wrapper
├── inference.py                     # Interactive baseline inference
├── dialogue_pipeline.py             # Hierarchical memory pipeline
│
├── memory/                          # Hierarchical memory modules
│   ├── __init__.py
│   ├── working_memory.py            # Sliding-window recent-context store
│   ├── episodic_memory.py           # FAISS vector-search past-interaction store
│   ├── semantic_memory.py           # Persistent user-fact store
│   └── memory_controller.py        # Storage & consolidation controller
│
├── evaluation/                      # Evaluation framework
│   ├── __init__.py
│   ├── evaluation_scenarios.py     # 5 controlled multi-session scenarios
│   └── metrics.py                  # Retention, coherence, hallucination metrics
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

### Run baseline

```bash
python inference.py
```

### Run hierarchical memory pipeline

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

### Run evaluation scenarios

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
Keeps the last `WORKING_MEMORY_MAX_TURNS` (default 20) turns active in context.

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

## Evaluation Metrics

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
| `WORKING_MEMORY_MAX_TURNS` | 20 | Sliding window size |
| `EPISODIC_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for FAISS |
| `EPISODIC_TOP_K` | 5 | Episodes retrieved per query |
| `MEMORY_CONSOLIDATION_INTERVAL` | 5 | Working → episodic consolidation frequency |
| `EVALUATION_NUM_SCENARIOS` | 5 | Number of evaluation scenarios |

---

## References

- Baddeley, A.D., 1992. Working memory. *Science*, 255(5044), pp.556–559.
- Johnson, J., Douze, M. and Jégou, H., 2017. Billion-scale similarity search with GPUs. *IEEE Trans. Big Data*, 7(3).
- Lewis, P. et al., 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*, 33.
- Touvron, H. et al., 2023. LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*.
- Vaswani, A. et al., 2017. Attention is all you need. *NeurIPS*, 30.