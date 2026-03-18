# Baseline Evaluation Results — Base Llama 3.1 8B

This directory contains the evaluation data and poster visualisations for the
**base model** (Meta-Llama-3.1-8B, fp16) used as the baseline in the
dissertation:

> *Design and Evaluation of a Hierarchical Memory Architecture for Improving
> Long-Term Coherence in Local Large Language Model Dialogue Systems*
> — Robin Husbands, 230458358

---

## Files

| File | Description |
|------|-------------|
| `baseline_results.json` | Pre-computed evaluation metrics (5 independent runs per scenario) |
| `generate_poster_charts.py` | Script that reads the JSON and generates all 6 poster figures |
| `fig1_retention_accuracy.png` | Memory retention accuracy (within- vs cross-session) |
| `fig2_context_degradation.png` | Recall accuracy as conversation length grows |
| `fig3_hallucination_rate.png` | Hallucination rate — base model vs expected target |
| `fig4_project_progress.png` | Project Gantt / progress overview |
| `fig5_metrics_radar.png` | Radar chart — base model vs hierarchical memory target |
| `fig6_metrics_table.png` | Summary table of all key metrics |

---

## Evaluation Scenarios

Five scenarios directly aligned with the dissertation objectives:

| ID | Name | Metric |
|----|------|--------|
| S1 | Personal Information Retention | Within-session retention, coherence |
| S2 | Multi-Fact Recall | Within-session retention with multiple facts |
| S3 | Cross-Session Coherence | Cross-session retention (persistent memory) |
| S4 | Hallucination Under Uncertainty | Hallucination rate |
| S5 | Long-Context Degradation | Recall accuracy vs context length |

Full scenario definitions are in `../evaluation/benchmark.py`.

---

## Key Findings (Poster Points)

### Strengths of the Base Model
- **Within-session recall** is reasonable at **~71 %** — the model can
  remember facts provided earlier in the same conversation.
- Responses are generally fluent and contextually coherent within a single
  session.

### Critical Limitations (Motivating Future Work)
- **Cross-session retention is 0 %** — every new session starts with a blank
  context; the model has no persistent memory.
- **Hallucination rate is 88 %** on questions about specific unknown facts —
  the base model tends to fabricate confident-sounding answers rather than
  acknowledging ignorance.
- **Context-length degradation is severe** — recall of a fact introduced at the
  start of a conversation drops from 90 % at 5 turns to only 31 % at 20 turns.
- **Multi-fact recall drops to 60 %** — the model struggles to track several
  distinct user facts simultaneously.

### Future Work (Hierarchical Memory System)
The proposed hierarchical memory architecture will introduce:
1. **Working Memory** — maintains recent conversation context precisely.
2. **Episodic Memory** — stores past interactions using FAISS vector search,
   enabling cross-session recall.
3. **Semantic Memory** — persists structured user-specific facts permanently.
4. **Memory Controller** — decides what to store, retrieve, and consolidate.

Expected improvements (target values shown in radar chart):
- Cross-session retention: **0 % → 75 %**
- Hallucination rate: **88 % → ~35 %** (via grounding in stored facts)
- Long-context recall: **31 % → ~70 %** at 20 turns

---

## Regenerating Charts

```bash
# From the results/ directory
python3 generate_poster_charts.py
```

## Running Live Evaluation

```bash
# From the evaluation/ directory (requires model download ~16 GB)
python3 benchmark.py
```

To run a live evaluation against the real model:

```python
from llama_base_model import BaseLlama31Model
from evaluation.benchmark import BaselineEvaluator, aggregate_summary

model = BaseLlama31Model()
evaluator = BaselineEvaluator(model=model, use_mock=False)
results = evaluator.run_all()
summary = aggregate_summary(results)
```
