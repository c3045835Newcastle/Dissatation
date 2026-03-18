# Design and Evaluation of a Hierarchical Memory Architecture for Improving Long-Term Coherence in Local Large Language Model Dialogue Systems

**Robin Husbands · 230458358 · Newcastle University · CSC3094 Dissertation**

---

## Project Overview

This project investigates whether a **hierarchical memory architecture** (working memory,
episodic memory, semantic memory) can improve long-term dialogue coherence in a locally
deployed LLaMA 3.1 8B dialogue system compared to a **baseline context-window-only system**.

This repository currently contains the **baseline system evaluation framework** and all
poster-ready data and figures for the baseline system.  The hierarchical memory system will
be implemented and compared in future work.

---

## Repository Structure

```
.
├── src/
│   └── baseline_system.py          # Baseline dialogue system (context-window only)
├── evaluation/
│   ├── test_scenarios.py           # 5 controlled multi-session test scenarios
│   ├── metrics.py                  # Metrics calculation (retention, coherence, etc.)
│   └── run_evaluation.py           # Main evaluation runner
├── results/
│   ├── baseline_results.json       # Full per-probe results (all scenarios)
│   ├── baseline_summary.json       # Aggregated metrics per scenario + overall
│   └── baseline_results.csv        # Flat CSV of every recall probe
├── poster_figures/
│   ├── generate_figures.py         # Generates all poster figures from results data
│   ├── fig1_scenario_retention.png # Retention accuracy per scenario (bar chart)
│   ├── fig2_outcome_distribution.png# Overall outcome breakdown (pie chart)
│   ├── fig3_coherence_decay.png    # Coherence drift over filler turns (line chart)
│   ├── fig4_within_vs_cross.png    # Within-session vs cross-session (grouped bar)
│   ├── fig5_outcome_breakdown.png  # Stacked outcome breakdown per scenario
│   └── fig6_summary_table.png      # Full metrics summary table
├── requirements.txt
└── CSC3094 Dissertation Proposal (1).pdf
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the evaluation

```bash
python evaluation/run_evaluation.py
```

This runs all five test scenarios against the simulated baseline system and writes
results to `results/`.

### 3. Regenerate poster figures

```bash
python poster_figures/generate_figures.py
```

All six figures are saved as high-resolution PNGs in `poster_figures/`.

---

## Evaluation Scenarios

| # | Scenario | Sessions | Facts | Key Finding |
|---|----------|----------|-------|-------------|
| 1 | Cross-Session Amnesia | 2 | 5 | 0 % recall after session restart |
| 2 | Context Window Overflow | 1 | 7 | 0 % recall once context window fills |
| 3 | Multi-Session Coherence | 3 | 4 | 100 % within-session → 0 % cross-session |
| 4 | Long-Session Coherence Drift | 1 | 6 | Smooth decay from 100 % → 0 % as turns increase |
| 5 | Rapid Fact Introduction | 2 | 8 | 100 % immediate recall → 0 % after session reset |

---

## Key Baseline Metrics

| Metric | Value |
|--------|-------|
| Overall Retention Accuracy | 28.7 % |
| Within-Session Retention | 65.0 % |
| Cross-Session Retention | **0.0 %** |
| Forgotten Rate | 33.5 % |
| Hallucination Rate | 37.8 % |
| Coherence Score | 30.3 % |

These results demonstrate the core limitation of a context-window-only system:
while within-session performance is acceptable when the context does not overflow,
**the system retains zero information across session boundaries**.  The hierarchical
memory system (future work) is designed to directly address this.

---

## Metrics Definitions

| Metric | Definition |
|--------|------------|
| **Retention accuracy** | Percentage of recall probes answered correctly |
| **Forgotten rate** | Percentage of probes where system reported no recollection |
| **Hallucination rate** | Percentage of probes where system produced a plausible but incorrect answer |
| **Coherence score** | `(correct + 0.5 × noisy) / total` – credits partial recalls |
| **Within-session retention** | Retention accuracy for probes in the same session as fact introduction |
| **Cross-session retention** | Retention accuracy for probes in a later session than fact introduction |

---

## Poster Figures

| File | Description |
|------|-------------|
| `fig1_scenario_retention.png` | Bar chart of retention accuracy per scenario with 95 % confidence intervals |
| `fig2_outcome_distribution.png` | Pie chart of overall correct / forgotten / hallucination split |
| `fig3_coherence_decay.png` | Line chart showing accuracy decay as filler turns increase |
| `fig4_within_vs_cross.png` | Grouped bar comparing within-session vs cross-session accuracy |
| `fig5_outcome_breakdown.png` | Stacked bar showing outcome breakdown per scenario |
| `fig6_summary_table.png` | Complete metrics summary table suitable for inclusion in a poster |

---

## Future Work

The next phase of this project will implement the **hierarchical memory architecture**:

- **Working memory** – recent conversation context (current context window)
- **Episodic memory** – past interactions stored and retrieved via FAISS vector similarity search
- **Semantic memory** – persistent user facts stored in a structured key-value store
- **Memory controller** – decides what to store, consolidate, and retrieve

The same five evaluation scenarios will be re-run against the hierarchical system, and the
results will be compared directly against the baseline figures produced here.

---

## References

- Vaswani et al. (2017) – Attention Is All You Need
- Touvron et al. (2023) – LLaMA: Open and Efficient Foundation Language Models
- Lewis et al. (2020) – Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Johnson et al. (2017) – Billion-Scale Similarity Search with GPUs (FAISS)
- Baddeley (1992) – Working Memory
- Bubeck et al. (2023) – Sparks of Artificial General Intelligence