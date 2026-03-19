# Baseline Evaluation Results – Poster Data Summary

**Project**: Design and Evaluation of a Hierarchical Memory Architecture for Improving  
Long-Term Coherence in Local Large Language Model Dialogue Systems  
**Student**: Robin Husbands (230458358) | **Model**: Llama 3.1 8B (BASE, no fine-tuning)  
**Evaluation Date**: 12 March 2026

---

## Overview

This document summarises all data and results generated from evaluating the **baseline**
Llama 3.1 8B dialogue system (standard context-window memory, no persistent or hierarchical memory).
These results provide the reference point against which the hierarchical memory system will be
compared in future work.

---

## 1. System Specifications

| Component | Specification |
|-----------|---------------|
| **Model** | Meta-Llama-3.1-8B (BASE – pre-trained, not instruction-tuned) |
| **Parameters** | 8.03 billion |
| **GPU** | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| **GPU VRAM Used** | 17.2 GB peak |
| **CPU** | AMD Ryzen 9 5950X (16-core / 32-thread) |
| **System RAM** | 64 GB DDR4-3600 |
| **Operating System** | Ubuntu 22.04.4 LTS |
| **Python** | 3.11.8 |
| **PyTorch** | 2.4.1+cu121 |
| **Transformers** | 4.44.2 |
| **Precision** | FP16 (no quantisation) |
| **Context Window (eval)** | 4,096 tokens |
| **Memory Architecture** | **None** – standard context window only |

---

## 2. Evaluation Methodology

### 2.1 Design

Five controlled dialogue scenarios were designed, each consisting of **two sessions**:

- **Session 1** (10 turns) – Facts are introduced naturally through conversation.
- **Session 2** (5–6 turns) – A fresh context window is opened (simulating a new session),  
  and the model is probed on the information introduced in Session 1.

This design directly measures the impact of the baseline system's inability to persist
information across context windows.

### 2.2 Scenarios

| ID | Scenario | Facts Introduced |
|----|----------|-----------------|
| S1 | Personal Identity and Preferences | 5 (name, job, hobby, language, location) |
| S2 | Project Development Tracking | 5 (project name, backend, database, frontend, deployment) |
| S3 | Factual Knowledge + Hallucination | 8 ground-truth facts (transformer, LLM specs, etc.) |
| S4 | Multi-Session Continuity | 5 (research topic, supervisor, deadline, framework, progress) |
| S5 | Coherence Under Information Overload | 7 (app type, budget, timeline, team, tech stack, market, features) |

### 2.3 Metrics

| Metric | Definition |
|--------|------------|
| **Retention Accuracy** | Proportion of introduced facts correctly recalled when probed (binary per fact; key-term matching ≥ 60%) |
| **Dialogue Coherence** | Proportion of responses logically consistent with established context (0.0–1.0 scale; inter-rater κ = 0.81) |
| **Hallucination Rate** | Proportion of factual responses containing incorrect or fabricated information |
| **Forgotten Information** | Proportion of introduced facts not recalled = 1 − Retention Accuracy |
| **Memory Retrieval Rate** | Proportion of relevant past facts successfully retrieved (always 0 for baseline) |

All scenarios were run **three times** (temperature = 0.7); reported values are means across runs.

---

## 3. Baseline Results

### 3.1 Aggregate Metrics

| Metric | Within-Session | Cross-Session |
|--------|---------------|---------------|
| **Retention Accuracy** | **56.1%** | **0.0%** |
| **Dialogue Coherence** | **80.2%** | **19.0%** |
| **Hallucination Rate** | **5.0%** | **36.7%** |
| **Forgotten Information** | 43.9% | **100.0%** |
| **Memory Retrieval Rate** | N/A | **0.0%** |

> **Key finding**: The baseline system retains **zero information** across sessions.  
> Cross-session hallucination rises to **36.7%** as the model generates plausible but  
> factually incorrect responses without prior context.

### 3.2 Per-Scenario Results

| Scenario | W-S Retention | X-S Retention | W-S Coherence | X-S Coherence | Hallucination (S1) | Hallucination (S2) |
|----------|--------------|--------------|--------------|--------------|-------------------|-------------------|
| S1 – Personal Identity | 83.3% | 0.0% | 81.2% | 26.7% | 0.0% | 33.3% |
| S2 – Project Tracking | 80.0% | 0.0% | 84.4% | 21.3% | 0.0% | 40.0% |
| S3 – Factual Knowledge | 0.0%¹ | 0.0% | 87.5% | 12.5% | 25.0% | 50.0% |
| S4 – Multi-Session | 60.0% | 0.0% | 75.6% | 18.7% | 0.0% | 20.0% |
| S5 – Info Overload | 57.1% | 0.0% | 72.2% | 15.6% | 0.0% | 40.0% |

¹ *S3 tests factual knowledge probes, not within-session recall of introduced facts.*

W-S = Within-Session; X-S = Cross-Session

### 3.3 Retention Decay Across Conversation Turns (Within-Session)

| Turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|---|---|---|---|----|
| Retention | 100% | 95% | 91% | 86% | 80% | 74% | 69% | 62% | 58% | 54% |

Even within a single session, retention degrades from 100% to **54%** by turn 10,
demonstrating that even context-window memory is imperfect over longer conversations.

### 3.4 Qualitative Observations

- **S2**: In 2/3 runs the model suggested Flask instead of Django REST Framework when
  asked about the backend in Session 2 (hallucination of a plausible alternative).
- **S3**: The model stated the Llama 3.1 context window was 32K (not 128K) in 1/3 runs.
- **S4**: The model hallucinated the supervisor's name as "Dr Jones" instead of "Dr Smith" in 1/3 runs.
- **S5**: With 7 facts, the model lost track of the £5,000 budget in 2/3 runs and confused
  the EU for UK as target market in 1 run.

---

## 4. Performance Benchmarks

| Metric | Value |
|--------|-------|
| Model load time | 38.4 s |
| Avg. time-to-first-token | 1.34 s |
| Avg. tokens / second | 28.7 tok/s |
| Avg. generation time / turn | 17.8 s |
| Peak GPU memory | 17.2 GB |
| Model disk size | 15.9 GB |
| Total evaluation wall time | ~31.2 minutes |
| Total inference calls | 71 |

---

## 5. Charts for Poster

The following chart files are saved in `results/charts/`:

| File | Description | Suggested Poster Use |
|------|-------------|---------------------|
| `fig1_retention_comparison.png` | Within- vs Cross-session retention per scenario | Core results panel |
| `fig2_coherence_comparison.png` | Within- vs Cross-session dialogue coherence | Core results panel |
| `fig3_hallucination_rates.png` | Hallucination frequency by scenario and session | Core results panel |
| `fig4_retention_over_turns.png` | Retention decay over 10 turns within a session | Methodology/results panel |
| `fig5_aggregate_radar.png` | Radar chart: all aggregate metrics at a glance | Overview/results panel |
| `fig6_performance_table.png` | Full performance and system specification table | System specs panel |

Generate/regenerate with:
```bash
python results/generate_poster_charts.py
```

---

## 6. Key Messages for Poster

### Motivation
LLMs have a hard **context-window limit**. When a session ends, all information is lost.  
Users must repeatedly provide the same context – reducing usability.

### Baseline Findings (this poster)
1. **Cross-session retention = 0%**: Every fact introduced in one session is forgotten when a new session starts.
2. **Within-session retention degrades** from 100% to 54% over 10 turns.
3. **Hallucination increases 7×** when context is absent (5.0% → 36.7%).
4. **Dialogue coherence drops by 76%** across sessions (80.2% → 19.0%).

### Future Work (Hierarchical Memory System)
The hierarchical memory architecture under development adds:
- **Working memory**: short-term buffer for recent turns
- **Episodic memory**: vector database (FAISS) storing past interactions, retrieved by similarity
- **Semantic memory**: persistent store for user-specific facts

Expected improvements: cross-session retention > 70%, hallucination rate < 10%.

---

## 7. Reproducibility

To reproduce these results:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Log in to Hugging Face (access to Meta-Llama-3.1-8B required)
huggingface-cli login

# 3. Run the baseline evaluation
python evaluation/run_baseline_evaluation.py
```

Results will be written to `results/baseline_results.json`.

---

*Evaluation conducted on 12 March 2026 using the methodology defined in*  
*`evaluation/evaluation_framework.py` and scenarios in `evaluation/test_scenarios.json`.*
