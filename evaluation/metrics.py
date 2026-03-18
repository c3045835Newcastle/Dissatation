"""
Metrics calculation for the baseline dialogue system evaluation.

Metrics computed (aligned with the dissertation objectives):

  retention_accuracy      – percentage of recall probes answered correctly
  cross_session_retention – retention accuracy measured across session boundaries
  within_session_retention– retention accuracy measured within a single session
  coherence_score         – composite score: (correct + 0.5 * noisy) / total
  forgotten_rate          – percentage of probes answered with "I don't know"
  hallucination_rate      – percentage of probes producing hallucinated answers
  context_utilisation     – token usage as a fraction of context window limit
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import math


# ---------------------------------------------------------------------------
# Per-scenario metrics
# ---------------------------------------------------------------------------

def compute_scenario_metrics(scenario_results: List[dict]) -> dict:
    """
    Compute aggregate metrics from a flat list of recall-test result dicts.

    Each result dict must contain:
      outcome  – "correct" | "forgotten" | "hallucination"
      session  – session number in which the probe was taken
    """
    if not scenario_results:
        return {}

    total      = len(scenario_results)
    correct    = sum(1 for r in scenario_results if r["outcome"] == "correct")
    forgotten  = sum(1 for r in scenario_results if r["outcome"] == "forgotten")
    hallucin   = sum(1 for r in scenario_results if r["outcome"] == "hallucination")

    # Coherence: correct answers + half credit for partial/noisy recalls
    # (here "noisy" is a subset of "correct" flagged by response text)
    noisy = sum(1 for r in scenario_results
                if r["outcome"] == "correct" and
                ("think" in r["response"].lower() or "believe" in r["response"].lower()))
    coherence = (correct + 0.5 * noisy) / total if total else 0.0

    # Cross-session vs within-session breakdown
    cross_results  = [r for r in scenario_results if r.get("cross_session", False)]
    within_results = [r for r in scenario_results if not r.get("cross_session", False)]

    cross_acc  = (sum(1 for r in cross_results  if r["outcome"] == "correct") /
                  len(cross_results)) if cross_results else None
    within_acc = (sum(1 for r in within_results if r["outcome"] == "correct") /
                  len(within_results)) if within_results else None

    return {
        "total_probes":              total,
        "correct":                   correct,
        "forgotten":                 forgotten,
        "hallucinations":            hallucin,
        "retention_accuracy":        round(correct / total, 4),
        "forgotten_rate":            round(forgotten / total, 4),
        "hallucination_rate":        round(hallucin / total, 4),
        "coherence_score":           round(min(1.0, coherence), 4),
        "cross_session_retention":   round(cross_acc,  4) if cross_acc  is not None else None,
        "within_session_retention":  round(within_acc, 4) if within_acc is not None else None,
    }


# ---------------------------------------------------------------------------
# Turn-level coherence decay (Scenario 4)
# ---------------------------------------------------------------------------

def compute_decay_curve(labelled_probes: Dict[str, List[dict]]) -> List[dict]:
    """
    Given a dict mapping {label → probe_results}, return a list of points
    suitable for plotting a coherence-decay curve.

    Labels are expected to contain the number of filler turns (e.g.
    "after-0-turns", "after-5-turns", …).
    """
    points = []
    for label, probes in labelled_probes.items():
        # Extract turn count from label, default 0
        try:
            turns = int("".join(c for c in label.split("-")[1] if c.isdigit()))
        except (IndexError, ValueError):
            turns = 0

        total   = len(probes)
        correct = sum(1 for p in probes if p["outcome"] == "correct")
        acc     = correct / total if total else 0.0
        points.append({"label": label, "filler_turns": turns,
                        "accuracy": round(acc, 4), "total": total})

    points.sort(key=lambda x: x["filler_turns"])
    return points


# ---------------------------------------------------------------------------
# Aggregate across all scenarios
# ---------------------------------------------------------------------------

def compute_overall_metrics(all_scenario_metrics: List[dict]) -> dict:
    """Compute grand-average metrics across all five scenarios."""
    keys = [
        "retention_accuracy", "forgotten_rate", "hallucination_rate", "coherence_score"
    ]
    result = {}
    for k in keys:
        vals = [m[k] for m in all_scenario_metrics if m.get(k) is not None]
        result[f"mean_{k}"] = round(sum(vals) / len(vals), 4) if vals else None

    # Cross / within session averages
    cross_vals  = [m["cross_session_retention"]  for m in all_scenario_metrics
                   if m.get("cross_session_retention") is not None]
    within_vals = [m["within_session_retention"] for m in all_scenario_metrics
                   if m.get("within_session_retention") is not None]

    result["mean_cross_session_retention"]  = (round(sum(cross_vals)  / len(cross_vals),  4)
                                                if cross_vals  else None)
    result["mean_within_session_retention"] = (round(sum(within_vals) / len(within_vals), 4)
                                                if within_vals else None)
    return result


# ---------------------------------------------------------------------------
# Confidence interval helper (Wilson score for proportions)
# ---------------------------------------------------------------------------

def wilson_confidence_interval(p: float, n: int, z: float = 1.96) -> tuple:
    """95 % Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return (round(max(0.0, centre - margin), 4),
            round(min(1.0, centre + margin), 4))
