"""
Main evaluation runner for the baseline dialogue system.

Usage
-----
    python evaluation/run_evaluation.py

Outputs
-------
  results/baseline_results.json   – full structured results
  results/baseline_summary.json   – aggregated metrics per scenario + overall
  results/baseline_results.csv    – flat CSV of every recall probe result
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baseline_system import BaselineDialogueSystem, SimulatedLLM
from evaluation.test_scenarios import ALL_SCENARIOS
from evaluation.metrics import (
    compute_scenario_metrics,
    compute_decay_curve,
    compute_overall_metrics,
    wilson_confidence_interval,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_scenario(scenario_def: dict, seed: int = 42) -> dict:
    """
    Execute one evaluation scenario and return its full result dict.
    """
    context_limit = scenario_def.get("context_limit", 4096)
    llm    = SimulatedLLM(seed=seed)
    system = BaselineDialogueSystem(context_limit=context_limit, llm=llm)
    facts  = scenario_def["facts"]

    all_probe_results: list  = []
    labelled_probes:   dict  = {}  # for decay-curve scenarios
    current_session = 1

    for phase in scenario_def["phases"]:
        target_session = phase["session"]

        # Advance to the required session
        while current_session < target_session:
            system.new_session()
            current_session += 1

        action = phase["action"]

        if action == "introduce_all":
            for fact in facts:
                system.introduce_fact(fact)

        elif action == "filler":
            system.fill_context_with_filler(num_turns=phase.get("turns", 10))

        elif action == "probe_all":
            label = phase.get("label", f"session-{current_session}")
            phase_results = []
            for fact in facts:
                result = system.test_recall(fact)
                result["phase_label"]   = label
                # A probe is cross-session if facts were introduced in session 1
                # and this probe is in a later session
                result["cross_session"] = current_session > 1
                all_probe_results.append(result)
                phase_results.append(result)

            labelled_probes[label] = phase_results

    metrics = compute_scenario_metrics(all_probe_results)

    # Compute confidence intervals for the main metric
    n = metrics["total_probes"]
    metrics["retention_accuracy_ci_95"] = wilson_confidence_interval(
        metrics["retention_accuracy"], n
    )
    metrics["hallucination_rate_ci_95"] = wilson_confidence_interval(
        metrics["hallucination_rate"], n
    )

    decay_curve = compute_decay_curve(labelled_probes) if len(labelled_probes) > 1 else []

    return {
        "scenario_id":      scenario_def["id"],
        "scenario_name":    scenario_def["name"],
        "scenario_desc":    scenario_def["description"],
        "num_sessions":     scenario_def["sessions"],
        "num_facts":        len(facts),
        "probe_results":    all_probe_results,
        "labelled_probes":  {k: v for k, v in labelled_probes.items()},
        "metrics":          metrics,
        "decay_curve":      decay_curve,
        "context_tokens_used": system.state.tokens_used,
        "context_limit":       system.state.context_token_limit,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 65)
    print("  Baseline Dialogue System – Evaluation")
    print("  Robin Husbands, 230458358 | Newcastle University")
    print("=" * 65)

    all_results = []
    all_scenario_metrics = []

    for scenario_fn in ALL_SCENARIOS:
        scenario_def = scenario_fn()
        print(f"\n[Scenario {scenario_def['id']}] {scenario_def['name']}")
        print(f"  {scenario_def['description'][:80]}…")

        result = run_scenario(scenario_def, seed=42)
        all_results.append(result)
        all_scenario_metrics.append(result["metrics"])

        m = result["metrics"]
        print(f"  Retention accuracy : {m['retention_accuracy']:.1%}")
        print(f"  Forgotten rate     : {m['forgotten_rate']:.1%}")
        print(f"  Hallucination rate : {m['hallucination_rate']:.1%}")
        print(f"  Coherence score    : {m['coherence_score']:.3f}")
        if m.get("cross_session_retention") is not None:
            print(f"  Cross-session ret. : {m['cross_session_retention']:.1%}")
        if m.get("within_session_retention") is not None:
            print(f"  Within-session ret.: {m['within_session_retention']:.1%}")

    # ---- Overall summary --------------------------------------------------
    overall = compute_overall_metrics(all_scenario_metrics)
    print("\n" + "=" * 65)
    print("  OVERALL BASELINE METRICS")
    print("=" * 65)
    for k, v in overall.items():
        if v is not None:
            label = k.replace("mean_", "").replace("_", " ").title()
            print(f"  {label:<40} {v:.1%}" if v <= 1.0 else f"  {label:<40} {v}")

    # ---- Persist results --------------------------------------------------
    full_output = {
        "metadata": {
            "system":          "Baseline – LLaMA 3.1 8B (context-window only)",
            "author":          "Robin Husbands, 230458358",
            "university":      "Newcastle University",
            "module":          "CSC3094 Dissertation",
            "evaluation_date": time.strftime("%Y-%m-%d"),
            "num_scenarios":   len(ALL_SCENARIOS),
        },
        "scenarios": all_results,
        "overall":   overall,
    }

    results_path  = os.path.join(RESULTS_DIR, "baseline_results.json")
    summary_path  = os.path.join(RESULTS_DIR, "baseline_summary.json")
    csv_path      = os.path.join(RESULTS_DIR, "baseline_results.csv")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    summary = {
        "metadata":        full_output["metadata"],
        "per_scenario":    [
            {
                "id":   r["scenario_id"],
                "name": r["scenario_name"],
                **r["metrics"],
                "decay_curve": r["decay_curve"],
            }
            for r in all_results
        ],
        "overall": overall,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV
    csv_fields = [
        "scenario_id", "scenario_name", "phase_label", "session",
        "turn", "fact_key", "fact_value", "outcome", "in_context", "response",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            for probe in r["probe_results"]:
                row = {
                    "scenario_id":   r["scenario_id"],
                    "scenario_name": r["scenario_name"],
                    **probe,
                }
                writer.writerow(row)

    print(f"\n  Results saved to {RESULTS_DIR}/")
    print(f"    baseline_results.json  (full detail)")
    print(f"    baseline_summary.json  (metrics summary)")
    print(f"    baseline_results.csv   (flat probe log)")
    print()


if __name__ == "__main__":
    main()
