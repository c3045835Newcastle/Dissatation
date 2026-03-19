"""
Run the baseline evaluation against the live Llama 3.1 8B model.

Usage
-----
    python evaluation/run_baseline_evaluation.py

Prerequisites
-------------
    1. Install dependencies:   pip install -r requirements.txt
    2. Log in to Hugging Face:  huggingface-cli login
    3. Model will be downloaded on first run (~16 GB).

Output
------
    results/baseline_results.json  – full results file
    results/poster_data_summary.md – human-readable summary for the poster

Note
----
    If you do not have the model available, pre-computed results already
    exist in results/baseline_results.json and can be used directly for
    the poster.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation_framework import DialogueEvaluator


def main():
    print("=" * 60)
    print("Baseline Llama 3.1 8B – Evaluation Run")
    print("=" * 60)
    print()
    print("Loading model…  (this may take several minutes on first run)")
    print()

    # Import here so the script fails gracefully if the model is not available
    try:
        from llama_base_model import BaseLlama31Model
        model = BaseLlama31Model()
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}")
        print()
        print("To use pre-computed results instead, see results/baseline_results.json")
        sys.exit(1)

    evaluator = DialogueEvaluator(model=model)

    scenarios_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_scenarios.json"
    )

    print("\nRunning evaluation scenarios…\n")
    report = evaluator.run_all_scenarios(scenarios_path)

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "baseline_results.json",
    )
    evaluator.save_results(report, output_path)

    _print_summary(report)


def _print_summary(report):
    m = report.aggregate_metrics
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model  : {report.model_name}")
    print(f"Date   : {report.evaluation_date}")
    print()
    print("Aggregate Metrics:")
    print(f"  Within-Session Retention Accuracy  : {m.get('mean_within_session_retention', 0):.1%}")
    print(f"  Cross-Session  Retention Accuracy  : {m.get('mean_cross_session_retention', 0):.1%}")
    print(f"  Within-Session Dialogue Coherence  : {m.get('mean_within_session_coherence', 0):.1%}")
    print(f"  Cross-Session  Dialogue Coherence  : {m.get('mean_cross_session_coherence', 0):.1%}")
    print(f"  Within-Session Hallucination Rate  : {m.get('mean_within_session_hallucination', 0):.1%}")
    print(f"  Cross-Session  Hallucination Rate  : {m.get('mean_cross_session_hallucination', 0):.1%}")
    print(f"  Forgotten Info Rate (cross-session): {m.get('mean_forgotten_cross_session', 0):.1%}")
    print(f"  Memory Retrieval Rate              : {m.get('memory_retrieval_rate', 0):.1%}")
    print()
    print("Scenario breakdown:")
    for sr in report.scenario_results:
        print(f"  [{sr.scenario_id}] {sr.scenario_name}")
        print(f"       Within-session retention : {sr.within_session_retention:.1%}")
        print(f"       Cross-session  retention : {sr.cross_session_retention:.1%}")
        print(f"       Hallucination  (session1): {sr.within_session_hallucination:.1%}")


if __name__ == "__main__":
    main()
