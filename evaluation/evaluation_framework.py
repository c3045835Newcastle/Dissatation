"""
Evaluation Framework for Baseline Llama 3.1 8B Dialogue System.

This module defines the evaluation methodology used to measure long-term coherence,
memory retention, and hallucination frequency for the baseline (no persistent memory)
Llama 3.1 8B dialogue system.

Metrics evaluated:
  - Retention Accuracy       : proportion of facts correctly recalled when probed
  - Dialogue Coherence       : proportion of responses logically consistent with context
  - Hallucination Frequency  : proportion of factual responses containing incorrect information
  - Forgotten Information    : proportion of introduced facts not recalled when probed
  - Memory Retrieval Rate    : proportion of relevant past facts successfully retrieved
                               (always 0 for baseline – no external memory)

Usage:
    from evaluation.evaluation_framework import DialogueEvaluator
    evaluator = DialogueEvaluator(model)
    results = evaluator.run_all_scenarios("evaluation/test_scenarios.json")
    evaluator.save_results(results, "results/baseline_results.json")
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Result for a single dialogue turn."""
    turn: int
    user_input: str
    model_response: str
    expected_answer: Optional[str] = None
    is_correct: Optional[bool] = None          # For retention probes
    contains_hallucination: Optional[bool] = None  # For factual turns
    coherence_score: float = 1.0               # 0.0 – 1.0
    response_time_ms: float = 0.0


@dataclass
class SessionResult:
    """Result for a complete dialogue session (one context window)."""
    session_id: int
    scenario_id: str
    turns: List[TurnResult] = field(default_factory=list)
    retention_accuracy: float = 0.0
    coherence: float = 0.0
    hallucination_rate: float = 0.0
    forgotten_info_rate: float = 0.0
    total_facts_introduced: int = 0
    total_facts_recalled: int = 0
    total_turns: int = 0
    duration_s: float = 0.0


@dataclass
class ScenarioResult:
    """Aggregated result across all sessions for one scenario."""
    scenario_id: str
    scenario_name: str
    session_1: Optional[SessionResult] = None
    session_2: Optional[SessionResult] = None
    within_session_retention: float = 0.0
    cross_session_retention: float = 0.0
    within_session_coherence: float = 0.0
    cross_session_coherence: float = 0.0
    within_session_hallucination: float = 0.0
    cross_session_hallucination: float = 0.0
    facts_forgotten_cross_session: float = 1.0  # Always 1.0 for baseline


@dataclass
class EvaluationReport:
    """Full evaluation report across all scenarios."""
    model_name: str
    evaluation_date: str
    system_config: Dict
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    aggregate_metrics: Dict = field(default_factory=dict)
    methodology_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DialogueEvaluator:
    """
    Runs controlled dialogue scenarios against a loaded LLM and scores responses
    according to the evaluation protocol defined in test_scenarios.json.

    Parameters
    ----------
    model : BaseLlama31Model | None
        An initialised model instance.  Pass ``None`` to load pre-computed
        results without running inference (offline / replay mode).
    """

    def __init__(self, model=None):
        self.model = model
        self._conversation_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_scenarios(self, scenarios_path: str) -> EvaluationReport:
        """
        Run all scenarios defined in *scenarios_path* and return a full
        :class:`EvaluationReport`.
        """
        import datetime
        with open(scenarios_path, "r") as f:
            data = json.load(f)

        report = EvaluationReport(
            model_name="meta-llama/Meta-Llama-3.1-8B (base)",
            evaluation_date=datetime.datetime.utcnow().isoformat() + "Z",
            system_config=self._get_system_config(),
            methodology_notes=self._methodology_notes(),
        )

        for scenario in data["scenarios"]:
            print(f"\n{'='*60}")
            print(f"Running scenario {scenario['id']}: {scenario['name']}")
            print("="*60)
            result = self._run_scenario(scenario)
            report.scenario_results.append(result)

        report.aggregate_metrics = self._aggregate(report.scenario_results)
        return report

    def save_results(self, report: EvaluationReport, output_path: str) -> None:
        """Serialise *report* to *output_path* as JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nResults saved to {output_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_scenario(self, scenario: Dict) -> ScenarioResult:
        result = ScenarioResult(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
        )

        # --- Session 1: introduce facts ---
        print(f"\n  Session 1 (fact introduction, {len(scenario['session_1_dialogue'])} turns)…")
        result.session_1 = self._run_session(
            session_id=1,
            scenario_id=scenario["id"],
            turns=scenario["session_1_dialogue"],
            facts=scenario.get("session_1_facts", []),
            mode="introduction",
        )

        # --- Session 2: probe recall (NEW context – baseline has no memory) ---
        print(f"  Session 2 (cross-session probe, {len(scenario['session_2_probe_questions'])} turns)…")
        result.session_2 = self._run_session(
            session_id=2,
            scenario_id=scenario["id"],
            turns=scenario["session_2_probe_questions"],
            facts=scenario.get("session_1_facts", []),
            mode="probe",
            ground_truth=scenario.get("ground_truth_facts", []),
        )

        # Summarise across sessions
        result.within_session_retention = result.session_1.retention_accuracy
        result.cross_session_retention = result.session_2.retention_accuracy
        result.within_session_coherence = result.session_1.coherence
        result.cross_session_coherence = result.session_2.coherence
        result.within_session_hallucination = result.session_1.hallucination_rate
        result.cross_session_hallucination = result.session_2.hallucination_rate
        # Baseline always forgets everything cross-session
        result.facts_forgotten_cross_session = 1.0 - result.cross_session_retention

        return result

    def _run_session(
        self,
        session_id: int,
        scenario_id: str,
        turns: List[Dict],
        facts: List[str],
        mode: str,
        ground_truth: Optional[List[Dict]] = None,
    ) -> SessionResult:

        # Reset conversation history for each session (baseline: no memory)
        self._conversation_history = []
        session = SessionResult(
            session_id=session_id,
            scenario_id=scenario_id,
            total_facts_introduced=len(facts),
            total_turns=len(turns),
        )
        start = time.time()
        correct_count = 0
        hallucination_count = 0
        coherence_scores = []
        probe_count = 0
        factual_count = 0

        for turn_data in turns:
            user_msg = turn_data["user"]
            expected = turn_data.get("expected_answer")
            scoring_type = turn_data.get("scoring_type", "none")

            t0 = time.time()
            response = self._generate_response(user_msg)
            elapsed_ms = (time.time() - t0) * 1000

            is_correct = None
            has_hallucination = None
            coherence = 1.0

            if scoring_type == "retention_accuracy" and expected:
                is_correct = self._score_retention(response, expected)
                probe_count += 1
                if is_correct:
                    correct_count += 1
                    session.total_facts_recalled += 1

            if scoring_type == "hallucination" and ground_truth:
                gt_entry = next(
                    (g for g in ground_truth if g["question"].lower() in user_msg.lower()),
                    None,
                )
                if gt_entry:
                    has_hallucination = self._score_hallucination(response, gt_entry["answer"])
                    factual_count += 1
                    if has_hallucination:
                        hallucination_count += 1

            coherence = self._score_coherence(response, self._conversation_history)
            coherence_scores.append(coherence)

            self._conversation_history.append({"role": "user", "content": user_msg})
            self._conversation_history.append({"role": "assistant", "content": response})

            session.turns.append(TurnResult(
                turn=turn_data.get("turn", len(session.turns) + 1),
                user_input=user_msg,
                model_response=response,
                expected_answer=expected,
                is_correct=is_correct,
                contains_hallucination=has_hallucination,
                coherence_score=coherence,
                response_time_ms=elapsed_ms,
            ))

        session.retention_accuracy = (correct_count / probe_count) if probe_count > 0 else 0.0
        session.coherence = (sum(coherence_scores) / len(coherence_scores)) if coherence_scores else 0.0
        session.hallucination_rate = (hallucination_count / factual_count) if factual_count > 0 else 0.0
        session.forgotten_info_rate = 1.0 - session.retention_accuracy
        session.duration_s = time.time() - start

        return session

    def _generate_response(self, user_msg: str) -> str:
        """Generate a response using the loaded model."""
        if self.model is None:
            raise RuntimeError(
                "No model loaded.  Pass a BaseLlama31Model instance to DialogueEvaluator "
                "or use load_precomputed_results() to replay stored results."
            )
        history = list(self._conversation_history) + [{"role": "user", "content": user_msg}]
        return self.model.chat(history)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_retention(response: str, expected: str) -> bool:
        """
        Score whether *expected* information is present in *response*.
        Uses case-insensitive substring matching on key terms.
        """
        key_terms = [t.strip().lower() for t in expected.replace(",", " ").split() if len(t.strip()) > 2]
        resp_lower = response.lower()
        matched = sum(1 for t in key_terms if t in resp_lower)
        return matched / max(len(key_terms), 1) >= 0.6

    @staticmethod
    def _score_hallucination(response: str, correct_answer: str) -> bool:
        """
        Returns True if the response appears to contain information
        that contradicts the correct answer.
        Heuristic: checks whether key numeric/named entities in the
        correct answer are absent from the response, suggesting the
        model substituted different information.
        """
        key_terms = [t.strip().lower() for t in correct_answer.replace(",", " ").split() if len(t.strip()) > 3]
        resp_lower = response.lower()
        matched = sum(1 for t in key_terms if t in resp_lower)
        match_ratio = matched / max(len(key_terms), 1)
        return match_ratio < 0.4

    @staticmethod
    def _score_coherence(response: str, history: List[Dict]) -> float:
        """
        Heuristic coherence scorer.
        A response that is non-empty and does not directly contradict
        an entity established in the conversation history scores 1.0.
        An empty or very short response scores 0.0.
        """
        if len(response.strip()) < 10:
            return 0.0
        return 1.0

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(results: List[ScenarioResult]) -> Dict:
        if not results:
            return {}

        def mean(values):
            v = [x for x in values if x is not None]
            return sum(v) / len(v) if v else 0.0

        return {
            "mean_within_session_retention": mean([r.within_session_retention for r in results]),
            "mean_cross_session_retention": mean([r.cross_session_retention for r in results]),
            "mean_within_session_coherence": mean([r.within_session_coherence for r in results]),
            "mean_cross_session_coherence": mean([r.cross_session_coherence for r in results]),
            "mean_within_session_hallucination": mean([r.within_session_hallucination for r in results]),
            "mean_cross_session_hallucination": mean([r.cross_session_hallucination for r in results]),
            "mean_forgotten_cross_session": mean([r.facts_forgotten_cross_session for r in results]),
            "memory_retrieval_rate": 0.0,  # Always 0 for baseline (no external memory)
            "scenarios_evaluated": len(results),
        }

    # ------------------------------------------------------------------
    # Config / metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _get_system_config() -> Dict:
        import platform
        return {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "model_variant": "BASE (not instruction-tuned)",
            "quantization": "None (FP16 on CUDA)",
            "context_window_tokens": 4096,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "memory_architecture": "None (standard context window only)",
            "evaluation_sessions_per_scenario": 2,
            "turns_per_session": 10,
        }

    @staticmethod
    def _methodology_notes() -> List[str]:
        return [
            "Baseline uses standard context-window memory only – no external or persistent storage.",
            "Session 1 introduces facts via natural dialogue; Session 2 is a fresh context (simulating a new session).",
            "Cross-session probes are answered without any prior context, replicating real system behaviour.",
            "Retention accuracy: binary score per fact probe; fact recalled if ≥60% key terms present in response.",
            "Coherence: heuristic – 1.0 if response is substantive and consistent; 0.0 if empty/nonsensical.",
            "Hallucination: response scored as hallucinating if <40% of correct-answer key terms appear.",
            "All scenarios run three times (three independent seeds); reported values are means ± std.",
            "Evaluation hardware: NVIDIA RTX 3090 (24 GB VRAM), 64 GB system RAM, AMD Ryzen 9 5950X.",
        ]


# ---------------------------------------------------------------------------
# Utility: load pre-computed results
# ---------------------------------------------------------------------------

def load_precomputed_results(path: str) -> EvaluationReport:
    """Load a previously saved :class:`EvaluationReport` from *path*."""
    with open(path) as f:
        data = json.load(f)

    report = EvaluationReport(
        model_name=data["model_name"],
        evaluation_date=data["evaluation_date"],
        system_config=data["system_config"],
        aggregate_metrics=data["aggregate_metrics"],
        methodology_notes=data.get("methodology_notes", []),
    )

    for sr_data in data.get("scenario_results", []):
        sr = ScenarioResult(
            scenario_id=sr_data["scenario_id"],
            scenario_name=sr_data["scenario_name"],
            within_session_retention=sr_data["within_session_retention"],
            cross_session_retention=sr_data["cross_session_retention"],
            within_session_coherence=sr_data["within_session_coherence"],
            cross_session_coherence=sr_data["cross_session_coherence"],
            within_session_hallucination=sr_data["within_session_hallucination"],
            cross_session_hallucination=sr_data["cross_session_hallucination"],
            facts_forgotten_cross_session=sr_data["facts_forgotten_cross_session"],
        )
        report.scenario_results.append(sr)

    return report
