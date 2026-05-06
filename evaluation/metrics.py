"""
Evaluation Metrics module.

Provides quantitative measures for the six evaluation criteria:

  1. Retention accuracy          – % of correctly recalled facts
  2. Dialogue coherence          – consistency score across sessions
  3. Memory retrieval consistency – % of relevant memories retrieved
  4. Forgotten information rate   – failed recall instances per conversation
  5. Error detection rate         – fraction of factual errors identified
  6. Hallucination frequency      – rate of fabricated facts per turn
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TurnResult:
    """Stores the ground-truth and model response for a single dialogue turn."""
    turn_id: int
    session_id: str
    user_message: str
    model_response: str
    expected_facts: List[str] = field(default_factory=list)
    # Populated by the scorer
    recalled_facts: List[str] = field(default_factory=list)
    errors_detected: List[str] = field(default_factory=list)
    hallucinations: List[str] = field(default_factory=list)


class EvaluationMetrics:
    """
    Computes evaluation metrics from a list of TurnResult objects.

    Usage::

        metrics = EvaluationMetrics()
        metrics.add_turn(turn_result)
        report = metrics.compute()
        print(metrics.format_report(report))
    """

    def __init__(self) -> None:
        self._turns: List[TurnResult] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_turn(self, turn: TurnResult) -> None:
        """Register a single evaluated turn."""
        self._turns.append(turn)

    def reset(self) -> None:
        """Clear all collected turn results."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(lst: List) -> float:
        """Return the arithmetic mean of *lst*, or 0.0 for an empty list."""
        return sum(lst) / len(lst) if lst else 0.0

    def compute(self) -> Dict[str, float]:
        """Compute all six evaluation metrics.

        Returns:
            A dict mapping metric name → score (all values in [0, 1] or
            raw counts as noted below).
        """
        if not self._turns:
            return {}

        retention_scores: List[float] = []
        retrieval_scores: List[float] = []
        forgotten_counts: List[int] = []
        error_scores: List[float] = []
        hallucination_counts: List[int] = []

        for turn in self._turns:
            # 1. Retention accuracy ─────────────────────────────────────
            if turn.expected_facts:
                recalled = sum(
                    1
                    for fact in turn.expected_facts
                    if self._fact_present(fact, turn.model_response)
                )
                retention_scores.append(recalled / len(turn.expected_facts))
                forgotten_counts.append(len(turn.expected_facts) - recalled)
            else:
                forgotten_counts.append(0)

            # 3. Memory retrieval consistency ───────────────────────────
            # Use turn.recalled_facts if the scorer populated them,
            # otherwise fall back to the retention check above.
            if turn.recalled_facts:
                if turn.expected_facts:
                    hit = sum(
                        1
                        for rf in turn.recalled_facts
                        if rf in turn.expected_facts
                    )
                    retrieval_scores.append(hit / len(turn.expected_facts))
            elif turn.expected_facts:
                retrieval_scores.append(retention_scores[-1])

            # 5. Error detection rate ───────────────────────────────────
            if turn.errors_detected:
                error_scores.append(1.0)  # at least one error was flagged
            else:
                error_scores.append(0.0)

            # 6. Hallucination frequency ────────────────────────────────
            hallucination_counts.append(len(turn.hallucinations))

        # 2. Dialogue coherence – approximate as average retention score
        coherence = self._mean(retention_scores) if retention_scores else 0.0

        return {
            "retention_accuracy": self._mean(retention_scores),
            "dialogue_coherence": coherence,
            "memory_retrieval_consistency": self._mean(retrieval_scores),
            "forgotten_information_rate": self._mean(forgotten_counts),
            "error_detection_rate": self._mean(error_scores),
            "hallucination_frequency": self._mean(hallucination_counts),
            "total_turns_evaluated": len(self._turns),
        }

    def compute_per_session(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics separately for each session."""
        sessions: Dict[str, List[TurnResult]] = {}
        for turn in self._turns:
            sessions.setdefault(turn.session_id, []).append(turn)

        results = {}
        for sid, turns in sessions.items():
            m = EvaluationMetrics()
            for t in turns:
                m.add_turn(t)
            results[sid] = m.compute()
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(metrics: Dict[str, float]) -> str:
        """Render *metrics* as a formatted text report."""
        lines = [
            "=" * 60,
            "Evaluation Report – Hierarchical Memory System",
            "=" * 60,
        ]
        labels = {
            "retention_accuracy": "1. Retention Accuracy",
            "dialogue_coherence": "2. Dialogue Coherence",
            "memory_retrieval_consistency": "3. Memory Retrieval Consistency",
            "forgotten_information_rate": "4. Forgotten Information Rate (avg/turn)",
            "error_detection_rate": "5. Error Detection Rate",
            "hallucination_frequency": "6. Hallucination Frequency (avg/turn)",
            "total_turns_evaluated": "   Total Turns Evaluated",
        }
        for key, label in labels.items():
            val = metrics.get(key, "N/A")
            if isinstance(val, float):
                if key in ("forgotten_information_rate", "hallucination_frequency",
                           "total_turns_evaluated"):
                    lines.append(f"  {label}: {val:.2f}")
                else:
                    lines.append(f"  {label}: {val * 100:.1f}%")
            else:
                lines.append(f"  {label}: {val}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fact_present(fact: str, response: str) -> bool:
        """Return True if *fact* (case-insensitive) appears in *response*."""
        return fact.lower() in response.lower()
