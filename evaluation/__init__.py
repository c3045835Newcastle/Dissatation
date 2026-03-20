"""
Evaluation framework for the hierarchical memory dialogue system.

Implements the metrics and controlled experimental scenarios described in
the dissertation proposal (Objectives 5 & 6).

Objective 5: Design and conduct controlled experimental dialogue scenarios
             consisting of at least 5 multi-session conversations, where
             specific information is introduced and later tested to evaluate
             memory retention and retrieval performance.

Objective 6: Evaluate performance using measurable criteria including:
  - Retention accuracy         (% of correctly recalled information)
  - Dialogue coherence         (consistency of responses across sessions)
  - Consistency across interactions (% of relevant memories retrieved)
  - Frequency of forgotten information (failed recall instances per conversation)
  - Error detection
  - Frequency of hallucination
"""

from .metrics import EvaluationMetrics
from .evaluation_scenarios import EvaluationScenarios, EvaluationRunner

__all__ = [
    "EvaluationMetrics",
    "EvaluationScenarios",
    "EvaluationRunner",
]
