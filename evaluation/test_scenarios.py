"""
Test scenarios for evaluating the baseline dialogue system.

Five multi-session conversation scenarios are defined here, each introducing
a set of known facts that are later probed for recall.  These scenarios are
designed to surface the key weaknesses of a context-window-only system:

  1. Cross-session amnesia   – facts forgotten after session restart
  2. Context overflow        – facts evicted when window fills up
  3. Multi-session coherence – contradictions across sessions
  4. Long-session drift      – coherence degrading over many turns
  5. Rapid fact introduction – many facts introduced quickly, some evicted

Each scenario returns a list of (action, payload) pairs consumed by
run_evaluation.py.
"""

from __future__ import annotations
from typing import List, Tuple, Any

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIO_1_FACTS = [
    {"key": "name",             "value": "Robin"},
    {"key": "age",              "value": "22"},
    {"key": "university",       "value": "Newcastle University"},
    {"key": "favourite colour", "value": "orange"},
    {"key": "pet",              "value": "a cat"},
]

SCENARIO_2_FACTS = [
    {"key": "name",             "value": "Jordan"},
    {"key": "job",              "value": "software engineer"},
    {"key": "city",             "value": "Newcastle"},
    {"key": "hobby",            "value": "playing guitar"},
    {"key": "favourite food",   "value": "pizza"},
    {"key": "siblings",         "value": "two brothers"},
    {"key": "car",              "value": "a blue Honda Civic"},
]

SCENARIO_3_FACTS = [
    {"key": "name",             "value": "Alex"},
    {"key": "project topic",    "value": "memory architectures for LLMs"},
    {"key": "supervisor",       "value": "Dr Smith"},
    {"key": "submission date",  "value": "May 2025"},
]

SCENARIO_4_FACTS = [
    {"key": "name",             "value": "Sam"},
    {"key": "age",              "value": "30"},
    {"key": "city",             "value": "London"},
    {"key": "job",              "value": "data scientist"},
    {"key": "favourite book",   "value": "Dune"},
    {"key": "language",         "value": "Python"},
]

SCENARIO_5_FACTS = [
    {"key": "name",                "value": "Morgan"},
    {"key": "email",               "value": "morgan@example.com"},
    {"key": "phone number",        "value": "07700900123"},
    {"key": "emergency contact",   "value": "Chris"},
    {"key": "medical condition",   "value": "none"},
    {"key": "nationality",         "value": "British"},
    {"key": "preferred language",  "value": "English"},
    {"key": "dietary requirement", "value": "vegetarian"},
]


def scenario_1_cross_session_amnesia() -> dict:
    """
    Scenario 1: Cross-Session Amnesia
    ──────────────────────────────────
    Introduce facts in Session 1.  Start a new session (context cleared).
    Immediately probe for each fact.  Expect near-zero recall.
    """
    return {
        "id":          1,
        "name":        "Cross-Session Amnesia",
        "description": ("Facts introduced in Session 1 are probed at the start "
                        "of Session 2.  A context-window-only system has no way "
                        "to persist information across session boundaries."),
        "sessions":    2,
        "facts":       SCENARIO_1_FACTS,
        "phases": [
            {"session": 1, "action": "introduce_all"},
            {"session": 1, "action": "filler", "turns": 5},
            {"session": 2, "action": "probe_all"},
        ],
    }


def scenario_2_context_overflow() -> dict:
    """
    Scenario 2: Context Window Overflow
    ─────────────────────────────────────
    Introduce several facts, then flood the context with long filler turns.
    Probe facts that were introduced earliest – they should have been evicted.
    """
    return {
        "id":          2,
        "name":        "Context Window Overflow",
        "description": ("Facts are introduced at the beginning of a long single "
                        "session.  Many filler turns push the early facts beyond "
                        "the context window limit.  Early facts should be "
                        "forgotten; recent facts may survive."),
        "sessions":    1,
        "facts":       SCENARIO_2_FACTS,
        "phases": [
            {"session": 1, "action": "introduce_all"},
            {"session": 1, "action": "filler", "turns": 30},
            {"session": 1, "action": "probe_all"},
        ],
    }


def scenario_3_multi_session_coherence() -> dict:
    """
    Scenario 3: Multi-Session Coherence
    ─────────────────────────────────────
    Facts are introduced and probed across three sessions.  The system should
    maintain coherence within a session but fail across sessions.
    """
    return {
        "id":          3,
        "name":        "Multi-Session Coherence",
        "description": ("The same facts are probed at the end of the session "
                        "they were introduced in (expected: good recall) and "
                        "again at the start of two subsequent sessions "
                        "(expected: poor recall)."),
        "sessions":    3,
        "facts":       SCENARIO_3_FACTS,
        "phases": [
            {"session": 1, "action": "introduce_all"},
            {"session": 1, "action": "probe_all",   "label": "within-session"},
            {"session": 2, "action": "probe_all",   "label": "cross-session-1"},
            {"session": 3, "action": "probe_all",   "label": "cross-session-2"},
        ],
    }


def scenario_4_long_session_drift() -> dict:
    """
    Scenario 4: Long-Session Coherence Drift
    ──────────────────────────────────────────
    Facts are introduced, then coherence is measured at increasing turn
    intervals (5, 10, 20, 30 turns) to show how recall degrades over time.

    A reduced context limit of 1024 tokens is used here to make the eviction
    curve visible within a manageable number of filler turns, reflecting the
    practical constraints of running a quantised local model.
    """
    return {
        "id":            4,
        "name":          "Long-Session Coherence Drift",
        "description": ("Facts are introduced at turn 0.  Recall accuracy is "
                        "sampled after 5, 10, 20, and 30 filler turns, showing "
                        "how performance degrades as context fills."),
        "sessions":      1,
        "facts":         SCENARIO_4_FACTS,
        "context_limit": 1024,       # constrained window to demonstrate decay
        "phases": [
            {"session": 1, "action": "introduce_all"},
            {"session": 1, "action": "probe_all",   "label": "after-0-turns"},
            {"session": 1, "action": "filler",      "turns": 5},
            {"session": 1, "action": "probe_all",   "label": "after-5-turns"},
            {"session": 1, "action": "filler",      "turns": 5},
            {"session": 1, "action": "probe_all",   "label": "after-10-turns"},
            {"session": 1, "action": "filler",      "turns": 10},
            {"session": 1, "action": "probe_all",   "label": "after-20-turns"},
            {"session": 1, "action": "filler",      "turns": 10},
            {"session": 1, "action": "probe_all",   "label": "after-30-turns"},
        ],
    }


def scenario_5_rapid_fact_introduction() -> dict:
    """
    Scenario 5: Rapid Fact Introduction Under Pressure
    ────────────────────────────────────────────────────
    Many facts are introduced rapidly in one session.  Half are probed
    immediately (should be recalled), then the session is reset and all
    are probed again (should all be forgotten).
    """
    return {
        "id":          5,
        "name":        "Rapid Fact Introduction Under Pressure",
        "description": ("Eight facts are introduced in quick succession.  "
                        "Immediate recall is tested (expected: high accuracy).  "
                        "After a session restart all facts are probed again "
                        "(expected: 0% recall)."),
        "sessions":    2,
        "facts":       SCENARIO_5_FACTS,
        "phases": [
            {"session": 1, "action": "introduce_all"},
            {"session": 1, "action": "probe_all",  "label": "immediate"},
            {"session": 2, "action": "probe_all",  "label": "post-session-reset"},
        ],
    }


ALL_SCENARIOS = [
    scenario_1_cross_session_amnesia,
    scenario_2_context_overflow,
    scenario_3_multi_session_coherence,
    scenario_4_long_session_drift,
    scenario_5_rapid_fact_introduction,
]
