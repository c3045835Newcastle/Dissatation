"""
Evaluation Scenarios module.

Provides five controlled multi-session conversation scenarios for evaluating
memory retention across dialogue sessions.

Each scenario:
  • Spans at least two simulated sessions.
  • Introduces specific facts in session 1 that are tested in session 2+.
  • Carries ground-truth `expected_facts` for automatic scoring.

Scenarios
─────────
  1. Personal Information Retention  – name, age, occupation
  2. Hobby and Preference Tracking   – hobbies, food preferences
  3. Goal and Study Tracking         – academic goals, revision topics
  4. Health and Allergy Information  – health conditions, dietary restrictions
  5. Cross-Session Knowledge Probe   – technical facts introduced then recalled
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from dialogue_pipeline import HierarchicalMemoryPipeline

from evaluation.metrics import EvaluationMetrics, TurnResult


@dataclass
class DialogueTurn:
    """A single turn in a scenario script."""
    role: str                          # 'user' only (assistant responses are generated)
    message: str
    expected_facts: List[str] = field(default_factory=list)  # facts that MUST appear in response
    session_id: Optional[str] = None   # if None, inherits from scenario


@dataclass
class Scenario:
    """A multi-session dialogue scenario."""
    name: str
    description: str
    sessions: List[List[DialogueTurn]]  # outer list = sessions, inner = turns


class EvaluationScenarios:
    """
    Collection of five controlled evaluation scenarios.

    Each scenario is authored with specific facts introduced early so
    that later queries can test whether the memory system retained them.
    """

    @staticmethod
    def scenario_1_personal_information() -> Scenario:
        """
        Scenario 1: Personal Information Retention.
        Tests whether the system remembers the user's name, age, and
        occupation across sessions.
        """
        return Scenario(
            name="Personal Information Retention",
            description=(
                "The user introduces their name, age, and occupation in session 1. "
                "Session 2 asks questions that require that information."
            ),
            sessions=[
                # Session 1 – introduce facts
                [
                    DialogueTurn(
                        role="user",
                        message="Hi! My name is Alice and I am 24 years old.",
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message="I am a software engineer working on machine learning projects.",
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message="Can you recommend a good Python library for data visualisation?",
                        expected_facts=[],
                    ),
                ],
                # Session 2 – test recall
                [
                    DialogueTurn(
                        role="user",
                        message="What is my name?",
                        expected_facts=["Alice"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="What do I do for a living?",
                        expected_facts=["software engineer", "machine learning"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="How old am I?",
                        expected_facts=["24"],
                    ),
                ],
            ],
        )

    @staticmethod
    def scenario_2_hobbies_and_preferences() -> Scenario:
        """
        Scenario 2: Hobby and Preference Tracking.
        Tests whether the system tracks hobbies and food preferences
        across sessions.
        """
        return Scenario(
            name="Hobby and Preference Tracking",
            description=(
                "The user mentions hobbies and food preferences in session 1. "
                "Session 2 asks for personalised recommendations."
            ),
            sessions=[
                [
                    DialogueTurn(
                        role="user",
                        message="I enjoy hiking and photography in my free time.",
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message="I love Italian food and I dislike spicy food.",
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message="Can you suggest a weekend activity for me?",
                        expected_facts=[],
                    ),
                ],
                [
                    DialogueTurn(
                        role="user",
                        message="What hobbies do you know I have?",
                        expected_facts=["hiking", "photography"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="What kind of food do I prefer?",
                        expected_facts=["Italian"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="Suggest a restaurant for me based on what you know.",
                        expected_facts=["Italian"],
                    ),
                ],
            ],
        )

    @staticmethod
    def scenario_3_academic_goals() -> Scenario:
        """
        Scenario 3: Goal and Study Tracking.
        Tests retention of academic goals and revision topics.
        """
        return Scenario(
            name="Goal and Study Tracking",
            description=(
                "The user describes their study goals and revision subjects. "
                "Later sessions test whether these are remembered."
            ),
            sessions=[
                [
                    DialogueTurn(
                        role="user",
                        message=(
                            "I am studying computer science at Newcastle University "
                            "and I want to pass my dissertation module."
                        ),
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message=(
                            "I am revising machine learning, natural language processing, "
                            "and neural networks this week."
                        ),
                        expected_facts=[],
                    ),
                ],
                [
                    DialogueTurn(
                        role="user",
                        message="What am I studying?",
                        expected_facts=["computer science", "Newcastle"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="What topics am I revising?",
                        expected_facts=["machine learning", "natural language processing"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="What is my main academic goal?",
                        expected_facts=["dissertation"],
                    ),
                ],
            ],
        )

    @staticmethod
    def scenario_4_health_information() -> Scenario:
        """
        Scenario 4: Health and Dietary Information Retention.
        Tests whether health-sensitive information is remembered.
        """
        return Scenario(
            name="Health and Dietary Information Retention",
            description=(
                "The user shares health conditions and dietary restrictions. "
                "Later sessions verify this safety-critical information is retained."
            ),
            sessions=[
                [
                    DialogueTurn(
                        role="user",
                        message="I am vegetarian and I do not eat any meat or fish.",
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message="I have a nut allergy so please avoid recommending anything with nuts.",
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message="Can you suggest a healthy lunch option?",
                        expected_facts=[],
                    ),
                ],
                [
                    DialogueTurn(
                        role="user",
                        message="What do you know about my diet?",
                        expected_facts=["vegetarian"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="Do I have any food allergies?",
                        expected_facts=["nut"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="Suggest a recipe for me.",
                        # The response should not include meat/fish/nuts
                        expected_facts=["vegetarian"],
                    ),
                ],
            ],
        )

    @staticmethod
    def scenario_5_technical_knowledge_probe() -> Scenario:
        """
        Scenario 5: Cross-Session Technical Knowledge Probe.
        Introduces a specific technical fact and tests recall.
        """
        return Scenario(
            name="Cross-Session Technical Knowledge Probe",
            description=(
                "A specific technical concept (FAISS indexing) is explained "
                "in session 1. Sessions 2 and 3 ask recall questions."
            ),
            sessions=[
                [
                    DialogueTurn(
                        role="user",
                        message=(
                            "I am building a system that uses FAISS for vector similarity "
                            "search to implement episodic memory retrieval."
                        ),
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message=(
                            "The embedding model I am using is all-MiniLM-L6-v2 which "
                            "produces 384-dimensional vectors."
                        ),
                        expected_facts=[],
                    ),
                    DialogueTurn(
                        role="user",
                        message="Can you explain how FAISS nearest-neighbour search works?",
                        expected_facts=[],
                    ),
                ],
                [
                    DialogueTurn(
                        role="user",
                        message="What vector search library am I using in my project?",
                        expected_facts=["FAISS"],
                    ),
                    DialogueTurn(
                        role="user",
                        message="What embedding model am I using and what is its output dimension?",
                        expected_facts=["all-MiniLM-L6-v2", "384"],
                    ),
                ],
                [
                    DialogueTurn(
                        role="user",
                        message="Summarise what you know about my project's memory retrieval approach.",
                        expected_facts=["FAISS", "384"],
                    ),
                ],
            ],
        )

    @classmethod
    def get_all(cls) -> List[Scenario]:
        """Return all five evaluation scenarios."""
        return [
            cls.scenario_1_personal_information(),
            cls.scenario_2_hobbies_and_preferences(),
            cls.scenario_3_academic_goals(),
            cls.scenario_4_health_information(),
            cls.scenario_5_technical_knowledge_probe(),
        ]


class EvaluationRunner:
    """
    Runs a single scenario against a HierarchicalMemoryPipeline and
    returns an EvaluationMetrics instance populated with the results.

    When the pipeline has no model loaded the runner records the memory
    state rather than LLM responses, allowing offline verification that
    the memory architecture (not the LLM) is functioning correctly.

    Args:
        pipeline  : The dialogue pipeline to evaluate.
        scenario  : The scenario to run.
        verbose   : Print per-turn output to stdout.
    """

    def __init__(
        self,
        pipeline: "HierarchicalMemoryPipeline",
        scenario: Scenario,
        verbose: bool = True,
    ):
        self.pipeline = pipeline
        self.scenario = scenario
        self.verbose = verbose

    def run(self) -> EvaluationMetrics:
        """Execute the scenario and return populated EvaluationMetrics."""
        metrics = EvaluationMetrics()
        global_turn = 0

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Scenario: {self.scenario.name}")
            print(f"{self.scenario.description}")
            print(f"{'=' * 60}")

        for sess_idx, session_turns in enumerate(self.scenario.sessions):
            session_id = f"{self.pipeline.session_id}_s{sess_idx + 1}"
            self.pipeline.controller.set_session(session_id)

            if self.verbose:
                print(f"\n--- Session {sess_idx + 1} ---")

            for turn in session_turns:
                if self.verbose:
                    print(f"\n  User: {turn.message}")

                response = self.pipeline.chat(turn.message)

                if self.verbose:
                    print(f"  Assistant: {response}")
                    if turn.expected_facts:
                        for fact in turn.expected_facts:
                            found = fact.lower() in response.lower()
                            symbol = "✓" if found else "✗"
                            print(f"    {symbol} Expected fact: '{fact}'")

                result = TurnResult(
                    turn_id=global_turn,
                    session_id=session_id,
                    user_message=turn.message,
                    model_response=response,
                    expected_facts=turn.expected_facts,
                )
                metrics.add_turn(result)
                global_turn += 1

        if self.verbose:
            report = metrics.compute()
            print("\n" + EvaluationMetrics.format_report(report))

        return metrics
