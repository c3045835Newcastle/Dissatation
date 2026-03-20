"""
Test script to validate the base model setup and hierarchical memory
architecture (without actually loading the LLM).

Tests the code structure and imports for all components described in the
dissertation proposal objectives.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import config
        print("✓ config.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False

    try:
        from llama_base_model import BaseLlama31Model
        print("✓ llama_base_model.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import llama_base_model: {e}")
        return False

    try:
        import inference
        print("✓ inference.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import inference: {e}")
        return False

    # Hierarchical memory modules
    try:
        from memory import WorkingMemory, EpisodicMemory, SemanticMemory, MemoryController
        print("✓ memory package imported successfully")
    except Exception as e:
        print(f"✗ Failed to import memory package: {e}")
        return False

    try:
        from dialogue_pipeline import HierarchicalMemoryPipeline
        print("✓ dialogue_pipeline.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import dialogue_pipeline: {e}")
        return False

    try:
        from evaluation import EvaluationMetrics, EvaluationScenarios, EvaluationRunner
        print("✓ evaluation package imported successfully")
    except Exception as e:
        print(f"✗ Failed to import evaluation package: {e}")
        return False

    return True


def test_config():
    """Test configuration values."""
    print("\nTesting configuration...")

    import config

    # Check that it's the base model
    if "Meta-Llama-3.1-8B" in config.MODEL_NAME:
        print(f"✓ Correct base model configured: {config.MODEL_NAME}")
    else:
        print(f"✗ Unexpected model name: {config.MODEL_NAME}")
        return False

    # Check basic parameters
    if config.MAX_NEW_TOKENS > 0:
        print(f"✓ MAX_NEW_TOKENS is valid: {config.MAX_NEW_TOKENS}")
    else:
        print(f"✗ Invalid MAX_NEW_TOKENS: {config.MAX_NEW_TOKENS}")
        return False

    if 0 <= config.TEMPERATURE <= 2:
        print(f"✓ TEMPERATURE is valid: {config.TEMPERATURE}")
    else:
        print(f"✗ Invalid TEMPERATURE: {config.TEMPERATURE}")
        return False

    # Check memory config (Objectives 2–3)
    if config.WORKING_MEMORY_MAX_TURNS >= 10:
        print(f"✓ WORKING_MEMORY_MAX_TURNS >= 10 (Objective 1): {config.WORKING_MEMORY_MAX_TURNS}")
    else:
        print(f"✗ WORKING_MEMORY_MAX_TURNS must be >= 10 (Objective 1): {config.WORKING_MEMORY_MAX_TURNS}")
        return False

    if config.EPISODIC_TOP_K > 0:
        print(f"✓ EPISODIC_TOP_K is valid: {config.EPISODIC_TOP_K}")
    else:
        print(f"✗ Invalid EPISODIC_TOP_K: {config.EPISODIC_TOP_K}")
        return False

    if config.EVALUATION_NUM_SCENARIOS >= 5:
        print(f"✓ EVALUATION_NUM_SCENARIOS >= 5 (Objective 5): {config.EVALUATION_NUM_SCENARIOS}")
    else:
        print(f"✗ EVALUATION_NUM_SCENARIOS must be >= 5 (Objective 5): {config.EVALUATION_NUM_SCENARIOS}")
        return False

    return True


def test_dependencies():
    """Test that required packages are importable."""
    print("\nTesting dependencies...")

    required = ['torch', 'transformers']
    recommended = [('accelerate', 'accelerate')]
    memory_deps = [('faiss', 'faiss-cpu'), ('sentence_transformers', 'sentence-transformers'), ('numpy', 'numpy')]
    all_ok = True

    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed (run: pip install -r requirements.txt)")
            all_ok = False

    for import_name, pip_name in recommended:
        try:
            __import__(import_name)
            print(f"✓ {pip_name} is installed")
        except ImportError:
            print(f"⚠ {pip_name} not installed (recommended for GPU; run: pip install {pip_name})")

    for import_name, pip_name in memory_deps:
        try:
            __import__(import_name)
            print(f"✓ {pip_name} is installed")
        except ImportError:
            print(f"⚠ {pip_name} not installed – memory will use fallback mode "
                  f"(run: pip install {pip_name})")
            # These are optional – don't fail the test

    return all_ok


def test_working_memory():
    """Test Working Memory (Objective 2a)."""
    print("\nTesting Working Memory (Objective 2a)...")

    from memory import WorkingMemory

    wm = WorkingMemory(max_turns=10)

    # Add 12 turns – the window should only keep the latest 10
    for i in range(12):
        role = "user" if i % 2 == 0 else "assistant"
        wm.add_turn(role, f"Turn {i}")

    turns = wm.get_context()
    if len(turns) == 10:
        print("✓ Sliding window evicts oldest turns correctly")
    else:
        print(f"✗ Expected 10 turns, got {len(turns)}")
        return False

    last2 = wm.get_last_n(2)
    if len(last2) == 2 and last2[-1]["content"] == "Turn 11":
        print("✓ get_last_n returns correct turns")
    else:
        print(f"✗ get_last_n failed: {last2}")
        return False

    return True


def test_semantic_memory():
    """Test Semantic Memory (Objective 2c)."""
    print("\nTesting Semantic Memory (Objective 2c)...")

    from memory import SemanticMemory

    sm = SemanticMemory()

    sm.store_fact("name", "Alice", "personal")
    sm.store_fact("occupation", "engineer", "personal")
    sm.store_fact("preference_hiking", "hiking", "preferences")

    if sm.get_fact("name", "personal") == "Alice":
        print("✓ store_fact and get_fact work correctly")
    else:
        print("✗ store_fact / get_fact failed")
        return False

    results = sm.search("Alice")
    if results and results[0]["value"] == "Alice":
        print("✓ search returns correct results")
    else:
        print(f"✗ search failed: {results}")
        return False

    ctx = sm.format_for_context()
    if "Alice" in ctx and "Personal" in ctx:
        print("✓ format_for_context includes stored facts")
    else:
        print(f"✗ format_for_context output unexpected: {ctx}")
        return False

    if len(sm) == 3:
        print(f"✓ __len__ returns correct count: {len(sm)}")
    else:
        print(f"✗ Expected 3 facts, got {len(sm)}")
        return False

    return True


def test_episodic_memory():
    """Test Episodic Memory (Objective 2b)."""
    print("\nTesting Episodic Memory (Objective 2b)...")

    from memory import EpisodicMemory

    em = EpisodicMemory(top_k=3)

    em.add_episode(
        text="User: What is machine learning?  Assistant: ML is …",
        summary="User asked about machine learning",
        session_id="s1",
        turn=1,
    )
    em.add_episode(
        text="User: Tell me about Python.  Assistant: Python is …",
        summary="User asked about Python programming",
        session_id="s1",
        turn=2,
    )

    if len(em) == 2:
        print("✓ Episodes stored correctly")
    else:
        print(f"✗ Expected 2 episodes, got {len(em)}")
        return False

    results = em.retrieve("machine learning")
    if results:
        print(f"✓ retrieve returns relevant episodes (top result: '{results[0]['summary'][:40]}…')")
    else:
        print("✗ retrieve returned no results")
        return False

    return True


def test_memory_controller():
    """Test Memory Controller (Objective 3)."""
    print("\nTesting Memory Controller (Objective 3)...")

    from memory import WorkingMemory, EpisodicMemory, SemanticMemory, MemoryController

    wm = WorkingMemory(max_turns=20)
    em = EpisodicMemory(top_k=3)
    sm = SemanticMemory()

    mc = MemoryController(wm, em, sm, consolidation_interval=3)
    mc.set_session("test_session")

    mc.process_user_turn("My name is Bob and I am a data scientist.")
    mc.process_assistant_turn("Nice to meet you, Bob!")

    if sm.get_fact("name", "personal") == "Bob":
        print("✓ Fact extraction: name 'Bob' stored in semantic memory")
    else:
        print(f"✗ Name not extracted. Semantic memory: {sm.get_all()}")
        return False

    # Trigger consolidation
    mc.process_user_turn("I enjoy reading technical books.")
    mc.process_assistant_turn("Great!")
    mc.process_user_turn("What can you tell me about NLP?")

    ctx = mc.build_context("NLP")
    if ctx:
        print(f"✓ build_context returns enriched context ({len(ctx)} chars)")
    else:
        print("⚠ build_context returned empty string (episodic consolidation may not have triggered)")

    return True


def test_dialogue_pipeline():
    """Test Dialogue Pipeline without LLM (Objective 4)."""
    print("\nTesting Dialogue Pipeline (Objective 4)...")

    from dialogue_pipeline import HierarchicalMemoryPipeline

    # load_model=False → model stub is used
    pipeline = HierarchicalMemoryPipeline(
        session_id="test",
        memory_base_path="/tmp/test_memory",
        load_model=False,
    )

    response = pipeline.chat("My name is Carol and I love chess.")
    if response:
        print("✓ pipeline.chat returns a response")
    else:
        print("✗ pipeline.chat returned empty response")
        return False

    summary = pipeline.get_memory_summary()
    if summary["working_memory_turns"] >= 1:
        print(f"✓ Working memory updated: {summary['working_memory_turns']} turn(s)")
    else:
        print(f"✗ Working memory not updated: {summary}")
        return False

    if summary["semantic_facts"] >= 1:
        print(f"✓ Semantic memory updated: {summary['semantic_facts']} fact(s)")
    else:
        print("⚠ No facts extracted yet (pattern matching may not have matched)")

    return True


def test_evaluation_scenarios():
    """Test Evaluation Scenarios (Objective 5)."""
    print("\nTesting Evaluation Scenarios (Objective 5)...")

    from evaluation import EvaluationScenarios

    scenarios = EvaluationScenarios.get_all()

    if len(scenarios) >= 5:
        print(f"✓ {len(scenarios)} scenarios available (>= 5 required by Objective 5)")
    else:
        print(f"✗ Only {len(scenarios)} scenarios; need at least 5")
        return False

    for s in scenarios:
        if len(s.sessions) >= 2:
            print(f"  ✓ '{s.name}' has {len(s.sessions)} sessions")
        else:
            print(f"  ✗ '{s.name}' has only {len(s.sessions)} session(s)")
            return False

    return True


def test_evaluation_metrics():
    """Test Evaluation Metrics (Objective 6)."""
    print("\nTesting Evaluation Metrics (Objective 6)...")

    from evaluation.metrics import EvaluationMetrics, TurnResult

    metrics = EvaluationMetrics()

    metrics.add_turn(TurnResult(
        turn_id=0, session_id="s1",
        user_message="What is my name?",
        model_response="Your name is Alice.",
        expected_facts=["Alice"],
    ))
    metrics.add_turn(TurnResult(
        turn_id=1, session_id="s1",
        user_message="What do I do?",
        model_response="I don't know what you do.",
        expected_facts=["engineer"],
    ))

    report = metrics.compute()
    expected_keys = [
        "retention_accuracy", "dialogue_coherence",
        "memory_retrieval_consistency", "forgotten_information_rate",
        "error_detection_rate", "hallucination_frequency",
        "total_turns_evaluated",
    ]
    for key in expected_keys:
        if key in report:
            print(f"  ✓ Metric '{key}' present: {report[key]:.3f}" if isinstance(report[key], float) else f"  ✓ '{key}': {report[key]}")
        else:
            print(f"  ✗ Missing metric: {key}")
            return False

    # Retention: 1/1 facts recalled in turn 0, 0/1 in turn 1 → 50%
    if abs(report["retention_accuracy"] - 0.5) < 0.01:
        print("✓ Retention accuracy computed correctly (50.0%)")
    else:
        print(f"✗ Retention accuracy incorrect: {report['retention_accuracy']:.3f}")
        return False

    text = EvaluationMetrics.format_report(report)
    if "Evaluation Report" in text:
        print("✓ format_report generates readable report")
    else:
        print("✗ format_report output unexpected")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Hierarchical Memory Dialogue System – Setup Validation")
    print("=" * 60)
    print()

    results = []

    # Original tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Dependencies", test_dependencies()))

    # New hierarchical memory tests
    results.append(("Working Memory (Obj. 2a)", test_working_memory()))
    results.append(("Semantic Memory (Obj. 2c)", test_semantic_memory()))
    results.append(("Episodic Memory (Obj. 2b)", test_episodic_memory()))
    results.append(("Memory Controller (Obj. 3)", test_memory_controller()))
    results.append(("Dialogue Pipeline (Obj. 4)", test_dialogue_pipeline()))
    results.append(("Evaluation Scenarios (Obj. 5)", test_evaluation_scenarios()))
    results.append(("Evaluation Metrics (Obj. 6)", test_evaluation_metrics()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! Setup looks good.")
        print("\nNote: This does not test actual LLM generation.")
        print("To test with the full model, run: python inference.py")
        print("To run the hierarchical memory pipeline: python dialogue_pipeline.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

