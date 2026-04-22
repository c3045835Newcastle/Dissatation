# Quick Reference - Base Llama 3.1 8B

## Installation
```bash
pip install -r requirements.txt
huggingface-cli login  # Enter your HF token
```

## Basic Usage

### Interactive Mode
```bash
python inference.py
```

### Single Prompt
```bash
python inference.py "Your prompt here"
```

### In Python Code
```python
from llama_base_model import BaseLlama31Model

# Initialize
model = BaseLlama31Model()

# Generate
response = model.generate("Your prompt")
print(response)
```

## Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| MODEL_NAME | meta-llama/Meta-Llama-3.1-8B | Base model ID |
| MAX_NEW_TOKENS | 512 | Max generation length |
| TEMPERATURE | 0.7 | Randomness (0-2) |
| TOP_P | 0.9 | Nucleus sampling |
| TOP_K | 50 | Top-k sampling |
| LOAD_IN_8BIT | False | Enable 8-bit quantization |
| LOAD_IN_4BIT | False | Enable 4-bit quantization |

## Memory Requirements

| Configuration | VRAM/RAM | Quality |
|---------------|----------|---------|
| Full (FP16) | ~16GB | Best |
| 8-bit | ~8GB | Good |
| 4-bit | ~4GB | Fair |

## Common Tasks

### Reduce Memory Usage
```python
# In config.py
LOAD_IN_8BIT = True  # or LOAD_IN_4BIT = True
```

### Adjust Creativity
```python
# More creative
model.generate(prompt, temperature=1.2)

# More deterministic
model.generate(prompt, temperature=0.1)
```

### Longer Responses
```python
model.generate(prompt, max_new_tokens=1024)
```

## Files

### Pre (Baseline) System
- `llama_base_model.py` - Base model class (no external memory)
- `inference.py` - Interactive inference
- `examples.py` - Usage examples
- `config.py` - Configuration
- `test_setup.py` - Setup validation
- `setup.sh` - Automated setup
- `benchmark.py` - Hardware performance benchmark

### Post (Hierarchical Memory) System
- `memory_dialogue_system.py` - Post system with hierarchical memory
- `memory/working_memory.py` - Sliding-window recent context
- `memory/episodic_memory.py` - FAISS vector store for past interactions
- `memory/semantic_memory.py` - Persistent user facts
- `memory/memory_controller.py` - Orchestrates all three memory layers

### Evaluation & Results
- `evaluate_pre_post.py` - Pre vs post evaluation pipeline
- `run_full_evaluation.sh` - One-command evaluation runner
- `EVALUATION_RESULTS.md` - Dissertation-ready comparison tables
- `results/` - Output directory for all result JSON files

## Post System Usage

### Basic Chat (Python)
```python
from memory_dialogue_system import HierarchicalMemoryDialogueSystem

system = HierarchicalMemoryDialogueSystem(precision="int4")  # int4 = 5 GB VRAM
result = system.chat("My name is Alice and I am a PhD student.")
print(result["response"])

# Start a new session (episodic + semantic memory persist)
system.new_session()
result = system.chat("What is my name?")  # Recalls "Alice" from memory
print(result["response"])
print(result["episodic_hits"])   # Number of past turns retrieved
print(result["memory_stats"])    # Current sizes of all memory layers
```

### Run Full Evaluation (one command)
```bash
# Full evaluation, both systems, FP16 (~4-6 hours on RX 9060 XT)
bash run_full_evaluation.sh

# Quick smoke-test, INT4, ~30 min
bash run_full_evaluation.sh --quick --precision int4

# Post system only
bash run_full_evaluation.sh --system post --precision int4

# Dry-run (tests pipeline without loading the model)
bash run_full_evaluation.sh --dry-run
```

### Memory Requirements (Post System)

| Configuration | VRAM | System RAM | Quality |
|---------------|------|-----------|---------|
| FP16 | ~15.9 GB | ~5.1 GB | Best |
| INT8 | ~8.4 GB | ~4.8 GB | Good |
| INT4 | ~4.8 GB | ~4.6 GB | Fair |

> System RAM includes ~0.9 GB for the sentence-transformers encoder.
> All three configurations fit within 16 GB VRAM and 16 GB RAM.

## Important Notes

✓ Pre system: BASE model (no external memory, context-window-limited)
✓ Post system: Hierarchical memory (Working + Episodic FAISS + Semantic)
✓ Both systems require Hugging Face access to Llama 3.1
✓ GPU highly recommended
✓ First run downloads ~16GB (model) + ~90MB (sentence-transformers encoder)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Access denied" | Request access on HF + login |
| Out of memory (VRAM) | Use `precision="int4"` |
| Out of memory (RAM) | Enable quantization or close other apps |
| Slow generation | GPU recommended; reduce max_new_tokens |
| Import error (main model) | `pip install -r requirements.txt` |
| Import error (memory system) | `pip install -r requirements-memory.txt` |

## See Also

- Full pre-system documentation: `README_MODEL.md`
- Pre vs post evaluation: `EVALUATION_RESULTS.md`
- Examples: `python examples.py`
- Test setup: `python test_setup.py`
