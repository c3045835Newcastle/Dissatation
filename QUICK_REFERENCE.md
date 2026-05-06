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

- `llama_base_model.py` - Main model class
- `inference.py` - Interactive inference
- `examples.py` - Usage examples
- `config.py` - Configuration
- `test_setup.py` - Setup validation
- `setup.sh` - Automated setup

## Important Notes

✓ This is the BASE model (not instruction-tuned)  
✓ Requires Hugging Face access to Llama 3.1  
✓ GPU highly recommended  
✓ First run downloads ~16GB  

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Access denied" | Request access on HF + login |
| Out of memory | Enable quantization |
| Slow generation | Use GPU / reduce max_tokens |
| Import error | Install requirements.txt |

## See Also

- Full documentation: `README_MODEL.md`
- Examples: `python examples.py`
- Test setup: `python test_setup.py`
