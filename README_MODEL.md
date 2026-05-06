# Base Llama 3.1 8B Local Deployment

This repository contains the implementation for locally deploying the **BASE** Llama 3.1 8B model (not fine-tuned) for dissertation research.

## Overview

This implementation provides a simple, clean interface to run Meta's Llama 3.1 8B base model locally on your machine. The base model is the pre-trained version without any fine-tuning or modifications.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU execution is supported)
- At least 16GB RAM (32GB recommended)
- ~16GB disk space for model files

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd Dissatation
```

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Hugging Face Access** (IMPORTANT):
   - You need to request access to Llama 3.1 models from Meta through Hugging Face
   - Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
   - Click "Request Access" and wait for approval (usually quick)
   - Create a Hugging Face account if you don't have one
   - Generate an access token: https://huggingface.co/settings/tokens
   - Login with your token:
   ```bash
   huggingface-cli login
   ```

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── config.py                 # Model configuration parameters
├── llama_base_model.py      # Main model loader and wrapper
├── inference.py             # Interactive inference script
└── README_MODEL.md          # This file
```

## Usage

### Interactive Mode

Run the model in interactive mode to chat with it:

```bash
python inference.py
```

This will load the base model and allow you to type prompts interactively.

### Single Prompt Mode

Pass a prompt directly as a command-line argument:

```bash
python inference.py "What is artificial intelligence?"
```

### Using in Your Code

```python
from llama_base_model import BaseLlama31Model

# Initialize the base model
model = BaseLlama31Model()

# Generate text
response = model.generate("The future of AI is")
print(response)

# Chat interface
messages = [
    {"role": "user", "content": "Explain quantum computing"}
]
response = model.chat(messages)
print(response)
```

## Configuration

Edit `config.py` to customize model behavior:

- **MODEL_NAME**: The Hugging Face model identifier (set to base model by default)
- **MAX_NEW_TOKENS**: Maximum length of generated text
- **TEMPERATURE**: Controls randomness (0.0 = deterministic, 1.0 = creative)
- **TOP_P**: Nucleus sampling parameter
- **TOP_K**: Top-k sampling parameter
- **LOAD_IN_8BIT**: Enable 8-bit quantization to reduce memory usage
- **LOAD_IN_4BIT**: Enable 4-bit quantization (requires bitsandbytes)

## Memory Optimization

If you encounter out-of-memory errors:

1. **Enable 8-bit quantization**:
   ```python
   # In config.py
   LOAD_IN_8BIT = True
   ```

2. **Enable 4-bit quantization** (requires bitsandbytes):
   ```bash
   pip install bitsandbytes
   ```
   ```python
   # In config.py
   LOAD_IN_4BIT = True
   ```

## Model Information

- **Model**: Meta-Llama-3.1-8B (BASE version)
- **Parameters**: 8 billion
- **Architecture**: Transformer-based decoder-only model
- **Training**: Pre-trained on large-scale text data (BASE, not instruction-tuned)
- **License**: Llama 3.1 Community License

## Important Notes

1. **This is the BASE model**: It is pre-trained but NOT instruction-tuned or fine-tuned. For better chat performance, consider using Meta-Llama-3.1-8B-Instruct (not included here as per requirements).

2. **First run downloads the model**: The first time you run the model, it will download ~16GB of model files. Subsequent runs will use the cached version.

3. **GPU recommended**: While the model can run on CPU, it will be significantly slower. A GPU with at least 16GB VRAM is recommended.

4. **Token limits**: The model has a context window of 128K tokens, but generating very long sequences may be slow.

## Troubleshooting

### "Access denied" error
- Make sure you've requested and received access to Llama 3.1 on Hugging Face
- Ensure you've logged in with `huggingface-cli login`

### Out of memory errors
- Enable quantization (8-bit or 4-bit) in config.py
- Reduce MAX_NEW_TOKENS
- Close other applications
- Use a machine with more RAM/VRAM

### Slow generation
- This is expected on CPU; use a GPU if possible
- Reduce MAX_NEW_TOKENS for faster responses
- Enable flash attention if your GPU supports it

## References

- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Llama 3.1 Paper](https://ai.meta.com/research/publications/llama-3-1/)

## License

This code is provided for educational/research purposes. The Llama 3.1 model itself is subject to Meta's Llama 3.1 Community License.
