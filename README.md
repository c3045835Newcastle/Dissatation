# Dissatation

This repository contains the implementation for locally deploying the Base Llama 3.1 8B model for dissertation research.

## Contents

- **Base Llama 3.1 8B Model Implementation**: A clean, modular implementation for running Meta's Llama 3.1 8B base model locally
- **Dissertation Proposal**: CSC3094 Dissertation Proposal document

## Quick Start

See [README_MODEL.md](README_MODEL.md) for detailed instructions on setting up and running the base Llama 3.1 8B model.

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for model access)
huggingface-cli login

# Run the model
python inference.py
```

## Files

- `requirements.txt` - Python dependencies
- `config.py` - Model configuration
- `llama_base_model.py` - Main model implementation
- `inference.py` - Interactive inference script
- `README_MODEL.md` - Detailed documentation