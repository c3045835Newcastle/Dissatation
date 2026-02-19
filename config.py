"""
Configuration file for Base Llama 3.1 8B Model
This configuration is for the BASE model (not fine-tuned).
"""

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"  # Base model identifier
MODEL_CACHE_DIR = "./models"  # Local cache directory for model files

# Generation parameters (base defaults)
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True

# Device configuration
DEVICE = "auto"  # Will automatically select cuda if available, else cpu

# Model loading parameters
LOAD_IN_8BIT = False  # Set to True to reduce memory usage
LOAD_IN_4BIT = False  # Set to True for even lower memory usage (requires bitsandbytes)
USE_FLASH_ATTENTION = False  # Requires flash-attention package
