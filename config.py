"""
Configuration file for Base Llama 3.1 8B Model and Hierarchical Memory System.
"""

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Hierarchical Memory Architecture
# ---------------------------------------------------------------------------

# Working Memory
WORKING_MEMORY_MAX_TURNS = 20

# Episodic Memory – vector similarity search via FAISS
EPISODIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-transformer model
EPISODIC_TOP_K = 5          # Episodes retrieved per query
EPISODIC_STORAGE_PATH = "./memory_store/episodic"

# Semantic Memory – persistent user facts
SEMANTIC_STORAGE_PATH = "./memory_store/semantic_memory.json"

# Memory Controller – consolidate working memory into episodic memory every N user turns
MEMORY_CONSOLIDATION_INTERVAL = 5

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

EVALUATION_NUM_SCENARIOS = 5

# Path for persisting evaluation results
EVALUATION_RESULTS_PATH = "./results/evaluation"
