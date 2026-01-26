"""
Hydra Configuration Constants

These values are locked to match inference and training requirements.
"""

# =============================================================================
# Version Management
# =============================================================================

AUTO_PROMOTE_THRESHOLD = 0.02      # 2% F1 improvement required (below HIGH_F1)
HIGH_F1_THRESHOLD = 0.95           # Above this, just require no regression
VERSION_RETENTION_COUNT = 5        # Keep last N versions per signal
MIN_SAMPLES_FOR_TRAINING = 5       # Minimum labeled samples

# =============================================================================
# Training
# =============================================================================

DEFAULT_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 4
VAL_RATIO = 0.2

# =============================================================================
# Inference
# =============================================================================

DEFAULT_SCORE_THRESHOLD = 0.5
MAX_DETECTIONS_PER_HEAD = 100

# =============================================================================
# FFT Parameters (LOCKED - must match inference.py)
# =============================================================================

INFERENCE_FFT_SIZE = 4096
INFERENCE_HOP_LENGTH = 2048
INFERENCE_DYNAMIC_RANGE_DB = 80.0
INFERENCE_OUTPUT_SIZE = (1024, 1024)

# =============================================================================
# File Paths (relative to g20_demo/)
# =============================================================================

BACKBONE_DIR = "models/backbone"
HEADS_DIR = "models/heads"
LEGACY_DIR = "models/legacy"
TRAINING_DATA_DIR = "training_data/signals"
REGISTRY_PATH = "models/registry.json"
