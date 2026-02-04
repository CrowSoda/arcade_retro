"""
Hydra Configuration Constants

These values are locked to match inference and training requirements.

Training presets are based on research from:
- TFA (ICML 2020): Fine-tuning with frozen backbone
- DeFRCN (ICCV 2021): SGD settings for detection
- CFA (CVPR 2022): Novel class adaptation
- Foundation Models (CVPR 2024): Adam with frozen backbone
- Expandable-RCNN (2024): Fast convergence settings
- Intel batch size research: Small batch → flat minima → better generalization
"""

from dataclasses import dataclass
from enum import Enum

# Logging
from logger_config import get_logger

logger = get_logger("config")


# =============================================================================
# Training Presets (Research-Based)
# =============================================================================


class TrainingPreset(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    EXTREME = "extreme"  # Maximum epochs for difficult signals


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration.

    Research basis:
    - TFA (ICML 2020): Frozen backbone fine-tuning
    - DeFRCN (ICCV 2021): Gradient decoupling
    - Few-shot warmup: 1000-2000 iterations for stability
    """

    epochs: int
    early_stop_patience: int
    learning_rate: float
    batch_size: int
    val_ratio: float
    min_samples: int
    warmup_iterations: int  # Research: critical for few-shot stability (1000-2000)
    description: str
    emoji: str


TRAINING_PRESETS: dict[TrainingPreset, TrainingConfig] = {
    # ⚡ FAST: Quick validation, check if labels work
    # Research basis: Foundation Models paper uses 3 epochs with frozen backbone
    # LR 0.001 (research standard for few-shot) - changed from 0.005
    TrainingPreset.FAST: TrainingConfig(
        epochs=15,
        early_stop_patience=2,  # TFA implementation uses patience=2
        learning_rate=0.001,  # Research: 0.001 for few-shot fine-tuning
        batch_size=8,  # Larger batch = fewer steps
        val_ratio=0.2,
        min_samples=5,
        warmup_iterations=500,  # Short warmup for fast preset
        description="Quick validation (~1-2 min)",
        emoji="⚡",
    ),
    # ⚖️ BALANCED: Production default
    # Research basis: TFA/DeFRCN standard settings
    # LR 0.001 is consensus for frozen backbone fine-tuning
    TrainingPreset.BALANCED: TrainingConfig(
        epochs=30,
        early_stop_patience=5,
        learning_rate=0.001,  # TFA/CFA standard
        batch_size=4,  # Balance noise vs stability
        val_ratio=0.2,
        min_samples=5,
        warmup_iterations=1000,  # Research: 1000-2000 for few-shot
        description="Production default (~3-5 min)",
        emoji="⚖️",
    ),
    # 🎯 QUALITY: Maximum accuracy
    # Research basis: Intel research shows small batches → flat minima → better generalization
    # LR 0.001 (research standard) with longer warmup
    TrainingPreset.QUALITY: TrainingConfig(
        epochs=75,
        early_stop_patience=10,
        learning_rate=0.001,  # Research: consistent 0.001 for few-shot
        batch_size=4,  # Research: 4-8 for regularization benefit
        val_ratio=0.2,
        min_samples=10,  # Require more data for quality
        warmup_iterations=1500,  # Longer warmup for quality
        description="Maximum accuracy (~10-15 min)",
        emoji="🎯",
    ),
    # 🔥 EXTREME: For difficult signals requiring extensive training
    # Research basis: When quality preset F1 plateaus, more epochs with very low LR
    # can help escape local minima and find better solutions
    TrainingPreset.EXTREME: TrainingConfig(
        epochs=150,
        early_stop_patience=20,  # Very patient - let it converge fully
        learning_rate=0.0005,  # Slightly lower for extended training
        batch_size=2,  # Small batch for regularization
        val_ratio=0.2,
        min_samples=15,  # Need good data diversity for long training
        warmup_iterations=2000,  # Full warmup for extreme
        description="Extended training (~25-40 min)",
        emoji="🔥",
    ),
}


def get_training_config(preset: TrainingPreset) -> TrainingConfig:
    """Get training configuration for a preset."""
    return TRAINING_PRESETS[preset]


def get_preset_by_name(name: str) -> TrainingPreset:
    """Get preset enum from string name."""
    name_lower = name.lower()
    for preset in TrainingPreset:
        if preset.value == name_lower:
            return preset
    raise ValueError(f"Unknown preset: {name}. Valid options: fast, balanced, quality, extreme")


# =============================================================================
# Legacy defaults (for backwards compatibility)
# =============================================================================

DEFAULT_PRESET = TrainingPreset.BALANCED
_default_config = TRAINING_PRESETS[DEFAULT_PRESET]

DEFAULT_EPOCHS = _default_config.epochs
EARLY_STOP_PATIENCE = _default_config.early_stop_patience
DEFAULT_LEARNING_RATE = _default_config.learning_rate
DEFAULT_BATCH_SIZE = _default_config.batch_size
VAL_RATIO = _default_config.val_ratio
MIN_SAMPLES_FOR_TRAINING = _default_config.min_samples

# =============================================================================
# Version Management
# =============================================================================

AUTO_PROMOTE_THRESHOLD = 0.02  # 2% F1 improvement required (below HIGH_F1)
HIGH_F1_THRESHOLD = 0.95  # Above this, just require no regression
VERSION_RETENTION_COUNT = 5  # Keep last N versions per signal

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
