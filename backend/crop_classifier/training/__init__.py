"""
Training components for crop classifier.

Classes:
- SiameseTrainer: Trains Siamese network from labeled pairs
- TrainingConfig: Training configuration dataclass
- TrainingResult: Training result dataclass

Functions:
- train_siamese_from_samples: Train from sample directory
"""

from .trainer import (
    SiameseTrainer,
    TrainingConfig,
    TrainingResult,
    train_siamese_from_samples,
)

__all__ = [
    "SiameseTrainer",
    "TrainingConfig",
    "TrainingResult",
    "train_siamese_from_samples",
]
