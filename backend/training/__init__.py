# Logging
from logger_config import get_logger

logger = get_logger("__init__")

"""
Training module - Train detection heads with frozen backbone.

Provides:
    - TrainingService: Manage training jobs
    - SpectrogramDataset: Load training data
    - SplitManager: Train/val split management
    - SampleManager: Save/manage training samples
"""
