"""
Hydra - Multi-head detection system with shared backbone.

This module provides:
    - HydraDetector: Shared-backbone multi-head inference
    - VersionManager: Model version control with auto-promotion
    - BackboneExtractor: One-time migration from full models
"""

# Logging
from logger_config import get_logger

from .config import *

logger = get_logger("__init__")
