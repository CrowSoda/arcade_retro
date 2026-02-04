"""
Crop-based signal classification module.

Two-stage detection approach:
1. Stage 1: Classical blob detection (position-invariant proposals)
2. Stage 2: Learned crop classifier (Siamese or CNN)

This eliminates the position bias inherent in end-to-end detection networks.
"""

from .inference.blob_detector import BlobDetector
from .inference.preprocessor import CropPreprocessor, preprocess_batch, preprocess_crop
from .models.losses import ContrastiveLoss, FocalLoss
from .models.siamese import SiameseClassifier, SiameseEncoder, SiameseNetwork

__all__ = [
    "BlobDetector",
    "CropPreprocessor",
    "preprocess_crop",
    "preprocess_batch",
    "SiameseEncoder",
    "SiameseNetwork",
    "SiameseClassifier",
    "ContrastiveLoss",
    "FocalLoss",
]
