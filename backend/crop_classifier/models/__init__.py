"""
Neural network models for crop classification.

- siamese.py: Siamese network for few-shot learning (25-100 labels)
- classifier.py: Direct CNN classifier (100+ labels) - Week 4
- losses.py: Contrastive loss, focal loss implementations
"""

from .losses import ContrastiveLoss, FocalLoss
from .siamese import SiameseClassifier, SiameseEncoder, SiameseNetwork

__all__ = [
    "SiameseEncoder",
    "SiameseNetwork",
    "SiameseClassifier",
    "ContrastiveLoss",
    "FocalLoss",
]
