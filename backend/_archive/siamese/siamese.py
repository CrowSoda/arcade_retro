"""
Siamese network for few-shot signal classification.

Siamese networks learn similarity metrics rather than decision boundaries,
making them ideal for scenarios with 25-100 labeled examples.

Architecture:
- 3-layer CNN encoder (32→64→128 channels)
- Global Average Pooling (position-invariant)
- 64-dimensional L2-normalized embedding

Key design choices:
1. NO ResNet - overkill for 32×64 crops, would overfit with <100 samples
2. L2 normalization - prevents mode collapse, embeddings on unit hypersphere
3. Global Average Pooling - reduces position encoding from conv layers
4. High dropout (0.3) - regularization for small datasets

Expected accuracy with training pairs:
- 25 labels (~300 pairs): 70-78%
- 50 labels (~1200 pairs): 75-82%
- 100 labels (~4900 pairs): 80-88%
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """
    Encodes a crop into a 64-dimensional embedding.

    Architecture designed for 32×64 (H×W) grayscale crops:
    - Block 1: 32×64 → 16×32, 32 channels
    - Block 2: 16×32 → 8×16, 64 channels
    - Block 3: 8×16 → global, 128 channels
    - Embedding: 128 → 64 dimensions

    The embedding is L2-normalized to lie on a unit hypersphere.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        input_channels: int = 1,
    ):
        """
        Initialize encoder.

        Args:
            embedding_dim: Output embedding dimension (default 64)
            dropout: Dropout rate before embedding layer
            input_channels: Number of input channels (1 for grayscale)
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            # Block 1: (1, 32, 64) → (32, 16, 32)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: (32, 16, 32) → (64, 8, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: (64, 8, 16) → (128, 1, 1)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Global Average Pooling - CRITICAL for position invariance
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # Embedding projection
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to L2-normalized embeddings.

        Args:
            x: (B, 1, H, W) input crops

        Returns:
            (B, embedding_dim) normalized embeddings
        """
        embedding = self.encoder(x)
        # L2 normalize - embeddings on unit hypersphere
        # Prevents mode collapse in contrastive learning
        return F.normalize(embedding, p=2, dim=1)


class SiameseNetwork(nn.Module):
    """
    Full Siamese network for similarity learning.

    Contains a single encoder (weight-shared) that processes both
    anchor and comparison images. Similarity is computed as the
    inverse of Euclidean distance in embedding space.

    Example:
        model = SiameseNetwork()
        emb1 = model.encode(crop1)  # (B, 64)
        emb2 = model.encode(crop2)  # (B, 64)
        similarity = model.similarity(emb1, emb2)  # (B,) values 0-1
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ):
        """
        Initialize Siamese network.

        Args:
            embedding_dim: Embedding dimension (default 64)
            dropout: Dropout rate in encoder
        """
        super().__init__()
        self.encoder = SiameseEncoder(embedding_dim, dropout)
        self.embedding_dim = embedding_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode single image or batch to embedding.

        Args:
            x: (B, 1, H, W) input crops

        Returns:
            (B, embedding_dim) normalized embeddings
        """
        return self.encoder(x)

    def forward(
        self,
        anchor: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between anchor and other.

        Args:
            anchor: (B, 1, H, W) anchor crops
            other: (B, 1, H, W) comparison crops

        Returns:
            (B,) similarity scores in range [0, 1]
            Higher = more similar
        """
        emb_anchor = self.encode(anchor)
        emb_other = self.encode(other)

        return self.similarity(emb_anchor, emb_other)

    def similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity from pre-computed embeddings.

        Uses inverse Euclidean distance: sim = 1 / (1 + distance)
        Range: [0, 1] where 1 = identical

        Args:
            emb1: (B, D) or (D,) embedding
            emb2: (B, D) or (D,) embedding

        Returns:
            (B,) or scalar similarity scores
        """
        # Ensure 2D
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)

        distance = F.pairwise_distance(emb1, emb2)
        return 1.0 / (1.0 + distance)


class SiameseClassifier:
    """
    Binary classifier using trained Siamese network.

    Compares each crop against a gallery of known signal embeddings.
    Classification is based on maximum similarity to any gallery item.

    Example:
        # After training Siamese network
        gallery = build_signal_gallery(model, positive_crops)
        classifier = SiameseClassifier(model, gallery, threshold=0.5)

        # Inference
        crops = preprocessor.extract_batch(spectrogram, boxes)
        scores, labels = classifier.classify(crops)
    """

    def __init__(
        self,
        model: SiameseNetwork,
        gallery_embeddings: torch.Tensor,
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Initialize classifier.

        Args:
            model: Trained Siamese network
            gallery_embeddings: (N, embedding_dim) embeddings of known signals
            threshold: Classification threshold (above = signal)
            device: Compute device
        """
        self.model = model.to(device)
        self.model.eval()

        self.gallery = gallery_embeddings.to(device)
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def classify(
        self,
        crops: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Classify batch of crops.

        Args:
            crops: (B, 1, H, W) preprocessed crops

        Returns:
            (scores, labels) where:
                scores: (B,) max similarity to gallery (confidence)
                labels: (B,) binary labels (1=signal, 0=not signal)
        """
        crops = crops.to(self.device)

        # Encode all crops
        crop_embeddings = self.model.encode(crops)  # (B, D)

        # Compute similarity to all gallery items
        # crop_embeddings: (B, D), gallery: (N, D)
        # Use matrix multiply for efficiency
        # Since embeddings are L2-normalized, dot product = cosine similarity
        # But we want inverse distance, so compute explicitly

        scores = self._compute_max_similarity(crop_embeddings)
        labels = (scores >= self.threshold).long()

        return scores, labels

    def _compute_max_similarity(self, crop_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum similarity to gallery for each crop.

        Uses batch matrix operations for efficiency.
        """
        # Expand for broadcasting
        # crop_embs: (B, D) → (B, 1, D)
        # gallery: (N, D) → (1, N, D)
        crop_exp = crop_embs.unsqueeze(1)  # (B, 1, D)
        gallery_exp = self.gallery.unsqueeze(0)  # (1, N, D)

        # Euclidean distance: (B, N)
        distances = torch.norm(crop_exp - gallery_exp, dim=2)

        # Min distance → max similarity
        min_dist, _ = distances.min(dim=1)

        # Convert to similarity
        return 1.0 / (1.0 + min_dist)

    @torch.no_grad()
    def get_confidence_batch(
        self,
        crops: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get confidence scores without labels.

        Args:
            crops: (B, 1, H, W) preprocessed crops

        Returns:
            (B,) confidence scores in [0, 1]
        """
        scores, _ = self.classify(crops)
        return scores

    def update_threshold(self, threshold: float) -> None:
        """Update classification threshold."""
        self.threshold = threshold

    def add_to_gallery(self, new_embeddings: torch.Tensor) -> None:
        """
        Add new signal embeddings to gallery.

        Args:
            new_embeddings: (M, D) new embeddings to add
        """
        new_embeddings = new_embeddings.to(self.device)
        self.gallery = torch.cat([self.gallery, new_embeddings], dim=0)


def build_signal_gallery(
    model: SiameseNetwork,
    positive_crops: list[torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute embeddings for all positive (signal) examples.

    These form the "gallery" that new crops are compared against
    during inference.

    Args:
        model: Trained Siamese network
        positive_crops: List of preprocessed signal crops
        device: Compute device

    Returns:
        Gallery tensor of shape (N, embedding_dim)
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        if isinstance(positive_crops, list):
            crops_tensor = torch.stack(positive_crops)
        else:
            crops_tensor = positive_crops

        crops_tensor = crops_tensor.to(device)
        gallery = model.encode(crops_tensor)

    return gallery.cpu()


def create_training_pairs(
    labeled_crops: list[torch.Tensor],
    labels: list[int],
    max_pairs: int | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float]]:
    """
    Generate training pairs from labeled samples.

    For N labeled samples, can generate up to N*(N-1)/2 pairs.
    With 25 samples: up to 300 pairs.

    Args:
        labeled_crops: List of preprocessed crop tensors
        labels: Binary labels (1=signal, 0=not-signal)
        max_pairs: Optional limit on total pairs generated

    Returns:
        (pairs, pair_labels) where:
            pairs: List of (crop_a, crop_b) tuples
            pair_labels: 1.0 if same class, 0.0 if different
    """
    import random

    # Separate by class
    positives = [
        (crop, i)
        for i, (crop, label) in enumerate(zip(labeled_crops, labels, strict=False))
        if label == 1
    ]
    negatives = [
        (crop, i)
        for i, (crop, label) in enumerate(zip(labeled_crops, labels, strict=False))
        if label == 0
    ]

    pairs = []
    pair_labels = []

    # Generate positive pairs (same class)
    for i, (crop_a, idx_a) in enumerate(positives):
        for crop_b, idx_b in positives[i + 1 :]:
            pairs.append((crop_a, crop_b))
            pair_labels.append(1.0)

    for i, (crop_a, idx_a) in enumerate(negatives):
        for crop_b, idx_b in negatives[i + 1 :]:
            pairs.append((crop_a, crop_b))
            pair_labels.append(1.0)

    # Generate negative pairs (different classes)
    for crop_pos, _ in positives:
        for crop_neg, _ in negatives:
            pairs.append((crop_pos, crop_neg))
            pair_labels.append(0.0)

    # Shuffle and optionally limit
    combined = list(zip(pairs, pair_labels, strict=False))
    random.shuffle(combined)

    if max_pairs is not None and len(combined) > max_pairs:
        combined = combined[:max_pairs]

    if combined:
        pairs, pair_labels = zip(*combined, strict=False)
        return list(pairs), list(pair_labels)

    return [], []
