"""
Loss functions for crop classifier training.

ContrastiveLoss: For Siamese network training (few-shot, 25-100 labels)
FocalLoss: For direct CNN classifier (100+ labels, handles class imbalance)
TripletLoss: Alternative to contrastive if mode collapse occurs
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.

    Similar pairs (label=1): minimize distance between embeddings
    Different pairs (label=0): maximize distance up to margin

    Loss = label * distance² + (1-label) * max(0, margin - distance)²

    The margin parameter controls how far apart different-class
    embeddings should be pushed. Default 1.0 works well for
    L2-normalized embeddings.

    Example:
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = model.encode(batch1)  # (B, 64)
        emb2 = model.encode(batch2)  # (B, 64)
        labels = torch.tensor([1.0, 0.0, 1.0, ...])  # 1=same, 0=different
        loss = loss_fn(emb1, emb2, labels)
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.

        Args:
            margin: Distance margin for negative pairs (default 1.0)
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embedding1: (B, D) embeddings from encoder
            embedding2: (B, D) embeddings from encoder
            label: (B,) binary labels, 1=similar, 0=different

        Returns:
            Scalar loss (mean over batch)
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)

        # Positive pairs: minimize distance
        positive_loss = label * distance.pow(2)

        # Negative pairs: push apart up to margin
        negative_loss = (1 - label) * F.relu(self.margin - distance).pow(2)

        loss = positive_loss + negative_loss
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss for Siamese networks.

    Alternative to contrastive loss that may be more stable.
    Uses (anchor, positive, negative) triplets instead of pairs.

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Example:
        loss_fn = TripletLoss(margin=0.5)
        anchor_emb = model.encode(anchors)
        positive_emb = model.encode(positives)
        negative_emb = model.encode(negatives)
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
    """

    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss.

        Args:
            margin: Margin between positive and negative distances
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: (B, D) anchor embeddings
            positive: (B, D) positive (same class) embeddings
            negative: (B, D) negative (different class) embeddings

        Returns:
            Scalar loss (mean over batch)
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with class imbalance.

    Focal loss down-weights easy examples and focuses on hard ones.
    This is critical when you have many more negatives (noise blobs)
    than positives (actual signals).

    FL = -α * (1-p)^γ * log(p)  for positive class
    FL = -(1-α) * p^γ * log(1-p)  for negative class

    Where:
    - α: Class weight (higher = more weight on positives)
    - γ: Focusing parameter (higher = more focus on hard examples)
    - p: Predicted probability

    Default γ=2.0 and α=0.25 work well for detection tasks.

    Example:
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = model(crops)  # (B, 1)
        labels = torch.tensor([1, 0, 1, 0, ...])  # (B,)
        loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weight for positive class (default 0.25)
            gamma: Focusing parameter (default 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: (B,) or (B, 1) logits (before sigmoid)
            targets: (B,) binary labels (0 or 1)

        Returns:
            Loss scalar (if reduction='mean'/'sum') or (B,) tensor
        """
        # Flatten inputs if needed
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities
        p = torch.sigmoid(inputs)

        # Compute focal weights
        # For positives: α * (1-p)^γ
        # For negatives: (1-α) * p^γ
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCE(nn.Module):
    """
    Binary cross-entropy with label smoothing.

    Converts hard labels [0, 1] to soft labels [ε, 1-ε].
    Helps prevent overconfident predictions and improves calibration.

    Example:
        loss_fn = LabelSmoothingBCE(smoothing=0.1)
        logits = model(crops)
        labels = torch.tensor([1, 0, 1, 0, ...])
        loss = loss_fn(logits, labels)
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing BCE.

        Args:
            smoothing: Label smoothing factor (default 0.1)
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute smoothed BCE loss.

        Args:
            inputs: (B,) or (B, 1) logits
            targets: (B,) binary labels

        Returns:
            Mean loss scalar
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Smooth labels: 0 → ε, 1 → 1-ε
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        return F.binary_cross_entropy_with_logits(inputs, smooth_targets)


class CombinedLoss(nn.Module):
    """
    Combine multiple losses with weights.

    Useful for multi-task learning or regularization.

    Example:
        loss_fn = CombinedLoss([
            (FocalLoss(), 1.0),
            (nn.L1Loss(), 0.1),  # Embedding regularization
        ])
    """

    def __init__(self, losses_and_weights: list[tuple[nn.Module, float]]):
        """
        Initialize combined loss.

        Args:
            losses_and_weights: List of (loss_fn, weight) tuples
        """
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses_and_weights])
        self.weights = [weight for _, weight in losses_and_weights]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute weighted sum of losses.

        All losses receive the same arguments.
        """
        total = 0.0
        for loss_fn, weight in zip(self.losses, self.weights, strict=False):
            total = total + weight * loss_fn(*args, **kwargs)
        return total
