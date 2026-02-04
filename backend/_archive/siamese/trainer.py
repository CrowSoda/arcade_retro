"""
Siamese network training loop for crop classification.

Trains from labeled pairs using contrastive loss.
Expected accuracy with training pairs:
- 25 labels (~300 pairs): 70-78%
- 50 labels (~1200 pairs): 75-82%
- 100 labels (~4900 pairs): 80-88%
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..models.losses import ContrastiveLoss
from ..models.siamese import SiameseNetwork, create_training_pairs

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for Siamese training."""

    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 16
    margin: float = 1.0
    weight_decay: float = 0.0001
    max_grad_norm: float = 1.0
    early_stop_patience: int = 10
    device: str = "cuda"
    checkpoint_dir: Path | None = None
    log_interval: int = 10


@dataclass
class TrainingResult:
    """Result of training run."""

    final_loss: float
    best_loss: float
    epochs_trained: int
    pairs_used: int
    model_path: Path | None = None


class PairDataset(Dataset):
    """Dataset of crop pairs for contrastive learning."""

    def __init__(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        labels: list[float],
    ):
        self.pairs = pairs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        anchor, other = self.pairs[idx]
        label = self.labels[idx]
        return anchor, other, label


class SiameseTrainer:
    """
    Trains Siamese network from labeled crop pairs.

    Example:
        trainer = SiameseTrainer(config)
        result = trainer.train(labeled_crops, labels)
        model = trainer.model
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        model: SiameseNetwork | None = None,
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.model = model or SiameseNetwork()
        self.model = self.model.to(self.device)

        self.criterion = ContrastiveLoss(margin=self.config.margin)
        self.optimizer: optim.Optimizer | None = None
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def train(
        self,
        labeled_crops: list[torch.Tensor],
        labels: list[int],
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> TrainingResult:
        """
        Train Siamese network from labeled data.

        Args:
            labeled_crops: List of preprocessed crop tensors (1, H, W)
            labels: Binary labels (1=signal, 0=not-signal)
            progress_callback: Called with (epoch, total_epochs, loss)

        Returns:
            TrainingResult with training statistics
        """
        logger.info(f"Creating training pairs from {len(labeled_crops)} samples")

        # Generate pairs
        pairs, pair_labels = create_training_pairs(labeled_crops, labels)

        if len(pairs) < 10:
            raise ValueError(
                f"Not enough training pairs ({len(pairs)}). "
                f"Need at least 10 pairs from labeled data."
            )

        logger.info(f"Generated {len(pairs)} training pairs")

        # Create dataloader
        dataset = PairDataset(pairs, pair_labels)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues on Windows
            pin_memory=self.device.type == "cuda",
        )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Training loop
        self.model.train()
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            epoch_loss = self._train_epoch(dataloader, epoch)

            # Early stopping check
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, epoch_loss)
            else:
                self.epochs_without_improvement += 1

            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, self.config.epochs, epoch_loss)

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {self.config.early_stop_patience} epochs)"
                )
                break

        # Load best model
        self._load_best_checkpoint()

        model_path = None
        if self.config.checkpoint_dir:
            model_path = self.config.checkpoint_dir / "siamese_best.pth"

        return TrainingResult(
            final_loss=epoch_loss,
            best_loss=self.best_loss,
            epochs_trained=epoch + 1,
            pairs_used=len(pairs),
            model_path=model_path,
        )

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train single epoch, return average loss."""
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (anchor, other, target) in enumerate(dataloader):
            anchor = anchor.to(self.device)
            other = other.to(self.device)
            target = target.float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            emb_anchor = self.model.encode(anchor)
            emb_other = self.model.encode(other)

            loss = self.criterion(emb_anchor, emb_other, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        return avg_loss

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint."""
        if not self.config.checkpoint_dir:
            return

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.checkpoint_dir / "siamese_best.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_path,
        )
        logger.debug(f"Saved checkpoint to {checkpoint_path}")

    def _load_best_checkpoint(self) -> None:
        """Load best model from checkpoint."""
        if not self.config.checkpoint_dir:
            return

        checkpoint_path = self.config.checkpoint_dir / "siamese_best.pth"
        if not checkpoint_path.exists():
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    def get_model(self) -> SiameseNetwork:
        """Get trained model."""
        return self.model

    def save_model(self, path: Path) -> None:
        """Save model weights to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")

    def load_model(self, path: Path) -> None:
        """Load model weights from file."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        logger.info(f"Loaded model from {path}")


def train_siamese_from_samples(
    sample_dir: Path,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """
    Convenience function to train from sample directory.

    Expects sample_dir to contain:
    - crops/*.png - Crop images
    - labels.json - {"crop_id": 0_or_1, ...}

    Args:
        sample_dir: Directory with crops and labels
        output_dir: Where to save trained model
        config: Training configuration

    Returns:
        TrainingResult
    """
    import json

    import cv2
    import numpy as np

    config = config or TrainingConfig()
    config.checkpoint_dir = output_dir

    # Load labels
    labels_path = sample_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path) as f:
        labels_dict = json.load(f)

    # Load crops
    crops_dir = sample_dir / "crops"
    if not crops_dir.exists():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    labeled_crops = []
    labels = []

    for crop_id, label in labels_dict.items():
        crop_path = crops_dir / f"{crop_id}.png"
        if not crop_path.exists():
            logger.warning(f"Crop not found: {crop_path}")
            continue

        # Load and convert to tensor
        img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)
        labeled_crops.append(tensor)
        labels.append(int(label))

    logger.info(f"Loaded {len(labeled_crops)} labeled crops")

    if len(labeled_crops) < 10:
        raise ValueError(f"Not enough labeled crops ({len(labeled_crops)})")

    # Train
    trainer = SiameseTrainer(config)
    result = trainer.train(labeled_crops, labels)

    # Save final model
    trainer.save_model(output_dir / "siamese.pth")

    return result
