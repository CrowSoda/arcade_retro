# Training Phase 2: Direct CNN Classifier (100+ labels)

## Overview

Once you have 100+ labeled samples, transition from Siamese to direct classification. This is simpler, faster at inference, and more accurate with sufficient data.

---

## When to Transition

| Metric | Siamese | Direct CNN |
|--------|---------|------------|
| Min labels | 25 | 100 |
| Inference time | ~15ms (gallery compare) | ~8ms (single pass) |
| Accuracy ceiling | ~88% | ~97% |
| Training complexity | Pair generation | Standard batches |

**Transition signal:** When Siamese accuracy plateaus AND you have 100+ labels.

---

## Focal Loss (Critical for Class Imbalance)

Blob detection generates many false positives. Your true signal:noise ratio might be 1:10 or worse. Focal loss dynamically down-weights easy examples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.

    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - alpha: Weight for rare (positive) class
    - gamma: Focusing parameter (higher = more focus on hard examples)

    Default params from paper: gamma=2.0, alpha=0.25
    Outperforms Online Hard Example Mining by 3.2 AP.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) raw model output (before sigmoid)
            targets: (B, 1) binary labels

        Returns:
            Scalar focal loss
        """
        # Standard BCE (unreduced)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Compute p_t
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final loss
        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()
```

---

## Training Loop

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Optional


def train_classifier(
    model: SignalClassifier,
    train_crops: torch.Tensor,
    train_labels: torch.Tensor,
    val_crops: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int = 75,
    learning_rate: float = 0.001,
    batch_size: int = 8,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 15,
    device: str = 'cuda',
    callback: Optional[Callable] = None,
) -> dict:
    """
    Train direct CNN classifier.

    Args:
        model: SignalClassifier instance
        train_crops: (N, 1, 64, 64) training crops
        train_labels: (N, 1) binary labels
        val_crops: Validation crops
        val_labels: Validation labels
        epochs: Max training epochs
        learning_rate: Initial LR
        batch_size: Batch size (8-16 for regularization)
        weight_decay: L2 regularization
        early_stop_patience: Stop if no improvement for N epochs
        device: Compute device
        callback: Optional progress callback

    Returns:
        Training history dict
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Warmup + cosine annealing
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Data loaders
    train_dataset = TensorDataset(train_crops, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Avoid tiny final batches
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
    }

    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        history['train_loss'].append(epoch_loss)

        # --- Validation ---
        metrics = evaluate_classifier(model, val_crops, val_labels, device)
        history['val_loss'].append(metrics['loss'])
        history['val_f1'].append(metrics['f1'])
        history['val_precision'].append(metrics['precision'])
        history['val_recall'].append(metrics['recall'])

        # --- Early stopping ---
        is_best = metrics['f1'] > best_f1
        if is_best:
            best_f1 = metrics['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Callback ---
        if callback:
            callback({
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'train_loss': epoch_loss,
                'val_f1': metrics['f1'],
                'is_best': is_best,
            })

        # --- Check early stop ---
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    return history


def evaluate_classifier(
    model: SignalClassifier,
    crops: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate classifier and compute metrics.
    """
    model.eval()
    criterion = FocalLoss()

    with torch.no_grad():
        crops = crops.to(device)
        labels = labels.to(device)

        logits = model(crops)
        loss = criterion(logits, labels).item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        # Metrics
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
```

---

## Regularization (Critical for <500 samples)

| Technique | Setting | Purpose |
|-----------|---------|---------|
| Dropout | 0.5 | Prevent co-adaptation |
| Weight decay | 1e-4 | L2 regularization |
| Batch size | 8-16 | Gradient noise for regularization |
| Early stopping | patience=15 | Prevent overfitting |
| Data augmentation | See doc 08 | Expand effective dataset |

---

## Handling Class Imbalance

Beyond Focal Loss, consider:

### 1. Oversampling Positives

```python
def balance_dataset(crops, labels, target_ratio=0.3):
    """
    Oversample minority class to achieve target positive ratio.
    """
    pos_mask = labels.squeeze() == 1
    pos_crops = crops[pos_mask]
    neg_crops = crops[~pos_mask]

    n_neg = len(neg_crops)
    n_pos_target = int(n_neg * target_ratio / (1 - target_ratio))

    # Repeat positives
    repeats = n_pos_target // len(pos_crops) + 1
    pos_crops_expanded = pos_crops.repeat(repeats, 1, 1, 1)[:n_pos_target]

    # Combine
    balanced_crops = torch.cat([neg_crops, pos_crops_expanded])
    balanced_labels = torch.cat([
        torch.zeros(n_neg, 1),
        torch.ones(n_pos_target, 1)
    ])

    # Shuffle
    perm = torch.randperm(len(balanced_crops))
    return balanced_crops[perm], balanced_labels[perm]
```

### 2. Weighted Sampling

```python
from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(labels):
    """
    Create sampler that draws positive/negative equally.
    """
    class_counts = [(labels == 0).sum(), (labels == 1).sum()]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = weights[labels.long().squeeze()]

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
```

---

## Expected Results

| Labels | Expected F1 | Notes |
|--------|-------------|-------|
| 100 | 82-88% | Focal loss critical |
| 250 | 88-93% | + augmentation |
| 500 | 92-96% | Production ready |
| 1000+ | 95-98% | Near ceiling |
