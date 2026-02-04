# Training Phase 1: Siamese Network (25-100 labels)

## Overview

Siamese networks learn **similarity metrics** rather than decision boundaries. This makes them ideal for few-shot learning where you have 25-100 labeled examples.

---

## Label Budget Allocation

With 25 total labels:

| Purpose | Count | Notes |
|---------|-------|-------|
| Initial seed set | 5-8 | Diverse random sampling |
| Active learning rounds | 12-17 | 4-5 labels × 3-4 rounds |
| Held-out validation | 2-3 | Never used for training |

---

## Pair Generation

Training Siamese networks requires **pairs** of images with similarity labels.

```python
import random
from typing import List, Tuple
import torch


def create_training_pairs(
    labeled_crops: List[torch.Tensor],
    labels: List[int],  # 1 = signal, 0 = not signal
    pairs_per_sample: int = 10,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[float]]:
    """
    Generate training pairs from labeled samples.

    For N labeled samples, can generate up to N*(N-1)/2 pairs.
    With 25 samples: up to 300 pairs.

    Args:
        labeled_crops: List of preprocessed crop tensors
        labels: Binary labels (1=signal, 0=not-signal)
        pairs_per_sample: How many pairs to create per sample

    Returns:
        (pairs, pair_labels) where:
            pairs: List of (crop_a, crop_b) tuples
            pair_labels: 1.0 if same class, 0.0 if different
    """
    n = len(labeled_crops)
    pairs = []
    pair_labels = []

    # Separate by class
    positives = [(crop, i) for i, (crop, label) in enumerate(zip(labeled_crops, labels)) if label == 1]
    negatives = [(crop, i) for i, (crop, label) in enumerate(zip(labeled_crops, labels)) if label == 0]

    # Generate positive pairs (same class)
    for crop_a, idx_a in positives:
        for crop_b, idx_b in positives:
            if idx_a < idx_b:
                pairs.append((crop_a, crop_b))
                pair_labels.append(1.0)

    for crop_a, idx_a in negatives:
        for crop_b, idx_b in negatives:
            if idx_a < idx_b:
                pairs.append((crop_a, crop_b))
                pair_labels.append(1.0)

    # Generate negative pairs (different classes)
    for crop_pos, _ in positives:
        for crop_neg, _ in negatives:
            pairs.append((crop_pos, crop_neg))
            pair_labels.append(0.0)

    return pairs, pair_labels
```

---

## Contrastive Loss

```python
import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss for Siamese networks.

    Similar pairs (label=1): minimize distance
    Different pairs (label=0): maximize distance up to margin

    Loss = label * distance^2 + (1-label) * max(0, margin - distance)^2
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embedding1: (B, D) embeddings from encoder
            embedding2: (B, D) embeddings from encoder
            label: (B,) binary labels, 1=similar, 0=different

        Returns:
            Scalar loss
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)

        # Contrastive loss
        positive_loss = label * distance.pow(2)
        negative_loss = (1 - label) * F.relu(self.margin - distance).pow(2)

        loss = positive_loss + negative_loss
        return loss.mean()
```

---

## Training Loop

```python
import torch
from torch.utils.data import DataLoader, TensorDataset


def train_siamese(
    model: SiameseNetwork,
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    train_labels: List[float],
    val_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    val_labels: List[float],
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    device: str = 'cuda',
) -> dict:
    """
    Train Siamese network on pairs.

    Returns:
        Training history dict
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(margin=1.0)

    # Prepare data
    train_a = torch.stack([p[0] for p in train_pairs])
    train_b = torch.stack([p[1] for p in train_pairs])
    train_y = torch.tensor(train_labels, dtype=torch.float32)

    train_dataset = TensorDataset(train_a, train_b, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        for batch_a, batch_b, batch_y in train_loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            emb_a = model.encode(batch_a)
            emb_b = model.encode(batch_b)

            loss = criterion(emb_a, emb_b, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        history['train_loss'].append(epoch_loss)

        # Validation
        val_loss, val_acc = evaluate_siamese(model, val_pairs, val_labels, device)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

    # Restore best
    model.load_state_dict(best_state)

    return history


def evaluate_siamese(
    model: SiameseNetwork,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: List[float],
    device: str,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """Evaluate Siamese model on pairs."""
    model.eval()
    criterion = ContrastiveLoss()

    with torch.no_grad():
        all_a = torch.stack([p[0] for p in pairs]).to(device)
        all_b = torch.stack([p[1] for p in pairs]).to(device)
        all_y = torch.tensor(labels, dtype=torch.float32).to(device)

        emb_a = model.encode(all_a)
        emb_b = model.encode(all_b)

        loss = criterion(emb_a, emb_b, all_y).item()

        # Accuracy: predict similar if distance < threshold
        distances = F.pairwise_distance(emb_a, emb_b)
        predictions = (distances < threshold).float()
        accuracy = (predictions == all_y).float().mean().item()

    return loss, accuracy
```

---

## Building the Gallery

After training, build a gallery of signal embeddings for inference:

```python
def build_signal_gallery(
    model: SiameseNetwork,
    positive_crops: List[torch.Tensor],
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Compute embeddings for all positive (signal) examples.

    These form the "gallery" that new crops are compared against.

    Args:
        model: Trained Siamese network
        positive_crops: All labeled signal crops
        device: Compute device

    Returns:
        Gallery tensor of shape (N, embedding_dim)
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        crops_tensor = torch.stack(positive_crops).to(device)
        gallery = model.encode(crops_tensor)

    return gallery.cpu()
```

---

## Preventing Mode Collapse

Siamese networks can collapse to outputting constant embeddings. Mitigations:

### 1. L2 Normalization

Already included in the encoder - embeddings live on unit hypersphere.

### 2. Triplet Loss (Alternative)

If contrastive loss causes collapse, switch to triplet loss:

```python
class TripletLoss(torch.nn.Module):
    """
    Triplet loss: anchor, positive, negative.

    Pull anchor-positive together, push anchor-negative apart.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
```

### 3. Hard Negative Mining

After initial epochs, focus on hard examples:

```python
def mine_hard_negatives(model, positives, negatives, k=5):
    """
    Find negatives that are closest to positives in embedding space.
    """
    model.eval()
    with torch.no_grad():
        pos_emb = model.encode(positives)
        neg_emb = model.encode(negatives)

        # Distance matrix: (N_pos, N_neg)
        distances = torch.cdist(pos_emb, neg_emb)

        # For each positive, find k closest negatives
        _, hard_indices = distances.topk(k, dim=1, largest=False)

    return hard_indices
```

---

## Expected Results

| Labels | Pairs Generated | Expected Accuracy |
|--------|-----------------|-------------------|
| 25 | ~300 | 70-78% |
| 50 | ~1,200 | 75-82% |
| 100 | ~4,900 | 80-88% |

These accuracies are sufficient for active learning - the model just needs to be better than random to prioritize uncertain samples.
