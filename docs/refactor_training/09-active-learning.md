# Active Learning and Pseudo-Labeling

## Overview

Active learning maximizes information gain from each label. With only 25 labels available, every decision must count.

---

## Query Strategy: Uncertainty + Diversity Hybrid

Pure uncertainty sampling creates redundant selections (all uncertain samples may be similar). Diversity is critical for small label budgets.

### Algorithm

1. Score all unlabeled samples by uncertainty
2. Pre-filter top β×batch_size most uncertain
3. Cluster these by embedding similarity
4. Select one sample per cluster (closest to centroid)

```python
import torch
import numpy as np
from sklearn.cluster import KMeans


class HybridQueryStrategy:
    """
    Combines uncertainty sampling with diversity selection.

    Pure uncertainty often selects similar samples.
    This approach ensures diverse uncertain samples.
    """

    def __init__(self, beta: int = 3):
        """
        Args:
            beta: Over-sample factor for diversity filtering
                  e.g., beta=3 means select 3x batch_size uncertain,
                  then cluster to get batch_size diverse samples
        """
        self.beta = beta

    def select_batch(
        self,
        model: torch.nn.Module,
        unlabeled_crops: torch.Tensor,
        batch_size: int = 5,
        device: str = 'cuda',
    ) -> list[int]:
        """
        Select next batch of samples to label.

        Args:
            model: Current classifier
            unlabeled_crops: All unlabeled crops
            batch_size: Number of samples to select
            device: Compute device

        Returns:
            List of indices into unlabeled_crops
        """
        model.eval()

        with torch.no_grad():
            crops = unlabeled_crops.to(device)

            # Get predictions
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(crops).squeeze()
            else:
                logits = model(crops)
                probs = torch.sigmoid(logits).squeeze()

        # --- Step 1: Compute uncertainty (margin sampling) ---
        # Uncertainty is highest when prob = 0.5
        uncertainty = 1 - torch.abs(probs - 0.5) * 2

        # --- Step 2: Pre-filter top uncertain ---
        top_k = min(self.beta * batch_size, len(unlabeled_crops))
        _, uncertain_indices = torch.topk(uncertainty, top_k)
        uncertain_indices = uncertain_indices.cpu().numpy()

        if top_k <= batch_size:
            # Not enough samples to cluster
            return uncertain_indices[:batch_size].tolist()

        # --- Step 3: Get embeddings for clustering ---
        with torch.no_grad():
            if hasattr(model, 'features'):
                # Direct CNN - use feature extractor
                features = model.features(crops[uncertain_indices]).cpu().numpy()
            elif hasattr(model, 'encode'):
                # Siamese - use encoder
                features = model.encode(crops[uncertain_indices]).cpu().numpy()
            else:
                # Fallback - use predictions as features
                features = probs[uncertain_indices].unsqueeze(1).cpu().numpy()

        # --- Step 4: Cluster for diversity ---
        kmeans = KMeans(n_clusters=batch_size, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # --- Step 5: Select one per cluster (closest to centroid) ---
        selected = []
        for c in range(batch_size):
            cluster_mask = cluster_labels == c
            cluster_features = features[cluster_mask]
            cluster_indices = uncertain_indices[cluster_mask]

            if len(cluster_features) == 0:
                continue

            # Find closest to centroid
            centroid = cluster_features.mean(axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest = distances.argmin()

            selected.append(int(cluster_indices[closest]))

        return selected


def compute_uncertainty_scores(
    model: torch.nn.Module,
    crops: torch.Tensor,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Compute uncertainty scores for all crops.

    Returns tensor of shape (N,) with values 0-1.
    Higher = more uncertain.
    """
    model.eval()

    with torch.no_grad():
        crops = crops.to(device)

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(crops).squeeze()
        else:
            logits = model(crops)
            probs = torch.sigmoid(logits).squeeze()

    # Margin-based uncertainty: highest at 0.5
    uncertainty = 1 - torch.abs(probs - 0.5) * 2

    return uncertainty.cpu()
```

---

## Pseudo-Labeling for Dataset Expansion

Use high-confidence predictions to expand training data automatically.

```python
class PseudoLabeler:
    """
    Generate pseudo-labels from confident predictions.

    After each labeling round, high-confidence predictions
    become training data for the next round.
    """

    def __init__(
        self,
        initial_threshold: float = 0.9,
        min_threshold: float = 0.7,
        decay_per_round: float = 0.02,
    ):
        """
        Args:
            initial_threshold: Starting confidence threshold
            min_threshold: Minimum threshold after decay
            decay_per_round: How much to lower threshold each round
        """
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.decay = decay_per_round

    def generate_pseudo_labels(
        self,
        model: torch.nn.Module,
        unlabeled_crops: torch.Tensor,
        device: str = 'cuda',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels for high-confidence samples.

        Returns:
            (pseudo_positive_mask, pseudo_negative_mask, confidences)
        """
        model.eval()

        with torch.no_grad():
            crops = unlabeled_crops.to(device)

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(crops).squeeze()
            else:
                logits = model(crops)
                probs = torch.sigmoid(logits).squeeze()

        probs = probs.cpu()

        # High confidence positives
        pseudo_positive = probs > self.threshold

        # High confidence negatives
        pseudo_negative = probs < (1 - self.threshold)

        return pseudo_positive, pseudo_negative, probs

    def step(self):
        """Decay threshold for next round."""
        self.threshold = max(self.min_threshold, self.threshold - self.decay)

    def reset(self):
        """Reset threshold to initial value."""
        self.threshold = self.initial_threshold


def expand_with_pseudo_labels(
    train_crops: torch.Tensor,
    train_labels: torch.Tensor,
    pseudo_crops: torch.Tensor,
    pseudo_labels: torch.Tensor,
    max_pseudo_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine real labels with pseudo-labels.

    Args:
        train_crops: Real labeled crops
        train_labels: Real labels
        pseudo_crops: Pseudo-labeled crops
        pseudo_labels: Pseudo-labels (0 or 1)
        max_pseudo_ratio: Maximum ratio of pseudo to real labels

    Returns:
        Combined (crops, labels)
    """
    n_real = len(train_crops)
    n_pseudo_max = int(n_real * max_pseudo_ratio)
    n_pseudo = min(len(pseudo_crops), n_pseudo_max)

    if n_pseudo == 0:
        return train_crops, train_labels

    # Random sample from pseudo-labels
    perm = torch.randperm(len(pseudo_crops))[:n_pseudo]

    combined_crops = torch.cat([train_crops, pseudo_crops[perm]])
    combined_labels = torch.cat([train_labels, pseudo_labels[perm].unsqueeze(1)])

    return combined_crops, combined_labels
```

---

## Label Noise Detection with Cleanlab

Identify potentially mislabeled samples using confident learning.

```python
# pip install cleanlab

from cleanlab.filter import find_label_issues


def detect_label_errors(
    model: torch.nn.Module,
    crops: torch.Tensor,
    labels: torch.Tensor,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Find samples that may be mislabeled.

    Returns:
        Array of indices that may have incorrect labels
    """
    model.eval()

    # Get predicted probabilities
    with torch.no_grad():
        crops = crops.to(device)

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(crops).cpu().numpy()
        else:
            logits = model(crops)
            probs = torch.sigmoid(logits).cpu().numpy()

    # Convert to 2-class format for cleanlab
    pred_probs = np.hstack([1 - probs, probs])
    labels_np = labels.numpy().squeeze().astype(int)

    # Find issues
    issue_indices = find_label_issues(
        labels=labels_np,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence',
    )

    return issue_indices
```

---

## Active Learning Session Flow

```python
class ActiveLearningSession:
    """
    Manages the full active learning workflow.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        all_crops: torch.Tensor,
        query_strategy: HybridQueryStrategy,
        pseudo_labeler: PseudoLabeler,
        device: str = 'cuda',
    ):
        self.model = model
        self.all_crops = all_crops
        self.query_strategy = query_strategy
        self.pseudo_labeler = pseudo_labeler
        self.device = device

        # State
        self.labeled_indices = set()
        self.labels = {}  # index -> label
        self.round = 0

    def get_next_batch(self, batch_size: int = 5) -> list[int]:
        """Get next batch of samples to label."""
        unlabeled_mask = torch.tensor([
            i not in self.labeled_indices
            for i in range(len(self.all_crops))
        ])

        unlabeled_crops = self.all_crops[unlabeled_mask]
        unlabeled_indices = torch.where(unlabeled_mask)[0].tolist()

        # Get batch from query strategy
        local_indices = self.query_strategy.select_batch(
            self.model, unlabeled_crops, batch_size, self.device
        )

        # Map back to global indices
        return [unlabeled_indices[i] for i in local_indices]

    def add_labels(self, indices: list[int], labels: list[int]):
        """Record user labels."""
        for idx, label in zip(indices, labels):
            self.labeled_indices.add(idx)
            self.labels[idx] = label

    def get_training_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get all labeled data for training."""
        indices = list(self.labeled_indices)
        crops = self.all_crops[indices]
        labels = torch.tensor([self.labels[i] for i in indices]).float().unsqueeze(1)
        return crops, labels

    def retrain_model(self):
        """Retrain model on current labels + pseudo-labels."""
        crops, labels = self.get_training_data()

        # Add pseudo-labels if we have enough real labels
        if len(crops) >= 10:
            unlabeled_mask = torch.tensor([
                i not in self.labeled_indices
                for i in range(len(self.all_crops))
            ])
            unlabeled_crops = self.all_crops[unlabeled_mask]

            pos_mask, neg_mask, _ = self.pseudo_labeler.generate_pseudo_labels(
                self.model, unlabeled_crops, self.device
            )

            pseudo_crops = torch.cat([
                unlabeled_crops[pos_mask],
                unlabeled_crops[neg_mask],
            ])
            pseudo_labels = torch.cat([
                torch.ones(pos_mask.sum()),
                torch.zeros(neg_mask.sum()),
            ])

            crops, labels = expand_with_pseudo_labels(
                crops, labels, pseudo_crops, pseudo_labels
            )

        # Train model...
        # (use training code from doc 06 or 07)

        self.round += 1
        self.pseudo_labeler.step()  # Decay threshold
```

---

## Expected Label Efficiency

| Real Labels | Pseudo Labels | Effective Dataset | Expected Accuracy |
|-------------|---------------|-------------------|-------------------|
| 25 | ~50 | ~75 | 75-80% |
| 50 | ~100 | ~150 | 80-85% |
| 100 | ~200 | ~300 | 85-90% |

Pseudo-labeling typically doubles effective dataset size with careful thresholding.
