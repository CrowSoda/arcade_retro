# Stage 2: Crop Classifier Models

## Model Selection by Label Count

| Labels | Model | Parameters | Rationale |
|--------|-------|------------|-----------|
| 25-100 | Siamese Network | ~200K | Learn similarity, not boundaries |
| 100-500 | Shallow CNN | ~100K | Direct classification viable |
| 500+ | Deeper CNN | ~500K | Can leverage more data |

**ResNet18 is overkill** - 11.7M parameters designed for 224×224 images. Your 64×64 crops would be reduced to 1×0 pixels after standard pooling layers.

---

## Siamese Network (25-100 labels)

### Why Siamese for Few-Shot

Siamese networks learn **similarity metrics** rather than decision boundaries:
- Achieves **69-78% accuracy with 5-20 examples per class**
- Standard CNNs fail to converge with this little data
- Learns "is this similar to known signals?" not "is this class A or B?"

### Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """
    Encodes a crop into a 64-dimensional embedding.

    Two identical encoders process anchor and comparison images,
    then distance between embeddings determines similarity.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global pooling: 8x8 → 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # Embedding projection
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized embedding."""
        embedding = self.encoder(x)
        return F.normalize(embedding, p=2, dim=1)


class SiameseNetwork(nn.Module):
    """
    Full Siamese network for similarity learning.

    Usage:
        model = SiameseNetwork()
        emb1 = model.encode(crop1)
        emb2 = model.encode(crop2)
        similarity = model.similarity(emb1, emb2)  # 0-1 score
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.encoder = SiameseEncoder(embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single image or batch."""
        return self.encoder(x)

    def forward(self, anchor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between anchor and other.

        Returns:
            Tensor of shape (batch,) with values 0-1
            Higher = more similar
        """
        emb_anchor = self.encode(anchor)
        emb_other = self.encode(other)

        # Euclidean distance → similarity
        distance = F.pairwise_distance(emb_anchor, emb_other)
        similarity = 1 / (1 + distance)  # Convert to 0-1 range

        return similarity

    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute similarity from pre-computed embeddings."""
        distance = F.pairwise_distance(emb1, emb2)
        return 1 / (1 + distance)
```

### Inference with Siamese

```python
class SiameseClassifier:
    """
    Use trained Siamese network for binary classification.

    Compares each crop against a gallery of known signal embeddings.
    """

    def __init__(self, model: SiameseNetwork, gallery_embeddings: torch.Tensor):
        self.model = model
        self.gallery = gallery_embeddings  # (N, embedding_dim)
        self.threshold = 0.5

    def classify(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Classify batch of crops.

        Returns:
            Confidence scores (batch,) - max similarity to any gallery item
        """
        # Encode all crops
        crop_embeddings = self.model.encode(crops)  # (B, 64)

        # Compare to each gallery item
        # crop_embeddings: (B, 64), gallery: (N, 64)
        # Want: (B, N) similarities

        similarities = []
        for gallery_emb in self.gallery:
            sim = self.model.similarity(
                crop_embeddings,
                gallery_emb.unsqueeze(0).expand(len(crops), -1)
            )
            similarities.append(sim)

        # Max similarity across gallery
        all_sims = torch.stack(similarities, dim=1)  # (B, N)
        max_sim, _ = all_sims.max(dim=1)  # (B,)

        return max_sim
```

---

## Direct CNN Classifier (100+ labels)

### Architecture

```python
class SignalClassifier(nn.Module):
    """
    Direct binary classifier for signal detection.

    Use when you have 100+ labeled examples.
    Simpler than Siamese, faster inference.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16x16 → Global
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global Average Pooling - CRITICAL for position invariance
            # Research shows position info concentrates channel-wise
            # GAP mitigates this by averaging spatially
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),  # Binary output (logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 1, 64, 64) grayscale crops

        Returns:
            (B, 1) logits (apply sigmoid for probability)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability scores."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
```

### Key Design Choices

1. **Global Average Pooling (GAP)**
   - Reduces position encoding
   - Position info concentrates channel-wise
   - GAP averages spatially, mitigating bias

2. **3 Conv Layers Only**
   - Prevents overfitting with <500 samples
   - Keeps inference under 1ms
   - Sufficient for 64×64 crops

3. **High Dropout (0.5)**
   - Critical regularization for small datasets
   - Applied before final classification layer

4. **BatchNorm After Every Conv**
   - Stabilizes training with small batches
   - Enables higher learning rates

---

## Model Comparison

| Aspect | Siamese | Direct CNN |
|--------|---------|------------|
| Min labels | 25 | 100 |
| Inference | Compare to gallery | Single forward pass |
| Speed (batch 32) | ~15ms | ~8ms |
| Parameters | ~200K | ~100K |
| Training | Contrastive loss | BCE/Focal loss |
| Extensibility | Add to gallery | Retrain |
