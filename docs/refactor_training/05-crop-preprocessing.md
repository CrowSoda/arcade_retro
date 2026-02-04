# Crop Preprocessing Pipeline

## Overview

Transform detected bounding boxes into normalized tensors suitable for classification.

---

## ⚠️ Configurable Crop Dimensions

**Do NOT hardcode 64×64.** Analyze your signals first using the pre-work script:

```bash
python scripts/test_blob_recall.py --spectrogram ... --labels ... --output shapes.json
# Output includes suggested crop size based on signal aspect ratios
```

### Choosing Crop Size

| Signal Profile | Suggested Crop | Rationale |
|----------------|----------------|-----------|
| Square-ish (aspect 0.5-2) | 64×64 | Default balanced |
| Wide signals (aspect 2-5) | 48×96 or 32×96 | Minimize padding |
| Very wide (aspect 5+) | 32×128 | Accommodate spread |

**Goal:** 80% of signals fit with <50% padding waste.

```
Raw Spectrogram + BBox
         │
         ▼
┌─────────────────┐
│ Extract Region  │  Add 15% padding
│ with Padding    │  Clamp to image bounds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aspect-Preserving│  Letterbox to 64×64
│ Resize          │  Zero-pad shorter dimension
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Per-Crop        │  Mean=0, Std=1
│ Normalization   │  Per-crop, not global
└────────┬────────┘
         │
         ▼
    Tensor (1, 64, 64)
```

---

## Why Aspect Ratio Preservation Matters

RF signals have characteristic aspect ratios:
- **Wide signals** (LTE, WiFi): 5:1 to 10:1
- **Narrow signals** (CW, beacons): 1:5 to 1:10
- **Square-ish** (bursts): 1:1 to 2:1

Stretching destroys frequency/time characteristics that define signal identity.

```
Original (100×20):           Stretched to 64×64:
┌──────────────────┐         ┌────────────────┐
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│    →    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
└──────────────────┘         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← Distorted!
                             └────────────────┘

Letterboxed to 64×64:
┌────────────────┐
│░░░░░░░░░░░░░░░░│  ← Zero padding
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← Signal preserved
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│░░░░░░░░░░░░░░░░│  ← Zero padding
└────────────────┘
```

---

## Full Implementation

```python
import cv2
import numpy as np
import torch


def preprocess_crop(
    image: np.ndarray,
    bbox: dict,
    target_size: tuple[int, int] = (64, 64),
    padding_pct: float = 0.15,
) -> torch.Tensor:
    """
    Extract and preprocess a crop for classification.

    Args:
        image: Source spectrogram (H, W) grayscale float32
        bbox: Dict with x_min, y_min, x_max, y_max
        target_size: Output size (height, width)
        padding_pct: Padding around bbox as fraction of box size

    Returns:
        Tensor of shape (1, target_h, target_w), normalized
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # --- Step 1: Add padding ---
    box_w = bbox['x_max'] - bbox['x_min']
    box_h = bbox['y_max'] - bbox['y_min']

    pad_x = int(box_w * padding_pct)
    pad_y = int(box_h * padding_pct)

    x1 = max(0, bbox['x_min'] - pad_x)
    y1 = max(0, bbox['y_min'] - pad_y)
    x2 = min(w, bbox['x_max'] + pad_x)
    y2 = min(h, bbox['y_max'] + pad_y)

    # --- Step 2: Extract crop ---
    crop = image[y1:y2, x1:x2].copy()

    if crop.size == 0:
        # Fallback for degenerate boxes
        return torch.zeros(1, target_h, target_w, dtype=torch.float32)

    # --- Step 3: Aspect-preserving resize (letterbox) ---
    crop_h, crop_w = crop.shape[:2]

    # Calculate scale to fit in target while preserving aspect
    scale = min(target_h / crop_h, target_w / crop_w)
    new_h = int(crop_h * scale)
    new_w = int(crop_w * scale)

    # Resize with bilinear interpolation
    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Create zero-padded output (letterbox)
    result = np.zeros((target_h, target_w), dtype=np.float32)

    # Center the resized crop
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # --- Step 4: Per-crop normalization ---
    mean = result.mean()
    std = result.std()

    if std > 1e-8:
        result = (result - mean) / std
    else:
        # Handle constant crops (rare edge case)
        result = result - mean

    # --- Step 5: Convert to tensor ---
    tensor = torch.from_numpy(result).unsqueeze(0)  # (1, H, W)

    return tensor


def preprocess_batch(
    image: np.ndarray,
    bboxes: list[dict],
    target_size: tuple[int, int] = (64, 64),
    padding_pct: float = 0.15,
) -> torch.Tensor:
    """
    Preprocess multiple crops efficiently.

    Args:
        image: Source spectrogram
        bboxes: List of bbox dicts
        target_size: Output size
        padding_pct: Padding fraction

    Returns:
        Tensor of shape (N, 1, H, W)
    """
    crops = [
        preprocess_crop(image, bbox, target_size, padding_pct)
        for bbox in bboxes
    ]

    if not crops:
        return torch.empty(0, 1, target_size[0], target_size[1])

    return torch.stack(crops)
```

---

## Configuration

```python
# In config/crop_preprocessing.py

CROP_CONFIG = {
    # Target size for all crops
    'target_size': (64, 64),

    # Padding around detected bbox
    'padding_pct': 0.15,

    # Normalization
    'normalize': 'per_crop',  # Options: 'per_crop', 'global', 'none'

    # Interpolation for resize
    'interpolation': cv2.INTER_LINEAR,

    # Minimum valid crop size (before resize)
    'min_crop_size': 4,
}
```

---

## Edge Cases

### 1. Crop at Image Boundary

When padding extends beyond image bounds, clamp coordinates:

```python
x1 = max(0, bbox['x_min'] - pad_x)  # Clamp left
x2 = min(w, bbox['x_max'] + pad_x)  # Clamp right
```

### 2. Very Small Crops

Crops smaller than 4×4 pixels produce poor features. Options:
- Skip classification (treat as noise)
- Enforce minimum size in letterbox

### 3. Very Large Crops

Crops larger than 256×256 indicate multiple merged signals:
- Consider splitting via watershed
- Or accept as single detection

### 4. Constant-Value Crops

If crop has zero variance (all same value):
- Set normalized output to zeros
- Will likely classify as background

---

## Performance Considerations

### Vectorized Batch Processing

For maximum throughput, use batch resize:

```python
import cv2

def batch_resize_letterbox(crops: list[np.ndarray], target_size: tuple) -> np.ndarray:
    """
    Resize multiple crops with potential for parallelization.
    """
    # OpenCV's resize is already highly optimized
    # For GPU acceleration, consider kornia or DALI
    results = []
    for crop in crops:
        result = letterbox_single(crop, target_size)
        results.append(result)
    return np.stack(results)
```

### GPU Preprocessing (Optional)

For very high throughput, move preprocessing to GPU:

```python
import kornia
import torch

def gpu_preprocess(image_tensor: torch.Tensor, bboxes: list[dict]) -> torch.Tensor:
    """
    GPU-accelerated crop extraction using kornia.

    ~10x faster for large batch sizes.
    """
    # Convert bboxes to tensor format
    boxes = torch.tensor([
        [b['x_min'], b['y_min'], b['x_max'], b['y_max']]
        for b in bboxes
    ], dtype=torch.float32, device=image_tensor.device)

    # Kornia's crop_and_resize handles everything
    crops = kornia.geometry.transform.crop_and_resize(
        image_tensor.unsqueeze(0),  # (1, 1, H, W)
        boxes.unsqueeze(0),          # (1, N, 4)
        (64, 64),
    )

    return crops.squeeze(0)  # (N, 1, 64, 64)
```
