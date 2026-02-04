# Stage 1: Blob Detection

## Purpose

Find all potential signal regions in the spectrogram using **classical (non-learned) methods**.

Key property: **No position bias** because no neural network is involved.

---

## Current Implementation

The existing blob detection in `unified_pipeline.py` uses:

1. **Otsu's Thresholding** - Automatic threshold selection
2. **Region Growing** - Connected component analysis
3. **Bounding Box Extraction** - Minimum enclosing rectangles

This pipeline remains **unchanged** - it already provides position-unbiased proposals.

---

## Required Modification: Padding Margin

### Why 15% Padding

The classifier needs **context** around the signal to distinguish it from noise. Edge information between signal and noise floor is discriminative.

```
Without padding:              With 15% padding:
┌─────────────┐              ┌─────────────────┐
│ Signal only │              │   ┌─────────┐   │
│             │              │   │ Signal  │   │
│             │              │   │         │   │
└─────────────┘              │   └─────────┘   │
                             └─────────────────┘
                                    ↑
                              Context visible
```

### Implementation

```python
def add_padding(bbox: dict, image_shape: tuple, padding_pct: float = 0.15) -> dict:
    """
    Add padding around bounding box.

    Args:
        bbox: Dict with x_min, y_min, x_max, y_max
        image_shape: (height, width) of source image
        padding_pct: Padding as fraction of box size (0.15 = 15%)

    Returns:
        Padded bbox dict, clamped to image bounds
    """
    h, w = image_shape

    # Calculate padding in pixels
    box_w = bbox['x_max'] - bbox['x_min']
    box_h = bbox['y_max'] - bbox['y_min']

    pad_x = int(box_w * padding_pct)
    pad_y = int(box_h * padding_pct)

    # Apply padding with clamping
    return {
        'x_min': max(0, bbox['x_min'] - pad_x),
        'y_min': max(0, bbox['y_min'] - pad_y),
        'x_max': min(w, bbox['x_max'] + pad_x),
        'y_max': min(h, bbox['y_max'] + pad_y),
    }
```

---

## Expected Output

For a typical 1024×1024 spectrogram with active signals:

| Metric | Expected Value |
|--------|----------------|
| Candidate boxes | 50-500 per frame |
| True positives | 5-20 (actual signals) |
| False positives | 30-480 (noise, artifacts) |
| Miss rate | <5% (blob detection is sensitive) |

The classifier's job is to reject the false positives.

---

## Configuration Parameters

```python
# In blob_detector.py or config

BLOB_DETECTION_CONFIG = {
    # Otsu parameters (unchanged)
    'use_otsu': True,

    # Region growing parameters (unchanged)
    'min_area': 50,      # Minimum blob area in pixels
    'max_area': 50000,   # Maximum blob area in pixels

    # NEW: Padding for classifier
    'padding_pct': 0.15,  # 15% padding around each box

    # Optional: Aspect ratio filtering
    'min_aspect_ratio': 0.1,   # Very wide signals OK
    'max_aspect_ratio': 10.0,  # Very tall signals OK
}
```

---

## Integration Point

The blob detector output feeds directly into crop extraction:

```python
# In inference pipeline

def process_frame(spectrogram: np.ndarray) -> List[Detection]:
    # Stage 1: Blob detection (existing code)
    raw_bboxes = blob_detector.detect(spectrogram)

    # Add padding to each box (NEW)
    padded_bboxes = [
        add_padding(bbox, spectrogram.shape, padding_pct=0.15)
        for bbox in raw_bboxes
    ]

    # Continue to crop extraction...
    return padded_bboxes
```

---

## Why Not Modify Blob Detection

The current blob detection is intentionally simple and conservative:

1. **High recall** - Catches almost all signals
2. **Position invariant** - No learned components
3. **Fast** - 5-10ms CPU time
4. **Deterministic** - Same input = same output

Adding learned components here would reintroduce position bias. Keep it classical.
