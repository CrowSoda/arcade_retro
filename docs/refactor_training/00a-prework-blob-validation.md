# Pre-Work: Validate Blob Detection (DO FIRST)

## Why This Matters

Before building anything on top of blob detection, we must verify it can actually find your signals. If blob detection misses 30% of signals, no amount of classifier training will recover them.

**This is a BLOCKING prerequisite.** Do not proceed to Week 1 until recall ≥ 90%.

---

## The Test

### Input
- Exhaustively-labeled spectrogram (all signals marked)
- Labels file with ground truth bounding boxes

### Output
- Recall percentage: what % of labeled signals does blob detection find?

### Decision Tree

```
Blob Recall Test
      │
      ▼
  recall >= 90%? ──YES──► Proceed with plan
      │
      NO
      │
      ▼
  Tune blob detector:
    - Lower min_area
    - Lower intensity threshold
    - Use "high sensitivity" preset
      │
      ▼
  Re-test
      │
      ▼
  recall >= 90%? ──YES──► Proceed with plan
      │
      NO
      │
      ▼
  STOP: Blob detection is not viable
  Consider alternative proposal methods
```

---

## Deliverable: `scripts/test_blob_recall.py`

```python
#!/usr/bin/env python3
"""
Blob Detection Recall Test

Validates that blob detection can find labeled signals before
building the crop classifier on top of it.

Usage:
    python scripts/test_blob_recall.py \
        --spectrogram path/to/spectrogram.npz \
        --labels path/to/labels.json \
        --iou-threshold 0.3 \
        --sensitivity high
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_spectrogram(path: str) -> np.ndarray:
    """Load spectrogram from file."""
    if path.endswith('.npz'):
        data = np.load(path)
        return data['spectrogram']
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float32) / 255.0


def load_labels(path: str) -> list[dict]:
    """
    Load ground truth labels.

    Expected format:
    [
        {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 40},
        ...
    ]
    """
    with open(path) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict) and 'boxes' in data:
        return data['boxes']
    return data


def compute_iou(box1: dict, box2: dict) -> float:
    """Compute Intersection over Union between two boxes."""
    # Intersection
    x1 = max(box1['x_min'], box2['x_min'])
    y1 = max(box1['y_min'], box2['y_min'])
    x2 = min(box1['x_max'], box2['x_max'])
    y2 = min(box1['y_max'], box2['y_max'])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union
    area1 = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    area2 = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def run_blob_detection(
    spectrogram: np.ndarray,
    sensitivity: str = 'medium'
) -> list[dict]:
    """
    Run blob detection on spectrogram.

    Args:
        spectrogram: (H, W) grayscale image
        sensitivity: 'low', 'medium', or 'high'

    Returns:
        List of bounding box dicts
    """
    import cv2

    # Sensitivity presets
    PRESETS = {
        'low': {'min_area': 100, 'otsu_mult': 1.2},
        'medium': {'min_area': 50, 'otsu_mult': 1.0},
        'high': {'min_area': 30, 'otsu_mult': 0.8},
    }

    params = PRESETS.get(sensitivity, PRESETS['medium'])

    # Normalize to 0-255 uint8
    if spectrogram.max() <= 1.0:
        img = (spectrogram * 255).astype(np.uint8)
    else:
        img = spectrogram.astype(np.uint8)

    # Otsu thresholding with sensitivity adjustment
    threshold, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adjust threshold based on sensitivity
    adjusted_thresh = int(threshold * params['otsu_mult'])
    _, binary = cv2.threshold(img, adjusted_thresh, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Extract bounding boxes
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params['min_area']:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append({
            'x_min': x,
            'y_min': y,
            'x_max': x + w,
            'y_max': y + h,
        })

    return boxes


def match_boxes(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    iou_threshold: float = 0.3,
) -> tuple[list[int], list[int], list[int]]:
    """
    Match predicted boxes to ground truth boxes.

    Returns:
        (matched_gt, matched_pred, unmatched_gt)
    """
    matched_gt = []
    matched_pred = []
    used_pred = set()

    for gt_idx, gt_box in enumerate(gt_boxes):
        best_iou = 0
        best_pred_idx = -1

        for pred_idx, pred_box in enumerate(pred_boxes):
            if pred_idx in used_pred:
                continue

            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_iou >= iou_threshold:
            matched_gt.append(gt_idx)
            matched_pred.append(best_pred_idx)
            used_pred.add(best_pred_idx)

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]

    return matched_gt, matched_pred, unmatched_gt


def analyze_signal_shapes(gt_boxes: list[dict]) -> dict:
    """Analyze aspect ratios of ground truth boxes."""
    if not gt_boxes:
        return {}

    widths = [b['x_max'] - b['x_min'] for b in gt_boxes]
    heights = [b['y_max'] - b['y_min'] for b in gt_boxes]
    aspects = [w / max(h, 1) for w, h in zip(widths, heights)]

    return {
        'count': len(gt_boxes),
        'avg_width': np.mean(widths),
        'avg_height': np.mean(heights),
        'avg_aspect_ratio': np.mean(aspects),
        'min_aspect_ratio': min(aspects),
        'max_aspect_ratio': max(aspects),
        'median_aspect_ratio': np.median(aspects),
        'suggested_crop_size': suggest_crop_size(widths, heights),
    }


def suggest_crop_size(widths: list, heights: list) -> tuple[int, int]:
    """Suggest optimal crop size based on signal dimensions."""
    # Use 80th percentile to cover most signals
    w80 = np.percentile(widths, 80)
    h80 = np.percentile(heights, 80)

    # Add 30% padding (15% each side)
    w_padded = w80 * 1.3
    h_padded = h80 * 1.3

    # Round up to nearest multiple of 16 (GPU friendly)
    def round_up_16(x):
        return int(np.ceil(x / 16) * 16)

    crop_w = max(32, round_up_16(w_padded))
    crop_h = max(32, round_up_16(h_padded))

    # Cap at 128 to prevent huge crops
    crop_w = min(128, crop_w)
    crop_h = min(128, crop_h)

    return (crop_h, crop_w)  # (H, W) format


def main():
    parser = argparse.ArgumentParser(description='Test blob detection recall')
    parser.add_argument('--spectrogram', required=True, help='Path to spectrogram file')
    parser.add_argument('--labels', required=True, help='Path to labels JSON file')
    parser.add_argument('--iou-threshold', type=float, default=0.3, help='IoU threshold for matching')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--output', help='Save detailed results to JSON file')

    args = parser.parse_args()

    # Load data
    print(f"Loading spectrogram from {args.spectrogram}...")
    spectrogram = load_spectrogram(args.spectrogram)
    print(f"  Shape: {spectrogram.shape}")

    print(f"Loading labels from {args.labels}...")
    gt_boxes = load_labels(args.labels)
    print(f"  Ground truth signals: {len(gt_boxes)}")

    # Analyze signal shapes
    print("\n=== Signal Shape Analysis ===")
    shape_stats = analyze_signal_shapes(gt_boxes)
    print(f"  Average size: {shape_stats['avg_width']:.0f} x {shape_stats['avg_height']:.0f}")
    print(f"  Aspect ratios: {shape_stats['min_aspect_ratio']:.2f} to {shape_stats['max_aspect_ratio']:.2f}")
    print(f"  Suggested crop size: {shape_stats['suggested_crop_size']}")

    # Run blob detection
    print(f"\n=== Blob Detection (sensitivity={args.sensitivity}) ===")
    pred_boxes = run_blob_detection(spectrogram, args.sensitivity)
    print(f"  Detected blobs: {len(pred_boxes)}")

    # Match boxes
    print(f"\n=== Recall Test (IoU >= {args.iou_threshold}) ===")
    matched_gt, matched_pred, unmatched_gt = match_boxes(
        gt_boxes, pred_boxes, args.iou_threshold
    )

    recall = len(matched_gt) / len(gt_boxes) if gt_boxes else 0
    false_positives = len(pred_boxes) - len(matched_pred)

    print(f"  Matched: {len(matched_gt)}/{len(gt_boxes)}")
    print(f"  Recall: {recall:.1%}")
    print(f"  False positives: {false_positives}")

    # Verdict
    print("\n" + "="*50)
    if recall >= 0.9:
        print("✅ PASS: Recall >= 90%")
        print("   Proceed with crop classifier implementation.")
    elif recall >= 0.8:
        print("⚠️  WARNING: Recall between 80-90%")
        print("   Consider tuning blob detector before proceeding.")
        print("   Try: --sensitivity high")
    else:
        print("❌ FAIL: Recall < 80%")
        print("   Blob detection cannot find enough signals.")
        print("   Options:")
        print("   1. Tune blob detector (lower thresholds)")
        print("   2. Use different proposal method")
        print("   3. Re-evaluate approach")
    print("="*50)

    # Show missed signals
    if unmatched_gt:
        print(f"\nMissed signals ({len(unmatched_gt)}):")
        for idx in unmatched_gt[:5]:  # Show first 5
            box = gt_boxes[idx]
            print(f"  Box {idx}: ({box['x_min']}, {box['y_min']}) to ({box['x_max']}, {box['y_max']})")
        if len(unmatched_gt) > 5:
            print(f"  ... and {len(unmatched_gt) - 5} more")

    # Save detailed results
    if args.output:
        results = {
            'spectrogram_path': args.spectrogram,
            'labels_path': args.labels,
            'sensitivity': args.sensitivity,
            'iou_threshold': args.iou_threshold,
            'ground_truth_count': len(gt_boxes),
            'detected_count': len(pred_boxes),
            'matched_count': len(matched_gt),
            'recall': recall,
            'false_positives': false_positives,
            'shape_analysis': shape_stats,
            'unmatched_indices': unmatched_gt,
            'verdict': 'PASS' if recall >= 0.9 else 'WARNING' if recall >= 0.8 else 'FAIL',
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.output}")

    return 0 if recall >= 0.9 else 1


if __name__ == '__main__':
    exit(main())
```

---

## Running the Test

### Step 1: Prepare Test Data

Create an exhaustively-labeled spectrogram:

```bash
# Use existing labeled data or create new
ls training_data/signals/*/samples/*.json
```

### Step 2: Run Test

```bash
python scripts/test_blob_recall.py \
    --spectrogram training_data/signals/my_signal/samples/abc123.npz \
    --labels training_data/signals/my_signal/samples/abc123.json \
    --sensitivity medium \
    --output recall_test_results.json
```

### Step 3: Interpret Results

**PASS (recall ≥ 90%):**
```
✅ PASS: Recall >= 90%
   Proceed with crop classifier implementation.
```

**WARNING (80-90%):**
```
⚠️  WARNING: Recall between 80-90%
   Try: --sensitivity high
```

Re-run with high sensitivity and see if recall improves.

**FAIL (<80%):**
```
❌ FAIL: Recall < 80%
   Blob detection cannot find enough signals.
```

Do NOT proceed until this is resolved.

---

## Tuning Blob Detection

If recall is insufficient, tune parameters:

### Option 1: Increase Sensitivity

```bash
python scripts/test_blob_recall.py \
    --spectrogram ... \
    --labels ... \
    --sensitivity high  # Lower thresholds, more detections
```

### Option 2: Modify Blob Detector Code

In `unified_pipeline.py` or new `blob_detector.py`:

```python
BLOB_SENSITIVITY_PRESETS = {
    'low': {
        'min_area': 100,
        'otsu_multiplier': 1.2,  # Higher threshold = fewer detections
    },
    'medium': {
        'min_area': 50,
        'otsu_multiplier': 1.0,
    },
    'high': {
        'min_area': 30,
        'otsu_multiplier': 0.8,  # Lower threshold = more detections
    },
    'max': {
        'min_area': 20,
        'otsu_multiplier': 0.6,  # For difficult signals
    },
}
```

### Option 3: Alternative Proposal Methods

If Otsu fails completely, consider:
- Adaptive thresholding
- Peak detection in frequency bins
- Sliding window energy detection
- Pre-trained region proposal network

---

## Bonus Output: Signal Shape Analysis

The script also analyzes your signals' aspect ratios:

```
=== Signal Shape Analysis ===
  Average size: 45 x 12
  Aspect ratios: 2.1 to 8.5
  Suggested crop size: (32, 96)
```

Use this to configure crop dimensions in `crop_classifier.yaml`.
