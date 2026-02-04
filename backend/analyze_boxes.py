"""
Box Shape & Anchor Coverage Analysis

Run from g20_demo/backend:
    python analyze_boxes.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import box_iou

# ============================================================
# 1. LOAD ALL TRAINING BOXES
# ============================================================

TRAINING_DATA_DIR = Path(__file__).parent.parent / "training_data" / "signals"

all_boxes = []
box_metadata = []

print("=== LOADING TRAINING DATA ===")
for signal_dir in TRAINING_DATA_DIR.iterdir():
    if not signal_dir.is_dir():
        continue

    samples_dir = signal_dir / "samples"
    if not samples_dir.exists():
        continue

    signal_name = signal_dir.name
    sample_count = 0

    for json_file in samples_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        for box in data.get("boxes", []):
            x_min = box.get("x_min", 0)
            y_min = box.get("y_min", 0)
            x_max = box.get("x_max", 1)
            y_max = box.get("y_max", 1)

            # Ensure valid box
            if x_max > x_min and y_max > y_min:
                all_boxes.append([x_min, y_min, x_max, y_max])
                box_metadata.append(
                    {
                        "signal": signal_name,
                        "sample_id": data.get("sample_id"),
                    }
                )
                sample_count += 1

    if sample_count > 0:
        print(f"  {signal_name}: {sample_count} boxes")

print(f"\nTotal boxes loaded: {len(all_boxes)}")

if len(all_boxes) == 0:
    print("ERROR: No training boxes found! Check training_data/signals/*/samples/*.json")
    exit(1)

# ============================================================
# 2. BOX SHAPE ANALYSIS
# ============================================================

widths = [box[2] - box[0] for box in all_boxes]
heights = [box[3] - box[1] for box in all_boxes]
aspects = [h / w if w > 0 else 0 for h, w in zip(heights, widths, strict=False)]
areas = [w * h for w, h in zip(widths, heights, strict=False)]

print("\n" + "=" * 50)
print("=== BOX SHAPE ANALYSIS ===")
print("=" * 50)
print(f"Total boxes: {len(all_boxes)}")
print(
    f"Width  - min: {min(widths):.1f}, max: {max(widths):.1f}, mean: {np.mean(widths):.1f}, std: {np.std(widths):.1f}"
)
print(
    f"Height - min: {min(heights):.1f}, max: {max(heights):.1f}, mean: {np.mean(heights):.1f}, std: {np.std(heights):.1f}"
)
print(
    f"Aspect - min: {min(aspects):.2f}, max: {max(aspects):.2f}, mean: {np.mean(aspects):.2f}, std: {np.std(aspects):.2f}"
)
print(f"Area   - min: {min(areas):.0f}, max: {max(areas):.0f}, mean: {np.mean(areas):.0f}")
print(
    f"Aspect ratio percentiles: 10%={np.percentile(aspects, 10):.2f}, 50%={np.percentile(aspects, 50):.2f}, 90%={np.percentile(aspects, 90):.2f}"
)

# Histogram of aspects
print("\nAspect ratio distribution:")
bins = [0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
hist, _ = np.histogram(aspects, bins=bins)
for i in range(len(bins) - 1):
    pct = 100 * hist[i] / len(aspects)
    print(f"  {bins[i]:.2f} - {bins[i+1]:.2f}: {hist[i]:4d} ({pct:5.1f}%)")

# ============================================================
# 3. CURRENT ANCHOR CONFIG (from Faster R-CNN defaults + Hydra)
# ============================================================

print("\n" + "=" * 50)
print("=== CURRENT ANCHOR CONFIG ===")
print("=" * 50)

# Default Faster R-CNN anchor config (check hydra/detector.py for any custom)
# These are torchvision defaults for Faster R-CNN
default_sizes = ((32,), (64,), (128,), (256,), (512,))
default_aspect_ratios = ((0.5, 1.0, 2.0),) * 5

print(f"Sizes (per feature level): {default_sizes}")
print(f"Aspect ratios: {default_aspect_ratios[0]}")
print(f"Total anchor templates: {len(default_sizes) * len(default_aspect_ratios[0])}")

# ============================================================
# 4. ANCHOR COVERAGE TEST
# ============================================================

print("\n" + "=" * 50)
print("=== ANCHOR COVERAGE TEST ===")
print("=" * 50)

# Generate anchors for 1024x1024 image
# Faster R-CNN FPN uses 5 feature levels with strides 4, 8, 16, 32, 64
IMAGE_SIZE = 1024
STRIDES = [4, 8, 16, 32, 64]

# Generate all anchors
all_anchors = []
for level_idx, (stride, sizes, ratios) in enumerate(
    zip(STRIDES, default_sizes, default_aspect_ratios, strict=False)
):
    feat_size = IMAGE_SIZE // stride

    for y in range(feat_size):
        for x in range(feat_size):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride

            for size in sizes:
                for ratio in ratios:
                    # ratio = h/w
                    w = size / np.sqrt(ratio)
                    h = size * np.sqrt(ratio)

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    all_anchors.append([x1, y1, x2, y2])

print(f"Total anchors generated: {len(all_anchors)}")

# Test coverage
anchor_tensor = torch.tensor(all_anchors, dtype=torch.float32)
all_max_ious = []

print("Testing coverage (this may take a moment)...")
for i, gt_box in enumerate(all_boxes):
    gt_tensor = torch.tensor([gt_box], dtype=torch.float32)
    ious = box_iou(anchor_tensor, gt_tensor)
    max_iou = ious.max().item()
    all_max_ious.append(max_iou)

covered_50 = sum(iou > 0.5 for iou in all_max_ious)
covered_30 = sum(iou > 0.3 for iou in all_max_ious)

print(
    f"\nBoxes with max IoU > 0.5: {covered_50}/{len(all_max_ious)} ({100*covered_50/len(all_max_ious):.1f}%)"
)
print(
    f"Boxes with max IoU > 0.3: {covered_30}/{len(all_max_ious)} ({100*covered_30/len(all_max_ious):.1f}%)"
)
print(f"Worst coverage (min max IoU): {min(all_max_ious):.3f}")
print(
    f"IoU percentiles: 10%={np.percentile(all_max_ious, 10):.2f}, 50%={np.percentile(all_max_ious, 50):.2f}"
)

# Show worst boxes
print("\nWorst covered boxes:")
sorted_indices = np.argsort(all_max_ious)[:5]
for idx in sorted_indices:
    box = all_boxes[idx]
    w = box[2] - box[0]
    h = box[3] - box[1]
    aspect = h / w if w > 0 else 0
    meta = box_metadata[idx]
    print(
        f"  {meta['signal']} - IoU={all_max_ious[idx]:.3f}, size={w:.0f}x{h:.0f}, aspect={aspect:.2f}"
    )

# ============================================================
# 5. IMAGE CONFIG
# ============================================================

print("\n" + "=" * 50)
print("=== IMAGE CONFIG ===")
print("=" * 50)
print(f"Spectrogram size: {IMAGE_SIZE} x {IMAGE_SIZE}")
print("FFT size (inference): 4096")
print("Hop size (inference): 2048")
print("Dynamic range: 80 dB")

# ============================================================
# 6. DYNAMIC ANCHOR COMPUTATION (IoU-based k-means)
# ============================================================

print("\n" + "=" * 50)
print("=== DYNAMIC ANCHOR COMPUTATION ===")
print("=" * 50)

# Import the new anchor module
from training.anchors import (
    anchors_to_generator_format,
    compute_anchor_coverage,
    compute_anchors_kmeans_iou,
)

# Compute optimal anchors using IoU-based k-means
dynamic_anchors = compute_anchors_kmeans_iou(all_boxes, num_anchors=9)
print(f"\nK-means anchors (w, h): {dynamic_anchors}")

# Convert to AnchorGenerator format
sizes, aspects = anchors_to_generator_format(dynamic_anchors)
print("\nAnchorGenerator format:")
print(f"  sizes: {sizes[0]}")
print(f"  aspects: {aspects[0]}")

# Test coverage with dynamic anchors
dynamic_coverage = compute_anchor_coverage(all_boxes, dynamic_anchors)
print("\n=== COVERAGE COMPARISON ===")
print(f"OLD (fixed anchors):    {100*covered_50/len(all_max_ious):.1f}% at IoU>0.5")
print(f"NEW (dynamic anchors):  {dynamic_coverage['coverage_pct_0.5']:.1f}% at IoU>0.5")
print(
    f"Improvement: +{dynamic_coverage['coverage_pct_0.5'] - 100*covered_50/len(all_max_ious):.1f}%"
)

if dynamic_coverage["coverage_pct_0.5"] > 90:
    print("\n✅ Dynamic anchors provide good coverage (>90%)")
else:
    print("\n⚠️  Coverage still low - may need more anchor templates or FCOS")
