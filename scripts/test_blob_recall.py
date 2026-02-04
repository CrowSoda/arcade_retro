#!/usr/bin/env python3
"""
Blob Detection Recall Test

Validates that blob detection can find labeled signals before
building the crop classifier on top of it.

Usage:
    # Single file:
    python scripts/test_blob_recall.py \
        --spectrogram path/to/spectrogram.npz \
        --labels path/to/labels.json \
        --sensitivity medium

    # All samples in a directory:
    python scripts/test_blob_recall.py \
        --samples-dir training_data/signals/Creamy_Shrimp/samples \
        --sensitivity medium
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed. Run: pip install opencv-python")
    exit(1)


def load_spectrogram(path: str) -> np.ndarray:
    """Load spectrogram from file."""
    path = str(path)
    if path.endswith(".npz"):
        data = np.load(path)
        # Try common key names
        for key in ["spectrogram", "arr_0", "data", "image"]:
            if key in data.files:
                return data[key]
        # Just use first array
        return data[data.files[0]]
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return img.astype(np.float32) / 255.0


def load_labels(path: str) -> list:
    """
    Load ground truth labels.

    Handles the format from training data:
    {
        "boxes": [{"x_min": ..., "x_max": ..., "y_min": ..., "y_max": ...}, ...]
    }
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "boxes" in data:
        return data["boxes"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown label format in {path}")


def compute_iou(box1: dict, box2: dict) -> float:
    """Compute Intersection over Union between two boxes."""
    x1 = max(box1["x_min"], box2["x_min"])
    y1 = max(box1["y_min"], box2["y_min"])
    x2 = min(box1["x_max"], box2["x_max"])
    y2 = min(box1["y_max"], box2["y_max"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
    area2 = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# Sensitivity presets for adaptive thresholding
# block_size: local neighborhood size (must be odd)
# C: constant subtracted from mean (higher = more sensitive)
# min_area: minimum blob area to consider
PRESETS = {
    "low": {"block_size": 101, "C": 3, "min_area": 100},
    "medium": {"block_size": 101, "C": 5, "min_area": 50},
    "high": {"block_size": 51, "C": 5, "min_area": 30},
    "max": {"block_size": 31, "C": 5, "min_area": 20},
}


def run_blob_detection(spectrogram: np.ndarray, sensitivity: str = "medium") -> list:
    """
    Run blob detection on spectrogram using adaptive thresholding.

    Adaptive thresholding is required because RF spectrograms have
    varying noise floors across frequencies, causing simple Otsu
    to create giant connected components.

    Args:
        spectrogram: (H, W) grayscale image
        sensitivity: 'low', 'medium', 'high', or 'max'

    Returns:
        List of bounding box dicts
    """
    params = PRESETS.get(sensitivity, PRESETS["medium"])

    # Normalize to 0-255 uint8
    if spectrogram.dtype != np.uint8:
        if spectrogram.max() <= 1.0:
            img = (spectrogram * 255).astype(np.uint8)
        else:
            img = (
                (spectrogram - spectrogram.min())
                / (spectrogram.max() - spectrogram.min() + 1e-8)
                * 255
            ).astype(np.uint8)
    else:
        img = spectrogram

    # Adaptive Gaussian thresholding
    # -C (negative) means we threshold above local mean, detecting bright spots
    binary = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params["block_size"],
        -params["C"],  # Negative = detect values ABOVE local mean
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params["min_area"]:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(
            {
                "x_min": x,
                "y_min": y,
                "x_max": x + w,
                "y_max": y + h,
            }
        )

    return boxes


def match_boxes(
    gt_boxes: list,
    pred_boxes: list,
    iou_threshold: float = 0.3,
) -> tuple:
    """Match predicted boxes to ground truth boxes."""
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


def analyze_signal_shapes(gt_boxes: list) -> dict:
    """Analyze aspect ratios of ground truth boxes."""
    if not gt_boxes:
        return {}

    widths = [b["x_max"] - b["x_min"] for b in gt_boxes]
    heights = [b["y_max"] - b["y_min"] for b in gt_boxes]
    aspects = [w / max(h, 1) for w, h in zip(widths, heights, strict=False)]

    # Suggest crop size based on 80th percentile
    w80 = np.percentile(widths, 80)
    h80 = np.percentile(heights, 80)

    # Add 30% padding
    w_padded = w80 * 1.3
    h_padded = h80 * 1.3

    # Round to multiples of 16
    def round_up_16(x):
        return int(np.ceil(x / 16) * 16)

    crop_w = min(128, max(32, round_up_16(w_padded)))
    crop_h = min(128, max(32, round_up_16(h_padded)))

    return {
        "count": len(gt_boxes),
        "avg_width": float(np.mean(widths)),
        "avg_height": float(np.mean(heights)),
        "min_width": int(min(widths)),
        "max_width": int(max(widths)),
        "min_height": int(min(heights)),
        "max_height": int(max(heights)),
        "avg_aspect_ratio": float(np.mean(aspects)),
        "min_aspect_ratio": float(min(aspects)),
        "max_aspect_ratio": float(max(aspects)),
        "suggested_crop_size": (crop_h, crop_w),
    }


def test_single_sample(
    spectrogram_path: str,
    labels_path: str,
    sensitivity: str,
    iou_threshold: float,
) -> dict:
    """Test blob detection on a single sample."""
    spectrogram = load_spectrogram(spectrogram_path)
    gt_boxes = load_labels(labels_path)

    pred_boxes = run_blob_detection(spectrogram, sensitivity)
    matched_gt, matched_pred, unmatched_gt = match_boxes(gt_boxes, pred_boxes, iou_threshold)

    recall = len(matched_gt) / len(gt_boxes) if gt_boxes else 1.0

    return {
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "matched": len(matched_gt),
        "missed": len(unmatched_gt),
        "recall": recall,
        "false_positives": len(pred_boxes) - len(matched_pred),
        "gt_boxes": gt_boxes,
    }


def main():
    parser = argparse.ArgumentParser(description="Test blob detection recall")
    parser.add_argument("--spectrogram", help="Path to single spectrogram file")
    parser.add_argument("--labels", help="Path to single labels JSON file")
    parser.add_argument("--samples-dir", help="Directory with paired .npz and .json files")
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3, help="IoU threshold for matching"
    )
    parser.add_argument("--sensitivity", choices=["low", "medium", "high", "max"], default="medium")
    parser.add_argument("--output", help="Save detailed results to JSON file")

    args = parser.parse_args()

    # Collect samples
    samples = []
    if args.samples_dir:
        samples_dir = Path(args.samples_dir)
        npz_files = sorted(samples_dir.glob("*.npz"))
        for npz_path in npz_files:
            json_path = npz_path.with_suffix(".json")
            if json_path.exists():
                samples.append((str(npz_path), str(json_path)))
    elif args.spectrogram and args.labels:
        samples.append((args.spectrogram, args.labels))
    else:
        parser.error("Either --samples-dir or both --spectrogram and --labels required")

    print(f"Testing {len(samples)} sample(s) with sensitivity={args.sensitivity}")
    print(f"IoU threshold: {args.iou_threshold}")
    print()

    # Run tests
    all_gt_boxes = []
    total_gt = 0
    total_matched = 0
    total_missed = 0
    total_pred = 0
    sample_results = []

    for spec_path, label_path in samples:
        result = test_single_sample(spec_path, label_path, args.sensitivity, args.iou_threshold)
        sample_results.append(
            {"file": Path(spec_path).stem, **{k: v for k, v in result.items() if k != "gt_boxes"}}
        )

        total_gt += result["gt_count"]
        total_matched += result["matched"]
        total_missed += result["missed"]
        total_pred += result["pred_count"]
        all_gt_boxes.extend(result["gt_boxes"])

        if result["recall"] < 1.0:
            print(
                f"  {Path(spec_path).stem}: recall={result['recall']:.1%} ({result['matched']}/{result['gt_count']})"
            )

    overall_recall = total_matched / total_gt if total_gt > 0 else 0
    total_fp = total_pred - total_matched

    # Signal shape analysis
    print()
    print("=" * 60)
    print("SIGNAL SHAPE ANALYSIS")
    print("=" * 60)
    shape_stats = analyze_signal_shapes(all_gt_boxes)
    if shape_stats:
        print(f"  Total signals: {shape_stats['count']}")
        print(f"  Avg size: {shape_stats['avg_width']:.0f}×{shape_stats['avg_height']:.0f} pixels")
        print(f"  Width range: {shape_stats['min_width']} - {shape_stats['max_width']} pixels")
        print(f"  Height range: {shape_stats['min_height']} - {shape_stats['max_height']} pixels")
        print(
            f"  Aspect ratios: {shape_stats['min_aspect_ratio']:.2f} to {shape_stats['max_aspect_ratio']:.2f}"
        )
        print(f"  Suggested crop size: {shape_stats['suggested_crop_size']} (H×W)")

    # Overall results
    print()
    print("=" * 60)
    print("RECALL TEST RESULTS")
    print("=" * 60)
    print(f"  Samples tested: {len(samples)}")
    print(f"  Ground truth signals: {total_gt}")
    print(f"  Detected blobs: {total_pred}")
    print(f"  Matched: {total_matched}")
    print(f"  Missed: {total_missed}")
    print(f"  False positives: {total_fp}")
    print()
    print(f"  RECALL: {overall_recall:.1%}")
    print()

    # Verdict
    if overall_recall >= 0.9:
        print("✅ PASS: Recall >= 90%")
        print("   Proceed with crop classifier implementation.")
        verdict = "PASS"
    elif overall_recall >= 0.8:
        print("⚠️  WARNING: Recall between 80-90%")
        print("   Consider tuning blob detector before proceeding.")
        if args.sensitivity != "max":
            next_sens = {"low": "medium", "medium": "high", "high": "max"}
            print(f"   Try: --sensitivity {next_sens.get(args.sensitivity, 'max')}")
        verdict = "WARNING"
    else:
        print("❌ FAIL: Recall < 80%")
        print("   Blob detection cannot find enough signals.")
        print("   Options:")
        print("   1. Tune blob detector (lower thresholds)")
        print("   2. Use different proposal method")
        print("   3. Re-evaluate approach")
        verdict = "FAIL"
    print("=" * 60)

    # Save detailed results
    if args.output:
        results = {
            "sensitivity": args.sensitivity,
            "iou_threshold": args.iou_threshold,
            "samples_tested": len(samples),
            "total_ground_truth": total_gt,
            "total_detected": total_pred,
            "total_matched": total_matched,
            "total_missed": total_missed,
            "overall_recall": overall_recall,
            "false_positives": total_fp,
            "shape_analysis": shape_stats,
            "sample_results": sample_results,
            "verdict": verdict,
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.output}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    exit(main())
