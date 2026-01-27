#!/usr/bin/env python3
"""
TENSORCADE Baseline Test Suite

This generates ground truth detections from your trained model
and compares them against the G20 demo system to ensure parity.

Usage:
    # Step 1: Generate baseline detections from TENSORCADE model
    python test_detection_baseline.py generate \
        --model models/modern_burst_gap_fold3.pth \
        --iq-dir data/test_iq/ \
        --output baselines/

    # Step 2: Compare G20 detections against baseline
    python test_detection_baseline.py compare \
        --baseline baselines/ \
        --g20-results g20_results/

    # Step 3: Run automated test
    python test_detection_baseline.py test \
        --model models/modern_burst_gap_fold3.pth \
        --iq-dir data/test_iq/
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class BaselineDetection:
    """A detection stored in the baseline."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


@dataclass
class BaselineFrame:
    """A single frame's baseline data."""

    iq_file: str
    chunk_index: int
    sample_rate: float
    chunk_samples: int
    detections: list[BaselineDetection]
    spectrogram_path: str | None = None


def compute_iou(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def match_detections(
    baseline: list[BaselineDetection],
    test: list[BaselineDetection],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Match test detections to baseline and compute metrics.

    Returns:
        {
            'tp': int,  # True positives (matched)
            'fp': int,  # False positives (extra detections)
            'fn': int,  # False negatives (missed)
            'precision': float,
            'recall': float,
            'f1': float,
            'matches': [(baseline_idx, test_idx, iou), ...]
        }
    """
    baseline_matched = [False] * len(baseline)
    test_matched = [False] * len(test)
    matches = []

    # Greedy matching by highest IoU
    for t_idx, t_det in enumerate(test):
        best_iou = 0
        best_b_idx = -1

        for b_idx, b_det in enumerate(baseline):
            if baseline_matched[b_idx]:
                continue
            if b_det.class_id != t_det.class_id:
                continue

            iou = compute_iou(
                (b_det.x1, b_det.y1, b_det.x2, b_det.y2),
                (t_det.x1, t_det.y1, t_det.x2, t_det.y2),
            )

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_b_idx = b_idx

        if best_b_idx >= 0:
            baseline_matched[best_b_idx] = True
            test_matched[t_idx] = True
            matches.append((best_b_idx, t_idx, best_iou))

    tp = len(matches)
    fp = sum(1 for m in test_matched if not m)
    fn = sum(1 for m in baseline_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def generate_baseline(
    model_path: str,
    iq_dir: str,
    output_dir: str,
    sample_rate: float = 20e6,
    chunk_ms: float = 200.0,
    score_threshold: float = 0.5,
    save_spectrograms: bool = True,
):
    """
    Generate baseline detections from TENSORCADE model.
    """
    from tensorcade_wrapper import TensorcadeConfig, TensorcadeWrapper, load_iq_file

    os.makedirs(output_dir, exist_ok=True)
    if save_spectrograms:
        os.makedirs(os.path.join(output_dir, "spectrograms"), exist_ok=True)

    # Load model
    config = TensorcadeConfig(score_threshold=score_threshold)
    wrapper = TensorcadeWrapper(model_path, config)

    # Find IQ files
    iq_dir = Path(iq_dir)
    iq_files = list(iq_dir.glob("*.sigmf-data"))
    if not iq_files:
        print(f"No .sigmf-data files found in {iq_dir}")
        return

    print(f"Found {len(iq_files)} IQ files")

    chunk_samples = int(sample_rate * chunk_ms / 1000)
    all_baselines = []

    for iq_path in iq_files:
        print(f"\nProcessing: {iq_path.name}")

        # Get file size to determine number of chunks
        file_size = iq_path.stat().st_size
        total_samples = file_size // 8  # complex64
        num_chunks = total_samples // chunk_samples

        print(f"  Chunks: {num_chunks}")

        for chunk_idx in range(num_chunks):
            # Load chunk
            iq_data = load_iq_file(str(iq_path), chunk_samples, chunk_idx)

            # Run inference
            detections, display = wrapper.detect(iq_data, frame_id=chunk_idx)

            # Convert to baseline format
            baseline_dets = [
                BaselineDetection(
                    x1=d.box_pixels[0],
                    y1=d.box_pixels[1],
                    x2=d.box_pixels[2],
                    y2=d.box_pixels[3],
                    score=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                )
                for d in detections
            ]

            # Save spectrogram if requested
            spect_path = None
            if save_spectrograms and baseline_dets:
                spect_filename = f"{iq_path.stem}_chunk{chunk_idx:04d}.npy"
                spect_path = os.path.join(output_dir, "spectrograms", spect_filename)
                np.save(spect_path, display)

            frame = BaselineFrame(
                iq_file=iq_path.name,
                chunk_index=chunk_idx,
                sample_rate=sample_rate,
                chunk_samples=chunk_samples,
                detections=baseline_dets,
                spectrogram_path=spect_path,
            )

            all_baselines.append(frame)

            if baseline_dets:
                print(f"    Chunk {chunk_idx}: {len(baseline_dets)} detections")

    # Save baselines
    output_path = os.path.join(output_dir, "baseline_detections.json")

    # Convert to JSON-serializable format
    json_data = []
    for frame in all_baselines:
        frame_dict = {
            "iq_file": frame.iq_file,
            "chunk_index": frame.chunk_index,
            "sample_rate": frame.sample_rate,
            "chunk_samples": frame.chunk_samples,
            "spectrogram_path": frame.spectrogram_path,
            "detections": [asdict(d) for d in frame.detections],
        }
        json_data.append(frame_dict)

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Summary
    total_detections = sum(len(f.detections) for f in all_baselines)
    frames_with_detections = sum(1 for f in all_baselines if f.detections)

    print(f"\n{'='*60}")
    print("Baseline Generation Complete")
    print(f"{'='*60}")
    print(f"Total frames: {len(all_baselines)}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Output: {output_path}")


def compare_results(
    baseline_path: str,
    g20_results_path: str,
    iou_threshold: float = 0.5,
):
    """
    Compare G20 detection results against baseline.
    """
    # Load baseline
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    # Load G20 results
    with open(g20_results_path) as f:
        g20_data = json.load(f)

    # Build lookup for G20 results
    g20_by_key = {}
    for frame in g20_data:
        key = (frame["iq_file"], frame["chunk_index"])
        g20_by_key[key] = [BaselineDetection(**d) for d in frame["detections"]]

    # Compare each frame
    all_results = []
    total_tp = total_fp = total_fn = 0

    for frame in baseline_data:
        key = (frame["iq_file"], frame["chunk_index"])
        baseline_dets = [BaselineDetection(**d) for d in frame["detections"]]
        g20_dets = g20_by_key.get(key, [])

        result = match_detections(baseline_dets, g20_dets, iou_threshold)
        result["iq_file"] = frame["iq_file"]
        result["chunk_index"] = frame["chunk_index"]
        all_results.append(result)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Comparison Results (IoU threshold: {iou_threshold})")
    print(f"{'='*60}")
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    # Check pass/fail
    PASS_THRESHOLD = 0.95  # 95% F1 required
    if f1 >= PASS_THRESHOLD:
        print(f"\n✓ PASS: F1 score {f1:.3f} >= {PASS_THRESHOLD}")
    else:
        print(f"\n✗ FAIL: F1 score {f1:.3f} < {PASS_THRESHOLD}")

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "passed": f1 >= PASS_THRESHOLD,
        "frame_results": all_results,
    }


def run_test(
    model_path: str,
    iq_dir: str,
    sample_rate: float = 20e6,
    chunk_ms: float = 200.0,
    num_chunks: int = 10,
):
    """
    Quick self-test: run model twice, verify consistent output.
    """
    from tensorcade_wrapper import TensorcadeConfig, TensorcadeWrapper, load_iq_file

    config = TensorcadeConfig()
    wrapper = TensorcadeWrapper(model_path, config)

    iq_dir = Path(iq_dir)
    iq_files = list(iq_dir.glob("*.sigmf-data"))
    if not iq_files:
        print(f"No IQ files found in {iq_dir}")
        return False

    chunk_samples = int(sample_rate * chunk_ms / 1000)

    print(f"\nRunning consistency test on {min(num_chunks, len(iq_files))} chunks...")

    all_passed = True
    for i, iq_path in enumerate(iq_files[:num_chunks]):
        iq_data = load_iq_file(str(iq_path), chunk_samples, 0)

        # Run twice
        dets1, spec1 = wrapper.detect(iq_data, frame_id=0)
        dets2, spec2 = wrapper.detect(iq_data, frame_id=0)

        # Compare
        spec_match = np.allclose(spec1, spec2, rtol=1e-5)
        det_match = len(dets1) == len(dets2)

        if det_match and len(dets1) > 0:
            for d1, d2 in zip(dets1, dets2, strict=False):
                if abs(d1.confidence - d2.confidence) > 1e-5:
                    det_match = False
                    break

        if spec_match and det_match:
            print(f"  ✓ {iq_path.name}: consistent ({len(dets1)} detections)")
        else:
            print(f"  ✗ {iq_path.name}: INCONSISTENT!")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All consistency tests PASSED")
    else:
        print("✗ Some consistency tests FAILED")
    print(f"{'='*60}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="TENSORCADE Baseline Test Suite")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate baseline detections")
    gen_parser.add_argument("--model", required=True, help="Path to .pth model")
    gen_parser.add_argument("--iq-dir", required=True, help="Directory with .sigmf-data files")
    gen_parser.add_argument("--output", default="baselines/", help="Output directory")
    gen_parser.add_argument("--sample-rate", type=float, default=20e6)
    gen_parser.add_argument("--chunk-ms", type=float, default=200.0)
    gen_parser.add_argument("--score-threshold", type=float, default=0.5)
    gen_parser.add_argument("--no-spectrograms", action="store_true")

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare G20 results to baseline")
    cmp_parser.add_argument("--baseline", required=True, help="Baseline JSON file")
    cmp_parser.add_argument("--g20-results", required=True, help="G20 results JSON file")
    cmp_parser.add_argument("--iou-threshold", type=float, default=0.5)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run self-consistency test")
    test_parser.add_argument("--model", required=True, help="Path to .pth model")
    test_parser.add_argument("--iq-dir", required=True, help="Directory with .sigmf-data files")
    test_parser.add_argument("--sample-rate", type=float, default=20e6)
    test_parser.add_argument("--num-chunks", type=int, default=10)

    args = parser.parse_args()

    if args.command == "generate":
        generate_baseline(
            model_path=args.model,
            iq_dir=args.iq_dir,
            output_dir=args.output,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            score_threshold=args.score_threshold,
            save_spectrograms=not args.no_spectrograms,
        )
    elif args.command == "compare":
        compare_results(
            baseline_path=args.baseline,
            g20_results_path=args.g20_results,
            iou_threshold=args.iou_threshold,
        )
    elif args.command == "test":
        success = run_test(
            model_path=args.model,
            iq_dir=args.iq_dir,
            sample_rate=args.sample_rate,
            num_chunks=args.num_chunks,
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
