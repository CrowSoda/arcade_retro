"""
Dynamic anchor generation from user's labeled boxes.
Uses IoU-based k-means (YOLO v2 style) for optimal coverage.

Eliminates the anchor mismatch problem where signals with unusual
aspect ratios (very wide or very tall) have no matching anchors.
"""

import json
from pathlib import Path

import numpy as np
from logger_config import get_logger

logger = get_logger("anchors")


def load_all_boxes_for_signal(training_data_dir: str, signal_name: str) -> list[list[float]]:
    """Load all ground truth boxes for a signal from JSON files.

    Returns:
        List of [x1, y1, x2, y2] in pixels (1024x1024 space)
    """
    samples_dir = Path(training_data_dir) / signal_name / "samples"
    boxes = []

    if not samples_dir.exists():
        logger.warning(f"No samples directory for {signal_name}")
        return boxes

    for json_file in samples_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        for box in data.get("boxes", []):
            x_min = box.get("x_min", 0)
            y_min = box.get("y_min", 0)
            x_max = box.get("x_max", 1)
            y_max = box.get("y_max", 1)

            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])

    logger.info(f"Loaded {len(boxes)} boxes for {signal_name}")
    return boxes


def compute_anchors_kmeans_iou(
    boxes: list[list[float]],
    num_anchors: int = 9,
    max_iter: int = 100,
) -> list[tuple[int, int]]:
    """
    Compute optimal anchors using IoU-based k-means (YOLO v2 style).

    Uses 1-IoU as distance metric, which better captures box overlap
    than Euclidean distance on (w, h).

    Args:
        boxes: List of [x1, y1, x2, y2] boxes
        num_anchors: Number of anchor templates to generate
        max_iter: Maximum k-means iterations

    Returns:
        List of (width, height) tuples, sorted by area
    """
    if len(boxes) < num_anchors:
        logger.warning(f"Only {len(boxes)} boxes, using default anchors")
        return get_default_anchors()

    # Extract widths and heights
    wh = np.array([[box[2] - box[0], box[3] - box[1]] for box in boxes])

    # Initialize centroids randomly
    rng = np.random.default_rng(42)  # Reproducible
    indices = rng.choice(len(wh), num_anchors, replace=False)
    centroids = wh[indices].copy()

    def box_iou_wh(wh1, wh2):
        """IoU for axis-aligned boxes at origin (just compare w, h)."""
        inter_w = min(wh1[0], wh2[0])
        inter_h = min(wh1[1], wh2[1])
        inter_area = inter_w * inter_h
        union_area = wh1[0] * wh1[1] + wh2[0] * wh2[1] - inter_area
        return inter_area / (union_area + 1e-8)

    for iteration in range(max_iter):
        # Assign each box to nearest centroid (by IoU)
        assignments = []
        for box_wh in wh:
            ious = [box_iou_wh(box_wh, c) for c in centroids]
            assignments.append(np.argmax(ious))
        assignments = np.array(assignments)

        # Update centroids (mean of assigned boxes)
        new_centroids = []
        for i in range(num_anchors):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids.append(wh[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids, atol=1.0):
            logger.info(f"K-means converged at iteration {iteration}")
            break
        centroids = new_centroids

    # Sort by area (small to large)
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(areas)
    centroids = centroids[sorted_indices]

    anchors = [(int(max(1, w)), int(max(1, h))) for w, h in centroids]
    logger.info(f"Computed anchors (w, h): {anchors}")
    return anchors


def compute_anchor_coverage(boxes: list[list[float]], anchors: list[tuple[int, int]]) -> dict:
    """
    Test anchor coverage on ground truth boxes.

    Returns:
        Dict with coverage statistics
    """
    import torch
    from torchvision.ops import box_iou

    if not boxes:
        return {
            "total_boxes": 0,
            "covered_at_0.5": 0,
            "coverage_pct_0.5": 0,
            "covered_at_0.3": 0,
            "coverage_pct_0.3": 0,
            "min_iou": 0,
            "median_iou": 0,
        }

    # Generate all anchor boxes for 1024x1024 image
    # FPN strides: 4, 8, 16, 32, 64
    IMAGE_SIZE = 1024
    STRIDES = [4, 8, 16, 32, 64]

    all_anchors = []
    for stride in STRIDES:
        feat_size = IMAGE_SIZE // stride
        for y in range(feat_size):
            for x in range(feat_size):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                for w, h in anchors:
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    all_anchors.append([x1, y1, x2, y2])

    anchor_tensor = torch.tensor(all_anchors, dtype=torch.float32)
    max_ious = []

    for gt_box in boxes:
        gt_tensor = torch.tensor([gt_box], dtype=torch.float32)
        ious = box_iou(anchor_tensor, gt_tensor)
        max_ious.append(ious.max().item())

    covered_50 = sum(iou > 0.5 for iou in max_ious)
    covered_30 = sum(iou > 0.3 for iou in max_ious)

    return {
        "total_boxes": len(boxes),
        "covered_at_0.5": covered_50,
        "coverage_pct_0.5": 100 * covered_50 / len(boxes),
        "covered_at_0.3": covered_30,
        "coverage_pct_0.3": 100 * covered_30 / len(boxes),
        "min_iou": min(max_ious),
        "median_iou": float(np.median(max_ious)),
    }


def anchors_to_generator_format(anchors: list[tuple[int, int]]) -> tuple[tuple, tuple]:
    """
    Convert anchor (w, h) list to AnchorGenerator format.

    CRITICAL: Keep k-means pairs intact. Do NOT separate sizes and aspects
    into independent lists (that would create N×M crossed combinations
    instead of N specific pairs).

    Args:
        anchors: List of (width, height) tuples from k-means

    Returns:
        (sizes, aspect_ratios) tuples for AnchorGenerator
        Each is a 5-tuple (one per FPN level) with same values
    """
    # Compute size (sqrt of area) and aspect (h/w) for each anchor
    sizes = tuple(int(np.sqrt(w * h)) for w, h in anchors)
    aspects = tuple(round(h / w, 2) for w, h in anchors)

    # Same anchors at all 5 FPN levels
    # This gives us exactly len(anchors) anchor shapes, NOT sizes × aspects
    return (sizes,) * 5, (aspects,) * 5


def get_default_anchors() -> list[tuple[int, int]]:
    """Default anchors covering common signal shapes (wide to tall)."""
    return [
        (100, 10),  # Very wide (aspect 0.1)
        (80, 20),  # Wide (aspect 0.25)
        (60, 30),  # Wide (aspect 0.5)
        (50, 50),  # Square (aspect 1.0)
        (40, 60),  # Slightly tall (aspect 1.5)
        (30, 60),  # Tall (aspect 2.0)
        (20, 80),  # Very tall (aspect 4.0)
        (100, 50),  # Large wide
        (50, 100),  # Large tall
    ]
