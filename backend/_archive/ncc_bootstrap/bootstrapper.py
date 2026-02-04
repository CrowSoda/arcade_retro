"""
Signal Bootstrapper: Seed-based signal expansion.

Flow:
1. User draws ~20 boxes → seeds
2. Blob detection → candidates
3. Template match → rank candidates by similarity to seeds
4. User confirms (swipe UI) → training data
5. Train Siamese → generalization

Template matching finds identical signals (fast, no training).
Siamese learns to generalize to variations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def score_candidates_by_seeds(
    seeds: list[np.ndarray],
    candidates: list[np.ndarray],
) -> list[float]:
    """
    Score each candidate by similarity to user's seed templates.

    Uses Normalized Cross-Correlation (NCC):
    - 1.0 = identical
    - 0.0 = uncorrelated
    - -1.0 = inverse

    Args:
        seeds: Crops from user's manually drawn boxes
        candidates: Crops from blob detection

    Returns:
        Similarity score for each candidate (max similarity to any seed)
    """
    if not seeds or not candidates:
        return [0.0] * len(candidates)

    scores = []

    # Normalize seeds to common size for comparison
    target_size = (64, 32)  # width, height

    # Pre-process seeds once
    seed_templates = []
    for seed in seeds:
        if seed.size == 0 or seed.shape[0] < 2 or seed.shape[1] < 2:
            continue
        resized = cv2.resize(seed, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32)
        # Zero-mean for NCC
        normalized = normalized - normalized.mean()
        std = normalized.std()
        if std > 1e-6:
            normalized = normalized / std
        seed_templates.append(normalized)

    if not seed_templates:
        return [0.0] * len(candidates)

    for candidate in candidates:
        if candidate.size == 0 or candidate.shape[0] < 2 or candidate.shape[1] < 2:
            scores.append(0.0)
            continue

        # Resize candidate to match template size
        cand_resized = cv2.resize(candidate, target_size, interpolation=cv2.INTER_AREA)
        cand_norm = cand_resized.astype(np.float32)
        cand_norm = cand_norm - cand_norm.mean()
        cand_std = cand_norm.std()
        if cand_std > 1e-6:
            cand_norm = cand_norm / cand_std

        best_score = -1.0

        for seed_norm in seed_templates:
            # Normalized cross-correlation (direct computation)
            # Since both are zero-mean and unit-std, NCC = dot product / N
            ncc = float(np.sum(seed_norm * cand_norm) / seed_norm.size)

            # Alternative: use cv2.matchTemplate
            # result = cv2.matchTemplate(cand_norm, seed_norm, cv2.TM_CCOEFF_NORMED)
            # ncc = float(result.max())

            best_score = max(best_score, ncc)

        scores.append(best_score)

    return scores


@dataclass
class BootstrapResult:
    """Result from find_similar()."""

    seeds: list[np.ndarray] = field(default_factory=list)
    candidates: list[np.ndarray] = field(default_factory=list)
    boxes: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


class SignalBootstrapper:
    """
    Bootstrap signal labeling from user's seed examples.

    Flow:
    1. User draws boxes → seeds
    2. Blob detection → candidates
    3. Template match → rank candidates
    4. User confirms → training data
    5. Train Siamese → generalization

    Example:
        bootstrapper = SignalBootstrapper(blob_detector)

        # User draws boxes
        result = bootstrapper.find_similar(spectrogram, seed_boxes, top_k=50)

        # Show swipe UI, get confirmations
        bootstrapper.confirm(result, confirmed_indices=[0,1,5], rejected_indices=[2,3,4])

        # Train
        crops, labels = bootstrapper.get_training_data()
    """

    def __init__(self, blob_detector):
        """
        Initialize bootstrapper.

        Args:
            blob_detector: BlobDetector instance for finding candidates
        """
        self.blob_detector = blob_detector

        # Accumulated labels across sessions
        self.positives: list[np.ndarray] = []  # Confirmed signals
        self.negatives: list[np.ndarray] = []  # Confirmed noise

        # Pending data from last find_similar() call
        self._pending_seeds: list[np.ndarray] = []
        self._pending_candidates: list[np.ndarray] = []
        self._pending_boxes: list[dict] = []
        self._pending_scores: list[float] = []

    def find_similar(
        self,
        spectrogram: np.ndarray,
        seed_boxes: list[dict],
        top_k: int = 50,
    ) -> BootstrapResult:
        """
        Find candidates similar to user's seeds.

        Args:
            spectrogram: Full spectrogram image (H, W)
            seed_boxes: User-drawn boxes [{'x_min', 'y_min', 'x_max', 'y_max'}, ...]
            top_k: Number of top candidates to return

        Returns:
            BootstrapResult with seeds, candidates, boxes, scores
        """
        logger.info(f"[Bootstrapper] Finding similar to {len(seed_boxes)} seeds")

        # 1. Extract seed crops from spectrogram
        seeds = []
        for box in seed_boxes:
            crop = self._extract_crop(spectrogram, box)
            if crop.size > 0:
                seeds.append(crop)

        if not seeds:
            logger.warning("[Bootstrapper] No valid seeds extracted")
            return BootstrapResult(seeds=seeds)

        # 2. Run blob detection
        result = self.blob_detector.detect(spectrogram)
        logger.info(f"[Bootstrapper] Blob detection found {result.count} candidates")

        # 3. Filter out boxes overlapping with seeds (IoU > 0.3)
        candidate_boxes = [
            box.to_dict()
            for box in result.boxes
            if not self._overlaps_seeds(box.to_dict(), seed_boxes)
        ]

        logger.info(f"[Bootstrapper] After overlap filter: {len(candidate_boxes)} candidates")

        if not candidate_boxes:
            return BootstrapResult(seeds=seeds)

        # 4. Extract candidate crops
        candidates = [self._extract_crop(spectrogram, box) for box in candidate_boxes]

        # Filter out empty crops
        valid = [(c, b) for c, b in zip(candidates, candidate_boxes, strict=False) if c.size > 0]
        if valid:
            candidates, candidate_boxes = zip(*valid, strict=False)
            candidates = list(candidates)
            candidate_boxes = list(candidate_boxes)
        else:
            candidates = []
            candidate_boxes = []

        # 5. Score by template similarity
        scores = score_candidates_by_seeds(seeds, candidates)

        # 6. Rank and return top-K
        ranked = sorted(
            zip(scores, candidates, candidate_boxes, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        if ranked:
            top_scores, top_candidates, top_boxes = zip(*ranked, strict=False)
        else:
            top_scores, top_candidates, top_boxes = [], [], []

        # Store for later confirmation
        self._pending_seeds = seeds
        self._pending_candidates = list(top_candidates)
        self._pending_boxes = list(top_boxes)
        self._pending_scores = list(top_scores)

        logger.info(
            f"[Bootstrapper] Returning top {len(top_candidates)} candidates, "
            f"score range: {min(top_scores) if top_scores else 0:.3f} - {max(top_scores) if top_scores else 0:.3f}"
        )

        return BootstrapResult(
            seeds=seeds,
            candidates=list(top_candidates),
            boxes=list(top_boxes),
            scores=list(top_scores),
        )

    def confirm(
        self,
        confirmed_indices: list[int],
        rejected_indices: list[int],
    ) -> dict:
        """
        Record user's confirmations from swipe UI.

        Args:
            confirmed_indices: Indices user accepted (swipe right)
            rejected_indices: Indices user rejected (swipe left)

        Returns:
            Stats dict with counts
        """
        # Seeds are always positive
        for seed in self._pending_seeds:
            self.positives.append(seed)

        # Confirmed candidates are positive
        for idx in confirmed_indices:
            if 0 <= idx < len(self._pending_candidates):
                self.positives.append(self._pending_candidates[idx])

        # Rejected candidates are negative
        for idx in rejected_indices:
            if 0 <= idx < len(self._pending_candidates):
                self.negatives.append(self._pending_candidates[idx])

        logger.info(
            f"[Bootstrapper] Confirmed {len(confirmed_indices)}, "
            f"rejected {len(rejected_indices)}. "
            f"Total: {len(self.positives)} pos, {len(self.negatives)} neg"
        )

        # Clear pending
        self._pending_seeds = []
        self._pending_candidates = []
        self._pending_boxes = []
        self._pending_scores = []

        return self.stats()

    def get_training_data(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Get accumulated training data.

        Returns:
            (crops, labels) where label 1 = signal, 0 = noise
        """
        crops = self.positives + self.negatives
        labels = [1] * len(self.positives) + [0] * len(self.negatives)
        return crops, labels

    def stats(self) -> dict:
        """Current labeling statistics."""
        return {
            "positives": len(self.positives),
            "negatives": len(self.negatives),
            "total": len(self.positives) + len(self.negatives),
            "ready_to_train": len(self.positives) >= 10,
        }

    def clear(self):
        """Clear all accumulated labels."""
        self.positives = []
        self.negatives = []
        self._pending_seeds = []
        self._pending_candidates = []
        self._pending_boxes = []
        self._pending_scores = []

    def _extract_crop(self, spectrogram: np.ndarray, box: dict) -> np.ndarray:
        """Extract crop region from spectrogram."""
        x_min = max(0, int(box.get("x_min", 0)))
        y_min = max(0, int(box.get("y_min", 0)))
        x_max = min(spectrogram.shape[1], int(box.get("x_max", 0)))
        y_max = min(spectrogram.shape[0], int(box.get("y_max", 0)))

        if x_max <= x_min or y_max <= y_min:
            return np.array([])

        return spectrogram[y_min:y_max, x_min:x_max].copy()

    def _overlaps_seeds(self, box: dict, seed_boxes: list[dict]) -> bool:
        """Check if box overlaps any seed (IoU > 0.3)."""
        for seed in seed_boxes:
            if self._iou(box, seed) > 0.3:
                return True
        return False

    def _iou(self, a: dict, b: dict) -> float:
        """Intersection over union."""
        x1 = max(a.get("x_min", 0), b.get("x_min", 0))
        y1 = max(a.get("y_min", 0), b.get("y_min", 0))
        x2 = min(a.get("x_max", 0), b.get("x_max", 0))
        y2 = min(a.get("y_max", 0), b.get("y_max", 0))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area_a = (a.get("x_max", 0) - a.get("x_min", 0)) * (a.get("y_max", 0) - a.get("y_min", 0))
        area_b = (b.get("x_max", 0) - b.get("x_min", 0)) * (b.get("y_max", 0) - b.get("y_min", 0))

        if area_a + area_b - inter <= 0:
            return 0.0

        return inter / (area_a + area_b - inter)
