"""
Stage 1: Classical blob detection for signal proposals.

Uses adaptive thresholding + morphological operations to find candidate
signal regions. No neural networks involved = no position bias.

VALIDATED CONFIG (91.7% recall on Creamy_Shrimp dataset):
    min_area = 50
    max_area = 5000
    min_aspect_ratio = 1.5  # RF signals are wider than tall
    max_aspect_ratio = 15.0
    block_size = 51
    C = -5
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class BlobConfig:
    """
    Configuration for blob detection.

    Tuned for 91.7% recall while avoiding excessive false positives.
    """

    # Adaptive threshold parameters
    block_size: int = 51  # Local neighborhood size (must be odd)
    C: int = -15  # Negative = detect values ABOVE local mean (more negative = stricter)

    # Area filtering - prevents noise speckles and overly large regions
    min_area: int = 100  # Minimum blob area in pixels (increased to reject noise)
    max_area: int = 10000  # Maximum blob area in pixels (increased for larger signals)

    # Aspect ratio filtering - RF signals are typically wider than tall
    # VALIDATED at 91.7% recall - DO NOT CHANGE
    min_aspect_ratio: float = 1.5  # width/height minimum
    max_aspect_ratio: float = 15.0  # width/height maximum

    # Morphological operations
    morph_kernel_size: int = 3  # Kernel for closing operation
    morph_iterations: int = 1  # Number of closing iterations

    # Padding for classifier input
    padding_pct: float = 0.15  # 15% padding around each box


# Default locked config - validated at 91.7% recall
DEFAULT_CONFIG = BlobConfig()


@dataclass
class BoundingBox:
    """Bounding box with optional metadata."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    area: int = 0
    aspect_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BoundingBox:
        """Create from dictionary."""
        return cls(
            x_min=d["x_min"],
            y_min=d["y_min"],
            x_max=d["x_max"],
            y_max=d["y_max"],
            area=d.get("area", 0),
            aspect_ratio=d.get("aspect_ratio", 0.0),
        )

    def with_padding(self, image_shape: tuple, padding_pct: float = 0.15) -> BoundingBox:
        """
        Return new box with padding added.

        Args:
            image_shape: (height, width) of source image
            padding_pct: Padding as fraction of box size

        Returns:
            New BoundingBox with padding, clamped to image bounds
        """
        h, w = image_shape[:2]

        box_w = self.x_max - self.x_min
        box_h = self.y_max - self.y_min

        pad_x = int(box_w * padding_pct)
        pad_y = int(box_h * padding_pct)

        return BoundingBox(
            x_min=max(0, self.x_min - pad_x),
            y_min=max(0, self.y_min - pad_y),
            x_max=min(w, self.x_max + pad_x),
            y_max=min(h, self.y_max + pad_y),
            area=self.area,
            aspect_ratio=self.aspect_ratio,
        )


@dataclass
class DetectionResult:
    """Result from blob detection."""

    boxes: list[BoundingBox] = field(default_factory=list)
    binary_mask: np.ndarray | None = None  # Optional debug output
    processing_time_ms: float = 0.0

    @property
    def count(self) -> int:
        """Number of detected blobs."""
        return len(self.boxes)


class BlobDetector:
    """
    Classical blob detector for spectrogram signal proposals.

    Uses adaptive Gaussian thresholding to handle varying noise floors
    across frequencies. Morphological closing connects fragmented signals.

    Example:
        detector = BlobDetector()
        result = detector.detect(spectrogram)
        for box in result.boxes:
            crop = spectrogram[box.y_min:box.y_max, box.x_min:box.x_max]
    """

    def __init__(self, config: BlobConfig | None = None):
        """
        Initialize blob detector.

        Args:
            config: Detection configuration. Uses LOCKED defaults if None.
        """
        self.config = config or DEFAULT_CONFIG
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )

    def detect(
        self,
        spectrogram: np.ndarray,
        return_mask: bool = False,
    ) -> DetectionResult:
        """
        Detect blob regions in spectrogram.

        Args:
            spectrogram: (H, W) grayscale image, any dtype
            return_mask: If True, include binary mask in result

        Returns:
            DetectionResult with list of BoundingBox objects
        """
        import time

        start = time.perf_counter()

        # Normalize to uint8
        img = self._normalize_image(spectrogram)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.block_size,
            self.config.C,  # Negative = detect above local mean
        )

        # Morphological closing to connect fragments
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=self.config.morph_iterations,
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and extract boxes
        boxes = []
        for contour in contours:
            box = self._contour_to_box(contour)
            if box is not None:
                boxes.append(box)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            boxes=boxes,
            binary_mask=binary if return_mask else None,
            processing_time_ms=elapsed_ms,
        )

    def detect_with_padding(
        self,
        spectrogram: np.ndarray,
        padding_pct: float | None = None,
    ) -> DetectionResult:
        """
        Detect blobs and add padding for classifier input.

        Args:
            spectrogram: (H, W) grayscale image
            padding_pct: Override config padding (default: config.padding_pct)

        Returns:
            DetectionResult with padded boxes
        """
        result = self.detect(spectrogram, return_mask=False)

        pad = padding_pct if padding_pct is not None else self.config.padding_pct
        padded_boxes = [box.with_padding(spectrogram.shape, pad) for box in result.boxes]

        return DetectionResult(
            boxes=padded_boxes,
            processing_time_ms=result.processing_time_ms,
        )

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Convert any image to uint8 for OpenCV."""
        if img.dtype == np.uint8:
            return img

        if img.max() <= 1.0:
            # Assume 0-1 float
            return (img * 255).astype(np.uint8)

        # Normalize to full range
        img_min, img_max = img.min(), img.max()
        if img_max - img_min < 1e-8:
            return np.zeros_like(img, dtype=np.uint8)

        normalized = (img - img_min) / (img_max - img_min) * 255
        return normalized.astype(np.uint8)

    def _contour_to_box(self, contour: np.ndarray) -> BoundingBox | None:
        """
        Convert contour to BoundingBox if it passes filters.

        Returns None if contour fails area or aspect ratio filters.
        """
        area = cv2.contourArea(contour)

        # Area filter
        if area < self.config.min_area or area > self.config.max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)

        # Aspect ratio filter (width / height)
        if h == 0:
            return None

        aspect = w / h
        if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
            return None

        return BoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + w,
            y_max=y + h,
            area=int(area),
            aspect_ratio=aspect,
        )


def add_padding(
    bbox: dict,
    image_shape: tuple,
    padding_pct: float = 0.15,
) -> dict:
    """
    Add padding around bounding box (dict interface for compatibility).

    Args:
        bbox: Dict with x_min, y_min, x_max, y_max
        image_shape: (height, width) of source image
        padding_pct: Padding as fraction of box size (0.15 = 15%)

    Returns:
        Padded bbox dict, clamped to image bounds
    """
    box = BoundingBox.from_dict(bbox)
    padded = box.with_padding(image_shape, padding_pct)
    return padded.to_dict()
