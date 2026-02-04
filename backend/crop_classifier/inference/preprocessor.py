"""
Crop preprocessing pipeline.

Extracts signal regions from spectrograms and normalizes them for classifier input.

Key design decisions:
1. Aspect-preserving resize (letterbox) - signals have characteristic aspect ratios
2. Zero-padding for letterbox regions (not edge replication)
3. Per-crop normalization (mean=0, std=1) - handles varying signal strengths
4. Target size 32×64 (H×W) - wider than tall, matching typical signal aspect ratios

The target size was chosen based on analysis of Creamy_Shrimp dataset where
signals tend to be wider than tall (RF signals spread in time/frequency).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .blob_detector import BoundingBox


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for crop preprocessing."""

    # Target dimensions (height, width)
    # 32×64 chosen because signals are typically wider than tall
    target_height: int = 32
    target_width: int = 64

    # Padding around detected bbox (fraction of box size)
    padding_pct: float = 0.15

    # Normalization mode: 'per_crop', 'global', or 'none'
    normalization: str = "per_crop"

    # Interpolation for resize
    interpolation: int = cv2.INTER_LINEAR

    # Minimum valid crop size before resize
    min_crop_size: int = 4

    @property
    def target_size(self) -> tuple[int, int]:
        """Return (height, width) tuple."""
        return (self.target_height, self.target_width)


# Default config - 32×64 (H×W) for wide signals
DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()


class CropPreprocessor:
    """
    Preprocesses spectrogram crops for classifier input.

    Handles:
    - Padding around bounding boxes
    - Aspect-preserving resize (letterbox)
    - Per-crop normalization

    Example:
        preprocessor = CropPreprocessor()
        crops = preprocessor.extract_batch(spectrogram, boxes)
        # crops shape: (N, 1, 32, 64)
    """

    def __init__(self, config: PreprocessConfig | None = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or DEFAULT_PREPROCESS_CONFIG

    def extract_single(
        self,
        image: np.ndarray,
        bbox: BoundingBox | dict,
        add_padding: bool = True,
    ) -> torch.Tensor:
        """
        Extract and preprocess a single crop.

        Args:
            image: Source spectrogram (H, W) grayscale
            bbox: Bounding box (BoundingBox or dict)
            add_padding: Whether to add padding around bbox

        Returns:
            Tensor of shape (1, target_h, target_w), normalized
        """
        # Convert dict to BoundingBox if needed
        if isinstance(bbox, dict):
            bbox = BoundingBox.from_dict(bbox)

        # Optionally add padding
        if add_padding:
            bbox = bbox.with_padding(image.shape, self.config.padding_pct)

        return preprocess_crop(
            image,
            bbox.to_dict(),
            target_size=self.config.target_size,
            padding_pct=0.0,  # Already applied above
            normalization=self.config.normalization,
            interpolation=self.config.interpolation,
            min_crop_size=self.config.min_crop_size,
        )

    def extract_batch(
        self,
        image: np.ndarray,
        bboxes: list[BoundingBox | dict],
        add_padding: bool = True,
    ) -> torch.Tensor:
        """
        Extract and preprocess multiple crops.

        Args:
            image: Source spectrogram (H, W) grayscale
            bboxes: List of bounding boxes
            add_padding: Whether to add padding around each bbox

        Returns:
            Tensor of shape (N, 1, target_h, target_w)
        """
        if not bboxes:
            return torch.empty(0, 1, self.config.target_height, self.config.target_width)

        crops = [self.extract_single(image, bbox, add_padding) for bbox in bboxes]
        return torch.stack(crops)


def preprocess_crop(
    image: np.ndarray,
    bbox: dict,
    target_size: tuple[int, int] = (32, 64),
    padding_pct: float = 0.15,
    normalization: str = "per_crop",
    interpolation: int = cv2.INTER_LINEAR,
    min_crop_size: int = 4,
) -> torch.Tensor:
    """
    Extract and preprocess a crop for classification.

    This is the standalone function version for direct use.

    Args:
        image: Source spectrogram (H, W) grayscale float32
        bbox: Dict with x_min, y_min, x_max, y_max
        target_size: Output size (height, width) - default 32×64
        padding_pct: Padding around bbox as fraction of box size
        normalization: 'per_crop', 'global', or 'none'
        interpolation: OpenCV interpolation flag
        min_crop_size: Minimum dimension before resize

    Returns:
        Tensor of shape (1, target_h, target_w), normalized
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # --- Step 1: Add padding ---
    box_w = bbox["x_max"] - bbox["x_min"]
    box_h = bbox["y_max"] - bbox["y_min"]

    pad_x = int(box_w * padding_pct)
    pad_y = int(box_h * padding_pct)

    x1 = max(0, bbox["x_min"] - pad_x)
    y1 = max(0, bbox["y_min"] - pad_y)
    x2 = min(w, bbox["x_max"] + pad_x)
    y2 = min(h, bbox["y_max"] + pad_y)

    # --- Step 2: Extract crop ---
    crop = image[y1:y2, x1:x2].copy()

    # Handle degenerate boxes
    if crop.size == 0 or crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
        return torch.zeros(1, target_h, target_w, dtype=torch.float32)

    # Ensure float32
    if crop.dtype != np.float32:
        if crop.dtype == np.uint8:
            crop = crop.astype(np.float32) / 255.0
        else:
            crop = crop.astype(np.float32)

    # --- Step 3: Aspect-preserving resize (letterbox) ---
    crop_h, crop_w = crop.shape[:2]

    # Calculate scale to fit in target while preserving aspect
    scale = min(target_h / crop_h, target_w / crop_w)
    new_h = max(1, int(crop_h * scale))
    new_w = max(1, int(crop_w * scale))

    # Resize with specified interpolation
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)

    # Create zero-padded output (letterbox)
    result = np.zeros((target_h, target_w), dtype=np.float32)

    # Center the resized crop
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    # --- Step 4: Normalization ---
    if normalization == "per_crop":
        mean = result.mean()
        std = result.std()

        if std > 1e-8:
            result = (result - mean) / std
        else:
            # Constant crop - just center it
            result = result - mean

    elif normalization == "global":
        # Placeholder for dataset-level stats
        # Would need external mean/std values
        global_mean = 0.5
        global_std = 0.25
        result = (result - global_mean) / global_std

    # 'none' = no normalization

    # --- Step 5: Convert to tensor ---
    tensor = torch.from_numpy(result).unsqueeze(0)  # (1, H, W)

    return tensor


def preprocess_batch(
    image: np.ndarray,
    bboxes: list[dict],
    target_size: tuple[int, int] = (32, 64),
    padding_pct: float = 0.15,
    normalization: str = "per_crop",
) -> torch.Tensor:
    """
    Preprocess multiple crops efficiently.

    Standalone function version for direct use.

    Args:
        image: Source spectrogram
        bboxes: List of bbox dicts
        target_size: Output size (height, width)
        padding_pct: Padding fraction

    Returns:
        Tensor of shape (N, 1, H, W)
    """
    if not bboxes:
        return torch.empty(0, 1, target_size[0], target_size[1])

    crops = [
        preprocess_crop(
            image,
            bbox,
            target_size=target_size,
            padding_pct=padding_pct,
            normalization=normalization,
        )
        for bbox in bboxes
    ]

    return torch.stack(crops)


def letterbox_resize(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_value: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Resize image with aspect preservation and zero-padding.

    Utility function for custom preprocessing pipelines.

    Args:
        image: Input image (H, W) or (H, W, C)
        target_size: (height, width) of output
        pad_value: Value for padding regions
        interpolation: OpenCV interpolation flag

    Returns:
        (resized_image, (y_offset, x_offset), scale)
    """
    target_h, target_w = target_size
    img_h, img_w = image.shape[:2]

    # Calculate scale
    scale = min(target_h / img_h, target_w / img_w)
    new_h = max(1, int(img_h * scale))
    new_w = max(1, int(img_w * scale))

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Create output with padding
    if len(image.shape) == 3:
        result = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        result = np.full((target_h, target_w), pad_value, dtype=image.dtype)

    # Center
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    if len(image.shape) == 3:
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w, :] = resized
    else:
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return result, (y_offset, x_offset), scale
