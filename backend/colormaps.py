"""
Colormap LUT Generation - Consolidated colormap lookup tables for waterfall display

Provides pre-computed 256-entry RGB lookup tables for efficient dB → color mapping.
All colormaps are generated from control points and cached as numpy arrays.

Available colormaps:
- VIRIDIS (0): Default, perceptually uniform, colorblind-friendly
- PLASMA (1): High contrast purple-pink-orange-yellow
- INFERNO (2): Black-red-orange-yellow-white
- MAGMA (3): Black-purple-red-orange-white
- TURBO (4): Rainbow-like, high color variation

Usage:
    from colormaps import get_colormap, COLORMAP_NAMES

    lut = get_colormap(0)  # Returns VIRIDIS LUT
    rgb = lut[db_normalized]  # Index with 0-255 normalized dB value
"""

import numpy as np

# Logging
from logger_config import get_logger

logger = get_logger("colormaps")


# =============================================================================
# COLORMAP CONTROL POINTS
# Each colormap is defined by RGB control points at specific positions (0-255)
# =============================================================================

# Viridis control points (68,1,84) → (253,231,37)
VIRIDIS_CONTROL_POINTS = [
    (68, 1, 84),
    (72, 40, 120),
    (62, 73, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (109, 205, 89),
    (180, 222, 44),
    (253, 231, 37),
]

# Plasma control points
PLASMA_CONTROL_POINTS = [
    (13, 8, 135),
    (75, 3, 161),
    (125, 3, 168),
    (168, 34, 150),
    (203, 70, 121),
    (229, 107, 93),
    (248, 148, 65),
    (253, 195, 40),
    (240, 249, 33),
]

# Inferno control points
INFERNO_CONTROL_POINTS = [
    (0, 0, 4),
    (22, 11, 57),
    (66, 10, 104),
    (106, 23, 110),
    (147, 38, 103),
    (188, 55, 84),
    (221, 81, 58),
    (243, 118, 27),
    (252, 165, 10),
    (246, 215, 70),
    (252, 255, 164),
]

# Magma control points
MAGMA_CONTROL_POINTS = [
    (0, 0, 4),
    (18, 14, 54),
    (51, 16, 101),
    (89, 26, 114),
    (129, 37, 120),
    (168, 50, 117),
    (205, 70, 108),
    (234, 107, 101),
    (250, 157, 117),
    (252, 207, 163),
    (252, 253, 191),
]

# Turbo control points (rainbow-like)
TURBO_CONTROL_POINTS = [
    (48, 18, 59),
    (68, 81, 191),
    (33, 145, 237),
    (40, 200, 179),
    (99, 228, 118),
    (170, 241, 68),
    (231, 229, 51),
    (252, 185, 56),
    (247, 123, 54),
    (220, 55, 35),
    (122, 4, 3),
]

# =============================================================================
# LUT GENERATION
# =============================================================================


def _interpolate_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linearly interpolate between two RGB colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _generate_lut(control_points: list) -> np.ndarray:
    """
    Generate a 256-entry RGB lookup table from control points.

    Args:
        control_points: List of (R, G, B) tuples defining the colormap

    Returns:
        numpy array of shape (256, 3) with uint8 RGB values
    """
    n_points = len(control_points)
    lut = np.zeros((256, 3), dtype=np.uint8)

    # Calculate positions for each control point (evenly spaced 0-255)
    positions = [int(255 * i / (n_points - 1)) for i in range(n_points)]

    for i in range(256):
        # Find which segment we're in
        for j in range(n_points - 1):
            if positions[j] <= i <= positions[j + 1]:
                # Interpolate within this segment
                t = (
                    (i - positions[j]) / (positions[j + 1] - positions[j])
                    if positions[j + 1] != positions[j]
                    else 0
                )
                lut[i] = _interpolate_color(control_points[j], control_points[j + 1], t)
                break

    return lut


# =============================================================================
# PRE-COMPUTED LOOKUP TABLES
# =============================================================================

VIRIDIS_LUT = _generate_lut(VIRIDIS_CONTROL_POINTS)
PLASMA_LUT = _generate_lut(PLASMA_CONTROL_POINTS)
INFERNO_LUT = _generate_lut(INFERNO_CONTROL_POINTS)
MAGMA_LUT = _generate_lut(MAGMA_CONTROL_POINTS)
TURBO_LUT = _generate_lut(TURBO_CONTROL_POINTS)

# All LUTs in indexed list
COLORMAPS = [VIRIDIS_LUT, PLASMA_LUT, INFERNO_LUT, MAGMA_LUT, TURBO_LUT]
COLORMAP_NAMES = ["Viridis", "Plasma", "Inferno", "Magma", "Turbo"]


# =============================================================================
# PUBLIC API
# =============================================================================


def get_colormap(index: int) -> np.ndarray:
    """
    Get colormap LUT by index.

    Args:
        index: 0=Viridis, 1=Plasma, 2=Inferno, 3=Magma, 4=Turbo

    Returns:
        numpy array of shape (256, 3) with uint8 RGB values
    """
    return COLORMAPS[index % len(COLORMAPS)]


def get_colormap_by_name(name: str) -> np.ndarray:
    """
    Get colormap LUT by name.

    Args:
        name: One of 'viridis', 'plasma', 'inferno', 'magma', 'turbo' (case insensitive)

    Returns:
        numpy array of shape (256, 3) with uint8 RGB values
    """
    name_lower = name.lower()
    for i, cmap_name in enumerate(COLORMAP_NAMES):
        if cmap_name.lower() == name_lower:
            return COLORMAPS[i]

    # Default to viridis
    return VIRIDIS_LUT


def apply_colormap(data: np.ndarray, colormap_index: int = 0) -> np.ndarray:
    """
    Apply colormap to normalized data (0-255 uint8).

    Args:
        data: Input data normalized to 0-255 (uint8)
        colormap_index: Which colormap to use (0-4)

    Returns:
        RGB array with shape (*data.shape, 3)
    """
    lut = get_colormap(colormap_index)
    return lut[data]


def apply_colormap_db(
    data_db: np.ndarray, min_db: float = -100.0, max_db: float = -20.0, colormap_index: int = 0
) -> np.ndarray:
    """
    Apply colormap to dB data with normalization.

    Args:
        data_db: Input data in dB
        min_db: Minimum dB value (maps to LUT index 0)
        max_db: Maximum dB value (maps to LUT index 255)
        colormap_index: Which colormap to use (0-4)

    Returns:
        RGB array with shape (*data_db.shape, 3)
    """
    # Normalize to 0-255
    normalized = np.clip((data_db - min_db) / (max_db - min_db) * 255, 0, 255).astype(np.uint8)
    return apply_colormap(normalized, colormap_index)


# =============================================================================
# VIRIDIS BACKGROUND COLOR (for buffer initialization)
# =============================================================================

# Viridis dark purple - used to initialize empty buffer
VIRIDIS_BACKGROUND = (68, 1, 84, 255)  # RGBA
