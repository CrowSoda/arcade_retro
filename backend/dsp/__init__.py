"""
DSP Module for G20 Sub-Band Tuning

Implements proper sub-band extraction for CNN training data:
- Mix → Filter → Decimate pipeline
- 60-80 dB stopband attenuation
- DC offset removal
- Unit power normalization
"""

# Logging
from logger_config import get_logger

from .filters import calculate_filter_taps, design_aa_filter
from .subband_extractor import ExtractionParams, SubbandExtractor

logger = get_logger("__init__")


__all__ = [
    "SubbandExtractor",
    "ExtractionParams",
    "design_aa_filter",
    "calculate_filter_taps",
]
