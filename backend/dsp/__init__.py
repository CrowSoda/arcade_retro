"""
DSP Module for G20 Sub-Band Tuning

Implements proper sub-band extraction for CNN training data:
- Mix → Filter → Decimate pipeline
- 60-80 dB stopband attenuation
- DC offset removal
- Unit power normalization
"""

from .subband_extractor import SubbandExtractor, ExtractionParams
from .filters import design_aa_filter, calculate_filter_taps

__all__ = [
    'SubbandExtractor',
    'ExtractionParams', 
    'design_aa_filter',
    'calculate_filter_taps',
]
