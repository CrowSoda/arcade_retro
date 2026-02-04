"""
DCM-based Signal Expansion Engine.

This module provides signal detection via:
1. DCM (Delayed Conjugate Multiply) preprocessing - ~7dB SNR improvement
2. FFT-based matched filter correlation - O(N log N)
3. CFAR detection with NMS - Constant False Alarm Rate

Flow:
    User labels ~20 signals → DCM templates → Correlate full recording
    → CFAR detect → Return ~200 candidates → User confirms → K-fold train

GPU REQUIRED: Uses CuPy for real-time performance on Jetson Orin.
"""

from .cfar import CFARDetector, cfar_detect
from .correlator import FFTCorrelator, fft_correlate_normalized
from .dcm import DCMProcessor, dcm_multi_delay
from .expander import ExpansionResult, SignalExpander

__all__ = [
    "DCMProcessor",
    "dcm_multi_delay",
    "FFTCorrelator",
    "fft_correlate_normalized",
    "CFARDetector",
    "cfar_detect",
    "SignalExpander",
    "ExpansionResult",
]
