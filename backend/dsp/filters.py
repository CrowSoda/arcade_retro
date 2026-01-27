"""
Filter design utilities for sub-band extraction.

Provides helper functions for designing anti-aliasing filters
with proper stopband attenuation for CNN training data.
"""

import numpy as np

# Logging
from logger_config import get_logger
from scipy.signal import firwin, kaiserord

logger = get_logger("filters")


def design_aa_filter(
    cutoff_hz: float,
    sample_rate: float,
    stopband_db: float = 60.0,
    transition_width_pct: float = 0.1,
) -> np.ndarray:
    """
    Design anti-aliasing filter with specified stopband attenuation.

    Uses Kaiser window design for precise control over stopband attenuation.

    Args:
        cutoff_hz: Filter cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        stopband_db: Desired stopband attenuation in dB (positive number)
        transition_width_pct: Transition band width as fraction of cutoff (0.1 = 10%)

    Returns:
        FIR filter coefficients (float32)
    """
    # Transition bandwidth
    transition_width = cutoff_hz * transition_width_pct
    nyquist = sample_rate / 2

    # Normalize for kaiserord (fraction of Nyquist)
    normalized_transition = transition_width / nyquist

    # Kaiser window design
    try:
        numtaps, beta = kaiserord(stopband_db, normalized_transition)
    except ValueError as e:
        # Fallback if parameters are out of range
        logger.error(f"[filters] kaiserord failed ({e}), using fallback")
        numtaps = 255
        beta = _estimate_kaiser_beta(stopband_db)

    # Ensure odd number of taps (linear phase)
    if numtaps % 2 == 0:
        numtaps += 1

    # Reasonable bounds
    numtaps = max(numtaps, 63)
    numtaps = min(numtaps, 4095)

    # Normalized cutoff
    normalized_cutoff = cutoff_hz / nyquist
    normalized_cutoff = min(normalized_cutoff, 0.99)

    # Design filter
    taps = firwin(numtaps, normalized_cutoff, window=("kaiser", beta))

    return taps.astype(np.float32)


def calculate_filter_taps(
    cutoff_hz: float,
    sample_rate: float,
    stopband_db: float = 60.0,
    transition_width_pct: float = 0.1,
) -> int:
    """
    Calculate number of filter taps needed for given specs.

    Useful for planning computation resources.

    Args:
        cutoff_hz: Filter cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        stopband_db: Desired stopband attenuation in dB
        transition_width_pct: Transition band width as fraction of cutoff

    Returns:
        Number of filter taps required
    """
    transition_width = cutoff_hz * transition_width_pct
    nyquist = sample_rate / 2
    normalized_transition = transition_width / nyquist

    try:
        numtaps, _ = kaiserord(stopband_db, normalized_transition)
    except ValueError:
        numtaps = 255

    if numtaps % 2 == 0:
        numtaps += 1

    return max(63, min(numtaps, 4095))


def _estimate_kaiser_beta(stopband_db: float) -> float:
    """
    Estimate Kaiser window beta parameter from stopband attenuation.

    Based on empirical Kaiser formula.
    """
    if stopband_db > 50:
        beta = 0.1102 * (stopband_db - 8.7)
    elif stopband_db >= 21:
        beta = 0.5842 * (stopband_db - 21) ** 0.4 + 0.07886 * (stopband_db - 21)
    else:
        beta = 0.0

    return beta
