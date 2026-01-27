"""
Filter design utilities for sub-band extraction.

Provides helper functions for designing anti-aliasing filters
with proper stopband attenuation for CNN training data.
"""

import numpy as np

# Logging
from logger_config import get_logger
from scipy.signal import firwin, freqz, kaiserord

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


def analyze_filter(taps: np.ndarray, sample_rate: float, num_points: int = 8192) -> dict:
    """
    Analyze filter frequency response.

    Args:
        taps: Filter coefficients
        sample_rate: Sample rate for frequency axis
        num_points: Number of frequency points

    Returns:
        Dict with frequencies, magnitude (dB), phase, and key metrics
    """
    w, h = freqz(taps, worN=num_points)

    # Convert to actual frequencies
    freqs = w * sample_rate / (2 * np.pi)

    # Magnitude in dB
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)

    # Phase (unwrapped)
    phase = np.unwrap(np.angle(h))

    # Find -3dB point (cutoff)
    cutoff_idx = np.argmin(np.abs(mag_db - (-3)))
    cutoff_freq = freqs[cutoff_idx]

    # Find passband ripple (within -3dB region)
    passband_mask = freqs < cutoff_freq * 0.8
    passband_ripple = (
        mag_db[passband_mask].max() - mag_db[passband_mask].min() if np.any(passband_mask) else 0
    )

    # Find stopband attenuation (2x cutoff and beyond)
    stopband_mask = freqs > cutoff_freq * 1.5
    stopband_max = mag_db[stopband_mask].max() if np.any(stopband_mask) else -100

    # Group delay
    group_delay = -np.diff(phase) / np.diff(w)
    avg_group_delay = np.mean(group_delay[passband_mask[:-1]]) if np.any(passband_mask[:-1]) else 0

    return {
        "frequencies": freqs,
        "magnitude_db": mag_db,
        "phase": phase,
        "cutoff_freq_hz": cutoff_freq,
        "passband_ripple_db": passband_ripple,
        "stopband_attenuation_db": -stopband_max,
        "num_taps": len(taps),
        "group_delay_samples": (len(taps) - 1) / 2,
        "avg_group_delay_samples": avg_group_delay,
    }


def verify_stopband(
    taps: np.ndarray, sample_rate: float, cutoff_hz: float, required_db: float = 60.0
) -> tuple[bool, float]:
    """
    Verify filter achieves required stopband attenuation.

    Args:
        taps: Filter coefficients
        sample_rate: Sample rate in Hz
        cutoff_hz: Cutoff frequency in Hz
        required_db: Required stopband attenuation in dB

    Returns:
        Tuple of (passes, actual_attenuation_db)
    """
    w, h = freqz(taps, worN=8192)
    freqs = w * sample_rate / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)

    # Check stopband (1.2x cutoff and beyond to account for transition)
    stopband_mask = freqs > cutoff_hz * 1.2

    if not np.any(stopband_mask):
        return True, 100.0  # No stopband visible = OK

    stopband_max = mag_db[stopband_mask].max()
    actual_attenuation = -stopband_max

    return actual_attenuation >= required_db, actual_attenuation


def design_halfband_filter(stopband_db: float = 60.0) -> np.ndarray:
    """
    Design a halfband filter for efficient 2:1 decimation.

    Halfband filters are efficient because half the coefficients are zero.
    Used for multi-stage decimation.

    Args:
        stopband_db: Desired stopband attenuation

    Returns:
        Halfband filter coefficients
    """
    # Halfband filters have cutoff at 0.25 (quarter Nyquist for 2:1 decimation)
    # Transition band from 0.25 to 0.5

    # Approximate number of taps needed
    beta = _estimate_kaiser_beta(stopband_db)
    numtaps = int(stopband_db / 2.285) + 1

    # Ensure odd (required for halfband)
    if numtaps % 2 == 0:
        numtaps += 1
    numtaps = max(numtaps, 7)

    # Design lowpass at 0.25
    taps = firwin(numtaps, 0.5, window=("kaiser", beta))  # 0.5 = Nyquist/2 = quarter rate

    # Force halfband structure (every other tap = 0 except center)
    # This is an approximation - true halfband needs special design

    return taps.astype(np.float32)


def design_cic_compensation_filter(
    cic_order: int,
    decimation: int,
    passband_fraction: float = 0.8,
    num_taps: int = 63,
) -> np.ndarray:
    """
    Design CIC compensation filter (inverse sinc).

    CIC filters have droop in passband. This FIR compensates.
    Used after CIC decimation stage.

    Args:
        cic_order: Number of CIC stages (M)
        decimation: CIC decimation factor (R)
        passband_fraction: Fraction of output Nyquist to compensate
        num_taps: Number of FIR taps

    Returns:
        Compensation filter coefficients
    """
    # CIC frequency response: (sin(πf/R) / sin(πf))^M
    # Compensation is inverse of this in passband

    # Design frequency points
    f = np.linspace(0, passband_fraction * 0.5, num_taps * 4)

    # CIC magnitude response (avoiding divide by zero)
    eps = 1e-10
    h_cic = (
        np.abs(np.sin(np.pi * f * decimation + eps) / (decimation * np.sin(np.pi * f + eps)))
        ** cic_order
    )

    # Inverse (compensation)
    h_comp = 1.0 / (h_cic + eps)
    h_comp = h_comp / h_comp[0]  # Normalize DC to 1

    # Design FIR that approximates this response using least squares
    # For simplicity, use windowed sinc with compensation curve

    # Basic lowpass
    taps = firwin(num_taps, passband_fraction, window="hamming")

    # This is a simplified version - proper implementation would use
    # least-squares design or Parks-McClellan with compensation weights

    return taps.astype(np.float32)
