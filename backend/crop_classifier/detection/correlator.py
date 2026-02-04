"""
FFT-based Matched Filter Correlation.

Cross-correlation in frequency domain is O(N log N) instead of O(N × M).

Math:
    Corr(template, signal) = IFFT( FFT(signal) × conj(FFT(template_padded)) )

Normalized Cross-Correlation (NCC):
    NCC = Corr / (σ_signal × σ_template × len(template))
    Output range: -1 to +1, where +1 = perfect match

GPU REQUIRED: Uses CuPy for real-time performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dcm import require_gpu

logger = logging.getLogger(__name__)

# Import CuPy (GPU requirement already checked in dcm.py)
try:
    import cupy as cp
    from cupyx.scipy import fft as cufft
except ImportError:
    cp = None
    cufft = None


@dataclass
class CorrelatorConfig:
    """Configuration for FFT correlator."""

    normalize: bool = True
    """Normalize output to NCC range [-1, 1]."""

    use_magnitude: bool = True
    """Return magnitude (abs) of correlation. If False, returns complex."""

    chunk_size: int = 2**24  # 16M samples
    """Max samples to process at once (GPU memory limit). ~128MB for complex64."""


class FFTCorrelator:
    """
    GPU-accelerated FFT cross-correlation.

    Efficiently correlates templates against long signals using frequency-domain
    multiplication.

    Example:
        correlator = FFTCorrelator()

        # Single template
        corr = correlator.correlate(template, signal)

        # Multiple templates - take max across all
        corr = correlator.correlate_multi(templates, signal)
    """

    def __init__(self, config: CorrelatorConfig | None = None):
        require_gpu()
        self.config = config or CorrelatorConfig()
        logger.info("[Correlator] Initialized")

    def correlate(
        self,
        template: np.ndarray,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Correlate single template against signal.

        Args:
            template: DCM-processed template (complex64)
            signal: DCM-processed signal to search (complex64)

        Returns:
            Correlation output. If normalized, values in [-1, 1].
            Length = len(signal) - len(template) + 1
        """
        if len(template) > len(signal):
            raise ValueError(f"Template ({len(template)}) longer than signal ({len(signal)})")

        # Check if chunking needed
        if len(signal) > self.config.chunk_size:
            return self._correlate_chunked(template, signal)

        return self._correlate_single(template, signal)

    def correlate_multi(
        self,
        templates: list[np.ndarray],
        signal: np.ndarray,
        reduction: str = "max",
    ) -> np.ndarray:
        """
        Correlate multiple templates and combine results.

        Args:
            templates: List of DCM-processed templates
            signal: DCM-processed signal to search
            reduction: How to combine results
                - "max": Maximum across templates (default)
                - "mean": Average across templates

        Returns:
            Combined correlation output
        """
        if not templates:
            raise ValueError("At least one template required")

        # Find common output length (shortest template determines this)
        min_template_len = min(len(t) for t in templates)
        output_len = len(signal) - min_template_len + 1

        if output_len <= 0:
            raise ValueError("Signal too short for templates")

        # Compute correlations
        correlations = []
        for template in templates:
            corr = self.correlate(template, signal)
            # Truncate to common length
            correlations.append(corr[:output_len])

        # Stack and reduce
        stacked = np.stack(correlations, axis=0)

        if reduction == "max":
            return np.max(stacked, axis=0)
        elif reduction == "mean":
            return np.mean(stacked, axis=0)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def _correlate_single(
        self,
        template: np.ndarray,
        signal: np.ndarray,
    ) -> np.ndarray:
        """FFT correlation without chunking."""
        # Transfer to GPU
        template_gpu = cp.asarray(template, dtype=cp.complex64)
        signal_gpu = cp.asarray(signal, dtype=cp.complex64)

        # Compute optimal FFT size (power of 2 for efficiency)
        n_fft = int(2 ** np.ceil(np.log2(len(signal) + len(template) - 1)))

        # FFT both signals (zero-padded to n_fft)
        template_fft = cufft.fft(template_gpu, n=n_fft)
        signal_fft = cufft.fft(signal_gpu, n=n_fft)

        # Correlation in frequency domain
        # Note: conj(template) because we want cross-correlation, not convolution
        corr_fft = signal_fft * cp.conj(template_fft)

        # Inverse FFT
        corr_raw = cufft.ifft(corr_fft)

        # Extract valid portion (linear correlation length)
        valid_len = len(signal) - len(template) + 1
        corr = corr_raw[:valid_len]

        # Normalize to NCC if requested
        if self.config.normalize:
            corr = self._normalize_ncc_gpu(corr, template_gpu, signal_gpu, valid_len)

        # Return magnitude or complex
        if self.config.use_magnitude:
            result = cp.abs(corr)
        else:
            result = corr

        return cp.asnumpy(result).astype(np.float32 if self.config.use_magnitude else np.complex64)

    def _correlate_chunked(
        self,
        template: np.ndarray,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Chunked correlation for large signals (GPU memory management)."""
        template_len = len(template)
        signal_len = len(signal)
        chunk_size = self.config.chunk_size

        # Overlap must be >= template_len to avoid missing detections
        overlap = template_len

        # Output length
        output_len = signal_len - template_len + 1
        result = np.zeros(output_len, dtype=np.float32)

        # Process chunks
        start = 0
        while start < signal_len - template_len:
            # Chunk end (with margin for valid correlation)
            end = min(start + chunk_size, signal_len)

            # Extract chunk
            chunk = signal[start:end]

            # Correlate
            chunk_corr = self._correlate_single(template, chunk)

            # Map back to output
            out_start = start
            out_end = out_start + len(chunk_corr)

            # Use max in overlap regions
            result[out_start:out_end] = np.maximum(result[out_start:out_end], chunk_corr)

            # Next chunk (with overlap)
            start = end - overlap

            # Prevent infinite loop
            if end >= signal_len:
                break

        return result

    def _normalize_ncc_gpu(
        self,
        corr: cp.ndarray,
        template: cp.ndarray,
        signal: cp.ndarray,
        valid_len: int,
    ) -> cp.ndarray:
        """
        Normalize correlation to NCC range.

        NCC = Corr / (σ_signal_local × σ_template × sqrt(n))

        For efficiency, we use global signal std rather than sliding window.
        This is an approximation but works well when signal has consistent noise.
        """
        template_len = len(template)

        # Template energy (constant)
        template_energy = cp.sqrt(cp.sum(cp.abs(template) ** 2))

        # Signal energy (use global std as approximation)
        signal_std = cp.std(cp.abs(signal))

        # Normalization factor
        # sqrt(template_len) accounts for sum length
        norm_factor = template_energy * signal_std * cp.sqrt(float(template_len))

        if norm_factor > 1e-10:
            return corr / norm_factor
        return corr


def fft_correlate_normalized(
    template: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """
    Convenience function for normalized FFT correlation.

    Args:
        template: Template to search for (complex)
        signal: Signal to search in (complex)

    Returns:
        Correlation magnitude, normalized to approximately [0, 1]

    Example:
        corr = fft_correlate_normalized(dcm_template, dcm_signal)
        peaks = np.where(corr > 0.5)[0]  # Find matches
    """
    correlator = FFTCorrelator()
    return correlator.correlate(template, signal)


def fft_correlate_multi(
    templates: list[np.ndarray],
    signal: np.ndarray,
) -> np.ndarray:
    """
    Convenience function for multi-template correlation.

    Takes maximum correlation across all templates at each position.

    Args:
        templates: List of templates to search for
        signal: Signal to search in

    Returns:
        Maximum correlation across templates
    """
    correlator = FFTCorrelator()
    return correlator.correlate_multi(templates, signal, reduction="max")


# Self-test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing FFT Correlator module...")
    require_gpu()

    # Generate test signal with embedded template
    fs = 1e6  # 1 MHz
    duration = 0.01  # 10 ms
    n_samples = int(fs * duration)

    # Create template: short burst
    template_len = 1000
    t_template = np.arange(template_len) / fs
    template = np.exp(2j * np.pi * 50e3 * t_template).astype(np.complex64)

    # Create signal: noise with embedded template at known position
    signal = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
    signal *= 0.1  # Low noise

    # Embed template at position 3000
    embed_pos = 3000
    signal[embed_pos : embed_pos + template_len] += template

    # Correlate
    correlator = FFTCorrelator()
    corr = correlator.correlate(template, signal)

    # Find peak
    peak_pos = np.argmax(corr)
    peak_val = corr[peak_pos]

    print(f"Template length: {template_len}")
    print(f"Signal length: {n_samples}")
    print(f"Correlation length: {len(corr)}")
    print(f"Embedded at: {embed_pos}")
    print(f"Peak found at: {peak_pos}")
    print(f"Peak value: {peak_val:.4f}")
    print(f"Position error: {abs(peak_pos - embed_pos)} samples")

    if abs(peak_pos - embed_pos) <= 5:
        print("✓ Correlation correctly found embedded signal")
    else:
        print("✗ Position error too large!")
