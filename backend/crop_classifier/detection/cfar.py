"""
CFAR (Constant False Alarm Rate) Detector.

Adapts detection threshold to local noise estimate, maintaining constant Pfa.

Algorithm:
    For each test cell i:
    1. Estimate noise from reference cells (excluding guard cells around i)
    2. Compute threshold: T = α × noise_estimate
    3. Detect if |corr[i]|² > T

Where α is set based on desired Pfa:
    α = chi2_ppf(1 - Pfa, 2×N_ref) / (2×N_ref)

Includes Non-Maximum Suppression (NMS) to get one detection per signal.

GPU REQUIRED: Uses CuPy for real-time performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from .dcm import require_gpu

logger = logging.getLogger(__name__)

# Import CuPy
try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass
class CFARConfig:
    """Configuration for CFAR detector."""

    pfa: float = 0.001
    """Probability of false alarm. Lower = fewer false detections."""

    guard_cells: int = 5
    """Guard cells on each side of test cell (excluded from noise estimate)."""

    reference_cells: int = 20
    """Reference cells on each side for noise estimation."""

    min_separation: int = 0
    """Minimum samples between detections (for NMS). If 0, uses template length."""

    use_ca_cfar: bool = True
    """Use Cell-Averaging CFAR. If False, uses Greatest-Of (GO-CFAR)."""

    def __post_init__(self):
        if not 0 < self.pfa < 1:
            raise ValueError(f"Pfa must be in (0, 1), got {self.pfa}")
        if self.guard_cells < 0:
            raise ValueError("guard_cells must be >= 0")
        if self.reference_cells < 1:
            raise ValueError("reference_cells must be >= 1")


@dataclass
class Detection:
    """Single detection result."""

    index: int
    """Sample index in correlation array."""

    score: float
    """Correlation score at this position."""

    threshold: float
    """Local adaptive threshold used."""

    snr_db: float = 0.0
    """Estimated SNR in dB (score/threshold in dB)."""

    def __post_init__(self):
        if self.threshold > 0:
            self.snr_db = 10 * np.log10(self.score / self.threshold)


@dataclass
class CFARResult:
    """Result from CFAR detection."""

    detections: list[Detection] = field(default_factory=list)
    """List of detections after NMS."""

    raw_count: int = 0
    """Number of detections before NMS."""

    threshold_mean: float = 0.0
    """Average threshold across signal."""

    @property
    def count(self) -> int:
        """Number of final detections."""
        return len(self.detections)

    @property
    def indices(self) -> np.ndarray:
        """Detection indices as array."""
        return np.array([d.index for d in self.detections])

    @property
    def scores(self) -> np.ndarray:
        """Detection scores as array."""
        return np.array([d.score for d in self.detections])


class CFARDetector:
    """
    GPU-accelerated CFAR detector with NMS.

    Example:
        detector = CFARDetector()
        result = detector.detect(correlation, min_separation=1000)

        for det in result.detections:
            print(f"Detection at {det.index}, score={det.score:.3f}")
    """

    def __init__(self, config: CFARConfig | None = None):
        require_gpu()
        self.config = config or CFARConfig()

        # Compute threshold multiplier from Pfa
        # For CA-CFAR with N reference cells, detection follows chi-squared distribution
        n_ref = 2 * self.config.reference_cells
        self._alpha = scipy_stats.chi2.ppf(1 - self.config.pfa, 2 * n_ref) / (2 * n_ref)

        logger.info(
            f"[CFAR] Initialized: Pfa={self.config.pfa}, "
            f"guard={self.config.guard_cells}, ref={self.config.reference_cells}, "
            f"α={self._alpha:.3f}"
        )

    def detect(
        self,
        correlation: np.ndarray,
        min_separation: int = 0,
    ) -> CFARResult:
        """
        Detect signals using CFAR with NMS.

        Args:
            correlation: Correlation magnitude array (output from FFTCorrelator)
            min_separation: Minimum samples between detections.
                           If 0, uses config.min_separation.

        Returns:
            CFARResult with detections
        """
        if len(correlation) == 0:
            return CFARResult()

        min_sep = min_separation if min_separation > 0 else self.config.min_separation

        # Compute adaptive threshold
        threshold = self._compute_threshold_gpu(correlation)

        # Find detections (above threshold)
        detection_mask = correlation > threshold
        detection_indices = np.where(detection_mask)[0]

        raw_count = len(detection_indices)

        if raw_count == 0:
            return CFARResult(raw_count=0, threshold_mean=float(np.mean(threshold)))

        # Apply NMS
        nms_indices = self._non_max_suppression(correlation, detection_indices, min_sep)

        # Build detection objects
        detections = []
        for idx in nms_indices:
            det = Detection(
                index=int(idx), score=float(correlation[idx]), threshold=float(threshold[idx])
            )
            detections.append(det)

        # Sort by score descending
        detections.sort(key=lambda d: d.score, reverse=True)

        return CFARResult(
            detections=detections, raw_count=raw_count, threshold_mean=float(np.mean(threshold))
        )

    def _compute_threshold_gpu(self, correlation: np.ndarray) -> np.ndarray:
        """Compute adaptive threshold using CA-CFAR on GPU."""
        corr_gpu = cp.asarray(correlation, dtype=cp.float32)
        n = len(correlation)

        guard = self.config.guard_cells
        ref = self.config.reference_cells
        window_half = guard + ref

        # Initialize threshold array
        threshold = cp.zeros(n, dtype=cp.float32)

        # Power (squared magnitude)
        power = corr_gpu**2

        # Compute local noise estimate using sliding window
        # This is a simplified implementation - could optimize with cumsum
        for i in range(n):
            # Reference cells on left
            left_start = max(0, i - window_half)
            left_end = max(0, i - guard)

            # Reference cells on right
            right_start = min(n, i + guard + 1)
            right_end = min(n, i + window_half + 1)

            # Gather reference cells
            left_refs = power[left_start:left_end] if left_end > left_start else cp.array([])
            right_refs = power[right_start:right_end] if right_end > right_start else cp.array([])

            # Combine
            if len(left_refs) > 0 and len(right_refs) > 0:
                all_refs = cp.concatenate([left_refs, right_refs])
            elif len(left_refs) > 0:
                all_refs = left_refs
            elif len(right_refs) > 0:
                all_refs = right_refs
            else:
                # Edge case: no reference cells
                all_refs = cp.array([cp.mean(power)])

            # Noise estimate
            if self.config.use_ca_cfar:
                noise_est = cp.mean(all_refs)
            else:
                # GO-CFAR: greatest of left/right
                left_mean = cp.mean(left_refs) if len(left_refs) > 0 else 0
                right_mean = cp.mean(right_refs) if len(right_refs) > 0 else 0
                noise_est = cp.maximum(left_mean, right_mean)

            # Threshold = α × noise_estimate
            threshold[i] = self._alpha * noise_est

        # Convert threshold from power to amplitude
        threshold = cp.sqrt(threshold)

        return cp.asnumpy(threshold)

    def _non_max_suppression(
        self,
        correlation: np.ndarray,
        indices: np.ndarray,
        min_separation: int,
    ) -> list[int]:
        """
        Non-maximum suppression to get one detection per signal.

        Greedy algorithm: pick highest, suppress neighbors, repeat.
        """
        if len(indices) == 0:
            return []

        if min_separation <= 0:
            # No NMS - return all
            return list(indices)

        # Get scores at detection indices
        scores = correlation[indices]

        # Sort by score descending
        sorted_order = np.argsort(scores)[::-1]
        sorted_indices = indices[sorted_order]

        # Greedy selection
        selected = []
        suppressed = set()

        for idx in sorted_indices:
            if idx in suppressed:
                continue

            selected.append(idx)

            # Suppress neighbors within min_separation
            for other in sorted_indices:
                if abs(other - idx) <= min_separation:
                    suppressed.add(other)

        return selected


def cfar_detect(
    correlation: np.ndarray,
    pfa: float = 0.001,
    min_separation: int = 0,
) -> CFARResult:
    """
    Convenience function for CFAR detection.

    Args:
        correlation: Correlation magnitude array
        pfa: Probability of false alarm
        min_separation: Min samples between detections (NMS)

    Returns:
        CFARResult with detections

    Example:
        result = cfar_detect(corr, pfa=0.001, min_separation=1000)
        print(f"Found {result.count} signals")
    """
    config = CFARConfig(pfa=pfa)
    detector = CFARDetector(config)
    return detector.detect(correlation, min_separation=min_separation)


def compute_pfa_threshold(
    correlation: np.ndarray,
    pfa: float = 0.001,
) -> float:
    """
    Compute global threshold for desired Pfa.

    Simpler than full CFAR - uses global noise estimate.

    Args:
        correlation: Correlation magnitude array
        pfa: Desired probability of false alarm

    Returns:
        Threshold value
    """
    # Estimate noise from lower percentile (avoid signal peaks)
    noise_estimate = np.percentile(correlation, 50)  # Median

    # Threshold for Pfa (assuming Rayleigh distribution for noise)
    # P(x > T) = exp(-T²/2σ²) = Pfa
    # T = σ × sqrt(-2 × ln(Pfa))
    threshold = noise_estimate * np.sqrt(-2 * np.log(pfa))

    return threshold


# Self-test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing CFAR Detector module...")
    require_gpu()

    # Generate test correlation with known peaks
    n = 10000
    noise = np.abs(np.random.randn(n) + 1j * np.random.randn(n)).astype(np.float32)
    noise *= 0.1  # Scale noise

    # Add peaks at known positions
    peak_positions = [1000, 3000, 5000, 7000]
    peak_values = [0.9, 0.8, 0.7, 0.95]

    correlation = noise.copy()
    for pos, val in zip(peak_positions, peak_values, strict=False):
        correlation[pos] = val

    # Detect
    detector = CFARDetector(CFARConfig(pfa=0.001))
    result = detector.detect(correlation, min_separation=500)

    print(f"Raw detections: {result.raw_count}")
    print(f"After NMS: {result.count}")
    print(f"Average threshold: {result.threshold_mean:.4f}")

    print("\nDetections:")
    for det in result.detections:
        print(f"  Index {det.index}: score={det.score:.3f}, SNR={det.snr_db:.1f}dB")

    # Check we found the peaks
    found_peaks = set(result.indices)
    expected_peaks = set(peak_positions)

    # Allow some tolerance
    matches = 0
    for expected in expected_peaks:
        for found in found_peaks:
            if abs(found - expected) <= 5:
                matches += 1
                break

    print(f"\nPeaks found: {matches}/{len(peak_positions)}")

    if matches == len(peak_positions):
        print("✓ CFAR correctly detected all embedded signals")
    else:
        print("✗ Some peaks missed")

    # Test Pfa accuracy on pure noise
    print("\nTesting Pfa accuracy on pure noise...")
    noise_only = np.abs(np.random.randn(100000) + 1j * np.random.randn(100000)).astype(np.float32)
    noise_only *= 0.1

    result_noise = detector.detect(noise_only, min_separation=0)
    actual_pfa = result_noise.raw_count / len(noise_only)

    print(f"Target Pfa: {detector.config.pfa}")
    print(f"Actual Pfa: {actual_pfa:.6f}")
    print(f"Ratio: {actual_pfa / detector.config.pfa:.2f}x")
