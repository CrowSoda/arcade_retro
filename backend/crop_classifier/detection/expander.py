"""
Signal Expander - Pipeline orchestration for label expansion.

Takes user's ~20 labeled signals and finds all similar instances in recording.

Flow:
    1. Extract IQ templates from user's labeled time bounds
    2. Resample templates to median length
    3. Apply DCM preprocessing to templates and full recording
    4. FFT correlate templates against recording
    5. CFAR detect with NMS
    6. Apply diversity sampling (optional)
    7. Return ranked candidates

GPU REQUIRED: Uses CuPy for real-time performance.
Target: 60s recording processed in <60s on Jetson Orin.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import signal as scipy_signal

from .cfar import CFARConfig, CFARDetector, CFARResult
from .correlator import CorrelatorConfig, FFTCorrelator
from .dcm import DCMConfig, DCMProcessor, require_gpu

logger = logging.getLogger(__name__)


@dataclass
class ExpanderConfig:
    """Configuration for signal expander."""

    # DCM settings
    dcm_delays: tuple[int, ...] = (1, 2, 3, 4, 5)
    """DCM delays for ~7dB SNR improvement."""

    # CFAR settings
    pfa: float = 0.001
    """Probability of false alarm."""

    # Results
    top_k: int = 200
    """Maximum candidates to return."""

    # Diversity sampling
    diversity_sampling: bool = True
    """Enable diversity sampling to avoid overfitting on clustered detections."""

    diversity_bins: int = 10
    """Number of time bins for diversity sampling."""

    # Auto-accept/reject thresholds
    auto_accept_threshold: float = 0.85
    """Correlation score above this is auto-accepted."""

    auto_reject_threshold: float = 0.15
    """Correlation score below this is auto-rejected."""

    # Template handling
    resample_templates: bool = True
    """Resample all templates to median length."""

    template_length_tolerance: float = 2.0
    """Warn if template lengths vary by more than this factor."""


@dataclass
class DetectionCandidate:
    """Single expansion candidate."""

    t_start: float
    """Start time in seconds."""

    t_end: float
    """End time in seconds."""

    score: float
    """Correlation score (0-1)."""

    sample_index: int
    """Sample index in recording."""

    snr_db: float = 0.0
    """Estimated SNR in dB."""

    time_bin: int = 0
    """Diversity bin (for UI grouping)."""

    auto_decision: str = "review"
    """'accept', 'reject', or 'review' based on thresholds."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "score": self.score,
            "sample_index": self.sample_index,
            "snr_db": self.snr_db,
            "time_bin": self.time_bin,
            "auto_decision": self.auto_decision,
        }


@dataclass
class ExpansionResult:
    """Result from signal expansion."""

    candidates: list[DetectionCandidate] = field(default_factory=list)
    """Detection candidates sorted by score."""

    templates_used: int = 0
    """Number of templates from user labels."""

    template_length_samples: int = 0
    """Median template length in samples."""

    total_detections: int = 0
    """Total detections before filtering."""

    processing_time_ms: float = 0.0
    """Total processing time."""

    # Statistics
    auto_accepted: int = 0
    """Count of auto-accepted (high confidence)."""

    auto_rejected: int = 0
    """Count of auto-rejected (low confidence)."""

    need_review: int = 0
    """Count needing manual review."""

    @property
    def count(self) -> int:
        """Number of candidates returned."""
        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "templates_used": self.templates_used,
            "template_length_samples": self.template_length_samples,
            "total_detections": self.total_detections,
            "processing_time_ms": self.processing_time_ms,
            "auto_accepted": self.auto_accepted,
            "auto_rejected": self.auto_rejected,
            "need_review": self.need_review,
            "count": self.count,
        }


class SignalExpander:
    """
    GPU-accelerated signal expansion pipeline.

    Example:
        expander = SignalExpander()

        # User's labeled boxes: [{t_start, t_end}, ...]
        seed_boxes = [
            {"t_start": 0.001, "t_end": 0.002},
            {"t_start": 0.015, "t_end": 0.016},
        ]

        result = expander.expand(
            rfcap_path="/path/to/file.rfcap",
            seed_boxes=seed_boxes,
        )

        # Send to swipe UI
        for candidate in result.candidates:
            if candidate.auto_decision == "review":
                show_to_user(candidate)
    """

    def __init__(self, config: ExpanderConfig | None = None):
        require_gpu()
        self.config = config or ExpanderConfig()

        # Initialize sub-components
        self.dcm = DCMProcessor(
            DCMConfig(
                delays=self.config.dcm_delays,
                normalize=True,
            )
        )

        self.correlator = FFTCorrelator(
            CorrelatorConfig(
                normalize=True,
                use_magnitude=True,
            )
        )

        self.cfar = CFARDetector(
            CFARConfig(
                pfa=self.config.pfa,
            )
        )

        logger.info(f"[Expander] Initialized with pfa={self.config.pfa}, top_k={self.config.top_k}")

    def expand(
        self,
        rfcap_path: str,
        seed_boxes: list[dict[str, float]],
        sample_rate: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ExpansionResult:
        """
        Expand user labels to find all similar signals.

        Args:
            rfcap_path: Path to RFCAP file
            seed_boxes: User's labeled time bounds [{t_start, t_end}, ...]
            sample_rate: Override sample rate (otherwise read from header)
            progress_callback: Called with (progress 0-1, status message)

        Returns:
            ExpansionResult with ranked candidates
        """
        start_time = time.time()

        def report(progress: float, msg: str):
            logger.info(f"[Expander] {msg}")
            if progress_callback:
                progress_callback(progress, msg)

        report(0.0, "Loading RFCAP file...")

        # 1. Load RFCAP
        header, iq_data = self._load_rfcap(rfcap_path)
        fs = sample_rate or header["sample_rate"]
        duration = len(iq_data) / fs

        report(0.1, f"Loaded {duration:.1f}s recording at {fs/1e6:.1f} MHz")

        # 2. Extract templates from seed boxes
        report(0.15, f"Extracting {len(seed_boxes)} templates...")
        templates = self._extract_templates(iq_data, seed_boxes, fs)

        if not templates:
            return ExpansionResult(processing_time_ms=(time.time() - start_time) * 1000)

        # Check template length variation
        lengths = [len(t) for t in templates]
        if max(lengths) / max(min(lengths), 1) > self.config.template_length_tolerance:
            logger.warning(
                f"[Expander] Template lengths vary significantly: "
                f"{min(lengths)} - {max(lengths)} samples"
            )

        # 3. Resample templates to median length
        if self.config.resample_templates:
            report(0.2, "Resampling templates...")
            median_len = int(np.median(lengths))
            templates = [self._resample(t, median_len) for t in templates]
        else:
            median_len = int(np.median(lengths))

        # 4. Apply DCM to templates
        report(0.25, "Applying DCM to templates...")
        dcm_templates = self.dcm.process_batch(templates)

        # 5. Apply DCM to full recording
        report(0.3, "Applying DCM to recording...")
        dcm_signal = self.dcm.process(iq_data)

        # 6. Correlate templates against recording
        report(0.5, "Running FFT correlation...")
        correlation = self.correlator.correlate_multi(dcm_templates, dcm_signal)

        # 7. CFAR detection
        report(0.7, "Running CFAR detection...")
        min_separation = median_len // 2  # Half template length
        cfar_result = self.cfar.detect(correlation, min_separation=min_separation)

        report(0.8, f"Found {cfar_result.count} detections")

        # 8. Convert to candidates
        template_duration = median_len / fs
        candidates = self._build_candidates(
            cfar_result,
            fs,
            template_duration,
            duration,
        )

        # 9. Filter overlaps with original seed boxes
        candidates = self._filter_seed_overlaps(candidates, seed_boxes)

        # 10. Apply diversity sampling
        if self.config.diversity_sampling:
            report(0.85, "Applying diversity sampling...")
            candidates = self._diversity_sample(candidates, duration)

        # 11. Sort by score and limit to top_k
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[: self.config.top_k]

        # 12. Categorize by auto-decision
        auto_accepted = sum(1 for c in candidates if c.auto_decision == "accept")
        auto_rejected = sum(1 for c in candidates if c.auto_decision == "reject")
        need_review = sum(1 for c in candidates if c.auto_decision == "review")

        processing_time_ms = (time.time() - start_time) * 1000

        report(1.0, f"Complete: {len(candidates)} candidates in {processing_time_ms:.0f}ms")

        return ExpansionResult(
            candidates=candidates,
            templates_used=len(templates),
            template_length_samples=median_len,
            total_detections=cfar_result.count,
            processing_time_ms=processing_time_ms,
            auto_accepted=auto_accepted,
            auto_rejected=auto_rejected,
            need_review=need_review,
        )

    def expand_from_iq(
        self,
        iq_data: np.ndarray,
        seed_boxes: list[dict[str, float]],
        sample_rate: float,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ExpansionResult:
        """
        Expand from pre-loaded IQ data (avoids file I/O).

        Args:
            iq_data: Complex IQ samples
            seed_boxes: User's labeled time bounds
            sample_rate: Sample rate in Hz
            progress_callback: Progress callback

        Returns:
            ExpansionResult
        """
        start_time = time.time()

        def report(progress: float, msg: str):
            logger.info(f"[Expander] {msg}")
            if progress_callback:
                progress_callback(progress, msg)

        fs = sample_rate
        duration = len(iq_data) / fs

        report(0.1, f"Processing {duration:.1f}s of IQ data")

        # Extract templates
        templates = self._extract_templates(iq_data, seed_boxes, fs)
        if not templates:
            return ExpansionResult(processing_time_ms=(time.time() - start_time) * 1000)

        # Resample
        lengths = [len(t) for t in templates]
        median_len = int(np.median(lengths))
        if self.config.resample_templates:
            templates = [self._resample(t, median_len) for t in templates]

        # DCM
        report(0.3, "Applying DCM...")
        dcm_templates = self.dcm.process_batch(templates)
        dcm_signal = self.dcm.process(iq_data)

        # Correlate
        report(0.5, "Correlating...")
        correlation = self.correlator.correlate_multi(dcm_templates, dcm_signal)

        # CFAR
        report(0.7, "CFAR detection...")
        min_separation = median_len // 2
        cfar_result = self.cfar.detect(correlation, min_separation=min_separation)

        # Build candidates
        template_duration = median_len / fs
        candidates = self._build_candidates(cfar_result, fs, template_duration, duration)
        candidates = self._filter_seed_overlaps(candidates, seed_boxes)

        if self.config.diversity_sampling:
            candidates = self._diversity_sample(candidates, duration)

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[: self.config.top_k]

        auto_accepted = sum(1 for c in candidates if c.auto_decision == "accept")
        auto_rejected = sum(1 for c in candidates if c.auto_decision == "reject")
        need_review = sum(1 for c in candidates if c.auto_decision == "review")

        processing_time_ms = (time.time() - start_time) * 1000

        return ExpansionResult(
            candidates=candidates,
            templates_used=len(templates),
            template_length_samples=median_len,
            total_detections=cfar_result.count,
            processing_time_ms=processing_time_ms,
            auto_accepted=auto_accepted,
            auto_rejected=auto_rejected,
            need_review=need_review,
        )

    def _load_rfcap(self, path: str) -> tuple[dict, np.ndarray]:
        """Load RFCAP file header and IQ data."""
        import struct

        with open(path, "rb") as f:
            # Read header (512 bytes)
            magic = f.read(4)
            if magic != b"G20\x00":
                raise ValueError(f"Invalid RFCAP magic: {magic!r}")

            version = struct.unpack("<I", f.read(4))[0]
            sample_rate = struct.unpack("<d", f.read(8))[0]
            center_freq = struct.unpack("<d", f.read(8))[0]
            bandwidth = struct.unpack("<d", f.read(8))[0]
            num_samples = struct.unpack("<Q", f.read(8))[0]

            header = {
                "version": version,
                "sample_rate": sample_rate,
                "center_freq": center_freq,
                "bandwidth": bandwidth,
                "num_samples": num_samples,
            }

            # Skip rest of header
            f.seek(512)

            # Read IQ data
            raw = f.read(num_samples * 8)
            iq_data = np.frombuffer(raw, dtype=np.complex64)

        return header, iq_data

    def _extract_templates(
        self,
        iq_data: np.ndarray,
        seed_boxes: list[dict[str, float]],
        sample_rate: float,
    ) -> list[np.ndarray]:
        """Extract IQ segments for each seed box."""
        templates = []

        for box in seed_boxes:
            t_start = box.get("t_start", 0)
            t_end = box.get("t_end", t_start + 0.001)

            start_idx = int(t_start * sample_rate)
            end_idx = int(t_end * sample_rate)

            # Bounds check
            start_idx = max(0, start_idx)
            end_idx = min(len(iq_data), end_idx)

            if end_idx > start_idx:
                segment = iq_data[start_idx:end_idx].copy()
                if len(segment) > self.dcm._max_delay:  # Enough for DCM
                    templates.append(segment)

        return templates

    def _resample(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """Resample signal to target length."""
        if len(signal) == target_len:
            return signal

        # Use scipy's resample for complex signals
        return scipy_signal.resample(signal, target_len).astype(np.complex64)

    def _build_candidates(
        self,
        cfar_result: CFARResult,
        sample_rate: float,
        template_duration: float,
        total_duration: float,
    ) -> list[DetectionCandidate]:
        """Convert CFAR detections to candidates."""
        candidates = []

        for det in cfar_result.detections:
            t_start = det.index / sample_rate
            t_end = t_start + template_duration

            # Skip if at recording boundary
            if t_end > total_duration:
                continue

            # Determine auto-decision
            if det.score >= self.config.auto_accept_threshold:
                auto_decision = "accept"
            elif det.score <= self.config.auto_reject_threshold:
                auto_decision = "reject"
            else:
                auto_decision = "review"

            candidate = DetectionCandidate(
                t_start=t_start,
                t_end=t_end,
                score=det.score,
                sample_index=det.index,
                snr_db=det.snr_db,
                auto_decision=auto_decision,
            )
            candidates.append(candidate)

        return candidates

    def _filter_seed_overlaps(
        self,
        candidates: list[DetectionCandidate],
        seed_boxes: list[dict[str, float]],
    ) -> list[DetectionCandidate]:
        """Remove candidates that overlap with original seeds."""
        filtered = []

        for candidate in candidates:
            overlaps = False
            for seed in seed_boxes:
                seed_start = seed.get("t_start", 0)
                seed_end = seed.get("t_end", seed_start + 0.001)

                # Check overlap
                if not (candidate.t_end <= seed_start or candidate.t_start >= seed_end):
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(candidate)

        return filtered

    def _diversity_sample(
        self,
        candidates: list[DetectionCandidate],
        duration: float,
    ) -> list[DetectionCandidate]:
        """
        Apply diversity sampling to avoid clustered detections.

        Divides recording into time bins and ensures representation from each.
        """
        if not candidates:
            return candidates

        n_bins = self.config.diversity_bins
        bin_size = duration / n_bins

        # Assign time bins
        for c in candidates:
            c.time_bin = min(int(c.t_start / bin_size), n_bins - 1)

        # Group by bin
        bins: dict[int, list[DetectionCandidate]] = {i: [] for i in range(n_bins)}
        for c in candidates:
            bins[c.time_bin].append(c)

        # Sort each bin by score
        for bin_candidates in bins.values():
            bin_candidates.sort(key=lambda c: c.score, reverse=True)

        # Interleave: take top from each bin in round-robin
        result = []
        max_per_bin = (self.config.top_k // n_bins) + 1

        for round_idx in range(max_per_bin):
            for bin_idx in range(n_bins):
                if round_idx < len(bins[bin_idx]):
                    result.append(bins[bin_idx][round_idx])

        return result


# Convenience function
def expand_labels(
    rfcap_path: str,
    seed_boxes: list[dict[str, float]],
    pfa: float = 0.001,
    top_k: int = 200,
    diversity_sampling: bool = True,
) -> ExpansionResult:
    """
    Convenience function for label expansion.

    Args:
        rfcap_path: Path to RFCAP file
        seed_boxes: User's labeled time bounds [{t_start, t_end}, ...]
        pfa: Probability of false alarm
        top_k: Maximum candidates to return
        diversity_sampling: Enable diversity sampling

    Returns:
        ExpansionResult with ranked candidates

    Example:
        result = expand_labels(
            "/path/to/file.rfcap",
            [{"t_start": 0.001, "t_end": 0.002}],
        )
        print(f"Found {result.count} candidates")
    """
    config = ExpanderConfig(
        pfa=pfa,
        top_k=top_k,
        diversity_sampling=diversity_sampling,
    )
    expander = SignalExpander(config)
    return expander.expand(rfcap_path, seed_boxes)


# Self-test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Signal Expander module...")
    require_gpu()

    # Create synthetic test data
    fs = 1e6  # 1 MHz
    duration = 1.0  # 1 second
    n_samples = int(fs * duration)

    # Generate noise
    iq_data = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
    iq_data *= 0.1

    # Add signals at known positions
    signal_positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # seconds
    signal_duration = 0.005  # 5ms each
    signal_freq = 100e3  # 100 kHz tone

    for pos in signal_positions:
        start_idx = int(pos * fs)
        n_signal = int(signal_duration * fs)
        t = np.arange(n_signal) / fs
        signal = np.exp(2j * np.pi * signal_freq * t).astype(np.complex64)
        iq_data[start_idx : start_idx + n_signal] += signal

    # Create "user labels" - only first 2 signals
    seed_boxes = [
        {"t_start": signal_positions[0], "t_end": signal_positions[0] + signal_duration},
        {"t_start": signal_positions[1], "t_end": signal_positions[1] + signal_duration},
    ]

    print("\nTest setup:")
    print(f"  Recording: {duration}s at {fs/1e6} MHz")
    print(f"  Signals embedded at: {signal_positions}")
    print(f"  User labeled: {signal_positions[:2]}")
    print(f"  Should find: {signal_positions[2:]}")

    # Run expansion
    expander = SignalExpander(
        ExpanderConfig(
            pfa=0.001,
            top_k=10,
            diversity_sampling=False,  # Disable for test
        )
    )

    result = expander.expand_from_iq(iq_data, seed_boxes, fs)

    print("\nResults:")
    print(f"  Templates used: {result.templates_used}")
    print(f"  Total detections: {result.total_detections}")
    print(f"  Candidates returned: {result.count}")
    print(f"  Processing time: {result.processing_time_ms:.0f}ms")

    print("\nCandidates:")
    for c in result.candidates[:5]:
        print(f"  t={c.t_start:.3f}s, score={c.score:.3f}, decision={c.auto_decision}")

    # Check if we found the unlabeled signals
    expected = set(signal_positions[2:])
    found = set()
    for c in result.candidates:
        for exp in expected:
            if abs(c.t_start - exp) < 0.01:  # Within 10ms
                found.add(exp)

    print(f"\nFound {len(found)}/{len(expected)} unlabeled signals")

    if found == expected:
        print("✓ Expander correctly found all unlabeled signals")
    else:
        missing = expected - found
        print(f"✗ Missing signals at: {missing}")
