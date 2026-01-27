"""
Sub-band extraction for CNN training data.

Implements proper DSP pipeline: mix → filter → decimate
with 60-80 dB stopband attenuation for clean training data.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

# Logging
from logger_config import get_logger
from scipy.signal import firwin, kaiserord, resample_poly

logger = get_logger("subband_extractor")


@dataclass
class ExtractionParams:
    """Parameters for sub-band extraction."""

    source_rate: float  # Hz, e.g., 20e6
    center_offset: float  # Hz from source center
    target_bandwidth: float  # Hz, signal bandwidth
    target_rate: float | None = None  # Hz, None = auto (2.5x BW)
    stopband_db: float = 60.0  # Stopband attenuation
    remove_dc: bool = True  # Remove DC offset
    normalize: bool = True  # Normalize to unit power


@dataclass
class ExtractionResult:
    """Result of sub-band extraction."""

    iq_data: np.ndarray  # Extracted complex64 IQ data
    source_rate: float  # Hz, original sample rate
    output_rate: float  # Hz, actual output sample rate
    bandwidth: float  # Hz, target bandwidth
    center_offset: float  # Hz, frequency offset applied
    filter_taps: int  # Number of filter taps used
    decimation_ratio: float  # Effective decimation factor
    processing_time: float  # Seconds


class SubbandExtractor:
    """
    Extract narrowband sub-band from wideband IQ data.

    Usage:
        extractor = SubbandExtractor(ExtractionParams(...))
        result = extractor.extract(wideband_iq)

    NOTE: Uses resample_poly with custom FIR taps - this combines
    filtering and decimation in ONE operation (no double filtering!).
    """

    def __init__(self, params: ExtractionParams):
        self.params = params

        # Calculate target rate if not specified (2.5x bandwidth)
        if params.target_rate is None:
            self.target_rate = params.target_bandwidth * 2.5
        else:
            self.target_rate = params.target_rate

        # Ensure target rate is at least Nyquist
        min_rate = params.target_bandwidth * 2.0
        if self.target_rate < min_rate:
            logger.warning(
                f"[SubbandExtractor] WARNING: Target rate {self.target_rate / 1e6:.2f} MHz "
                f"below Nyquist ({min_rate / 1e6:.2f} MHz), adjusting to 2.5x BW"
            )
            self.target_rate = params.target_bandwidth * 2.5

        # Calculate decimation ratio
        self.decim_ratio = params.source_rate / self.target_rate

        # Find rational approximation for resampling
        ratio = Fraction(self.target_rate / params.source_rate).limit_denominator(100)
        self.up = ratio.numerator
        self.down = ratio.denominator

        # Check rate approximation error
        actual_rate = params.source_rate * self.up / self.down
        error_pct = abs(actual_rate - self.target_rate) / self.target_rate * 100
        if error_pct > 1.0:
            logger.error(f"[SubbandExtractor] WARNING: Rate approximation error {error_pct:.1f}%")
        self.actual_target_rate = actual_rate

        # Design anti-aliasing filter for resample_poly
        self.filter_taps = self._design_filter()

        logger.info("[SubbandExtractor] Initialized:")
        logger.info(f"  Source: {params.source_rate / 1e6:.2f} MHz")
        logger.info(f"  Target: {actual_rate / 1e6:.2f} MHz ({self.up}:{self.down})")
        logger.info(f"  Bandwidth: {params.target_bandwidth / 1e6:.2f} MHz")
        logger.info(f"  Filter: {len(self.filter_taps)} taps, {params.stopband_db} dB stopband")

    def _design_filter(self) -> np.ndarray:
        """
        Design anti-aliasing filter for resample_poly.

        IMPORTANT: Filter is designed at the INTERPOLATED rate (source × up),
        not at source rate. This is what resample_poly expects.
        """
        p = self.params

        # Cutoff = half of target bandwidth
        cutoff_hz = p.target_bandwidth / 2

        # Filter designed at interpolated rate (source × up factor)
        interp_rate = p.source_rate * self.up

        # Transition bandwidth = 10% of cutoff (sharper = more taps)
        transition_width = cutoff_hz * 0.1
        nyquist = interp_rate / 2

        # Normalize transition width for kaiserord
        normalized_transition = transition_width / nyquist

        # Kaiser window design for specified attenuation
        try:
            numtaps, beta = kaiserord(p.stopband_db, normalized_transition)
        except ValueError:
            # Fallback if transition width is too small
            numtaps = 255
            beta = 8.6  # Approx for 60 dB attenuation

        # Ensure odd number (linear phase)
        if numtaps % 2 == 0:
            numtaps += 1

        # Minimum 63 taps for reasonable response
        numtaps = max(numtaps, 63)

        # Maximum taps to prevent memory issues (can be tuned)
        numtaps = min(numtaps, 4095)

        # Normalized cutoff for firwin (relative to interpolated Nyquist)
        normalized_cutoff = cutoff_hz / nyquist
        normalized_cutoff = min(normalized_cutoff, 0.99)  # Must be < 1

        # Design FIR lowpass at interpolated rate
        taps = firwin(numtaps, normalized_cutoff, window=("kaiser", beta))

        return taps.astype(np.float32)

    def extract(
        self, iq_data: np.ndarray, progress_callback: Callable[[float], None] | None = None
    ) -> ExtractionResult:
        """
        Extract sub-band from wideband IQ data.

        Pipeline: DC removal → frequency shift → (filter + decimate) → normalize

        NOTE: Filter and decimate are COMBINED in resample_poly to avoid
        double filtering. Custom filter taps passed via window parameter.

        Args:
            iq_data: Complex64 IQ samples at source_rate
            progress_callback: Optional callback with progress 0-1

        Returns:
            ExtractionResult with extracted IQ and metadata
        """
        start_time = time.perf_counter()
        p = self.params

        if progress_callback:
            progress_callback(0.0)

        # Ensure complex64
        if iq_data.dtype != np.complex64:
            iq_data = iq_data.astype(np.complex64)

        # Step 1: DC offset removal (before frequency shift!)
        if p.remove_dc:
            iq_data = iq_data - np.mean(iq_data)

        if progress_callback:
            progress_callback(0.2)

        # Step 2: Frequency translation
        if p.center_offset != 0:
            t = np.arange(len(iq_data), dtype=np.float64) / p.source_rate
            nco = np.exp(-2j * np.pi * p.center_offset * t).astype(np.complex64)
            iq_data = iq_data * nco

        if progress_callback:
            progress_callback(0.4)

        # Step 3+4 COMBINED: Anti-alias + decimate in ONE operation
        # resample_poly accepts custom FIR taps via window parameter
        # This avoids double-filtering (don't use separate lfilter!)
        if self.up != 1 or self.down != 1:
            iq_decimated = resample_poly(iq_data, self.up, self.down, window=self.filter_taps)
        else:
            iq_decimated = iq_data

        if progress_callback:
            progress_callback(0.8)

        # Step 5: Normalize to unit power
        if p.normalize:
            power = np.mean(np.abs(iq_decimated) ** 2)
            if power > 0:
                iq_decimated = iq_decimated / np.sqrt(power)

        elapsed = time.perf_counter() - start_time

        if progress_callback:
            progress_callback(1.0)

        return ExtractionResult(
            iq_data=iq_decimated.astype(np.complex64),
            source_rate=p.source_rate,
            output_rate=self.actual_target_rate,
            bandwidth=p.target_bandwidth,
            center_offset=p.center_offset,
            filter_taps=len(self.filter_taps),
            decimation_ratio=self.decim_ratio,
            processing_time=elapsed,
        )

    def get_output_sample_count(self, input_samples: int) -> int:
        """Calculate output sample count for given input."""
        return int(input_samples * self.up / self.down)

    def get_filter_delay_samples(self) -> int:
        """Get filter group delay in output samples."""
        input_delay = (len(self.filter_taps) - 1) // 2
        return int(input_delay * self.up / self.down)

    def get_info(self) -> dict:
        """Get extractor configuration info."""
        return {
            "source_rate_hz": self.params.source_rate,
            "target_rate_hz": self.actual_target_rate,
            "target_bandwidth_hz": self.params.target_bandwidth,
            "center_offset_hz": self.params.center_offset,
            "decimation_ratio": self.decim_ratio,
            "resample_up": self.up,
            "resample_down": self.down,
            "filter_taps": len(self.filter_taps),
            "stopband_db": self.params.stopband_db,
            "remove_dc": self.params.remove_dc,
            "normalize": self.params.normalize,
        }


# Header format constants
RFCAP_HEADER_SIZE = 512

# G20 format (what Flutter uses) - this is the primary format
G20_MAGIC = b"G20\x00"

# Legacy RFCAP format (for backwards compatibility)
RFCAP_MAGIC = b"RFCAP\x00\x00\x00"


def _read_g20_header(filepath: str) -> dict:
    """
    Read G20 header (512 bytes) - the standard format used by Flutter.

    G20 Header Layout:
    | Offset | Size | Type     | Field              |
    |--------|------|----------|--------------------|
    | 0      | 4    | char[4]  | Magic ("G20\0")    |
    | 4      | 4    | uint32   | Version            |
    | 8      | 8    | float64  | Sample rate (Hz)   |
    | 16     | 8    | float64  | Center freq (Hz)   |
    | 24     | 8    | float64  | Bandwidth (Hz)     |
    | 32     | 8    | uint64   | Number of samples  |
    | 40     | 8    | float64  | Start time (epoch) |
    | 48     | 32   | char[32] | Signal name        |
    | 80     | 8    | float64  | Latitude           |
    | 88     | 8    | float64  | Longitude          |
    """
    import struct

    with open(filepath, "rb") as f:
        header = f.read(RFCAP_HEADER_SIZE)

    if len(header) < RFCAP_HEADER_SIZE:
        raise ValueError(f"File too small for G20 header: {len(header)} bytes")

    # Parse G20 header
    version = struct.unpack("<I", header[4:8])[0]
    sample_rate = struct.unpack("<d", header[8:16])[0]
    center_freq = struct.unpack("<d", header[16:24])[0]
    bandwidth = struct.unpack("<d", header[24:32])[0]
    num_samples = struct.unpack("<Q", header[32:40])[0]
    start_time = struct.unpack("<d", header[40:48])[0]

    # Signal name at offset 48, 32 bytes null-terminated
    signal_name_raw = header[48:80]
    signal_name = signal_name_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    # Lat/lon at offset 80
    latitude = struct.unpack("<d", header[80:88])[0]
    longitude = struct.unpack("<d", header[88:96])[0]

    return {
        "version": version,
        "sample_rate": sample_rate,
        "center_freq": center_freq,
        "bandwidth": bandwidth,
        "num_samples": num_samples,
        "signal_name": signal_name,
        "latitude": latitude,
        "longitude": longitude,
        "start_time": start_time,
    }


def _read_legacy_rfcap_header(filepath: str) -> dict:
    """Read legacy RFCAP header format (backwards compatibility)."""
    import struct

    with open(filepath, "rb") as f:
        header = f.read(RFCAP_HEADER_SIZE)

    if len(header) < RFCAP_HEADER_SIZE:
        raise ValueError(f"File too small for RFCAP header: {len(header)} bytes")

    # Parse legacy RFCAP header
    version = struct.unpack("<I", header[8:12])[0]
    sample_rate = struct.unpack("<d", header[12:20])[0]
    center_freq = struct.unpack("<d", header[20:28])[0]
    bandwidth = struct.unpack("<d", header[28:36])[0]
    num_samples = struct.unpack("<Q", header[32:40])[0]

    # Signal name at offset 40, 32 bytes null-terminated
    signal_name_raw = header[40:72]
    signal_name = signal_name_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    # Lat/lon at offset 72
    latitude = struct.unpack("<d", header[72:80])[0]
    longitude = struct.unpack("<d", header[80:88])[0]

    # Start time at offset 88
    start_time = struct.unpack("<d", header[88:96])[0]

    return {
        "version": version,
        "sample_rate": sample_rate,
        "center_freq": center_freq,
        "bandwidth": bandwidth,
        "num_samples": num_samples,
        "signal_name": signal_name,
        "latitude": latitude,
        "longitude": longitude,
        "start_time": start_time,
    }


def _read_rfcap_header(filepath: str) -> dict:
    """
    Read header from RFCAP file (auto-detects G20 vs legacy format).

    Returns metadata dict with sample_rate, center_freq, bandwidth, etc.
    """
    with open(filepath, "rb") as f:
        magic = f.read(8)

    # Check which format
    if magic[:4] == G20_MAGIC:
        logger.info(f"[SubbandExtractor] Detected G20 format: {filepath}")
        return _read_g20_header(filepath)
    elif magic == RFCAP_MAGIC:
        logger.info(f"[SubbandExtractor] Detected legacy RFCAP format: {filepath}")
        return _read_legacy_rfcap_header(filepath)
    else:
        raise ValueError(f"Unknown header format. Magic: {magic!r}. Expected G20 or RFCAP.")


def _read_iq_data(
    filepath: str, offset_samples: int = 0, num_samples: int | None = None
) -> np.ndarray:
    """Read IQ data from RFCAP file."""
    # Skip header, seek to data start
    with open(filepath, "rb") as f:
        f.seek(RFCAP_HEADER_SIZE + offset_samples * 8)  # 8 bytes per complex sample

        if num_samples is not None:
            raw = f.read(num_samples * 8)
        else:
            raw = f.read()

    if len(raw) == 0:
        return np.array([], dtype=np.complex64)

    # Parse as complex64
    return np.frombuffer(raw, dtype=np.complex64)


def _write_g20_header(filepath: str, metadata: dict):
    """
    Write G20 header (512 bytes) - matches Flutter's format.

    G20 Header Layout:
    | Offset | Size | Type     | Field              |
    |--------|------|----------|--------------------|
    | 0      | 4    | char[4]  | Magic ("G20\0")    |
    | 4      | 4    | uint32   | Version            |
    | 8      | 8    | float64  | Sample rate (Hz)   |
    | 16     | 8    | float64  | Center freq (Hz)   |
    | 24     | 8    | float64  | Bandwidth (Hz)     |
    | 32     | 8    | uint64   | Number of samples  |
    | 40     | 8    | float64  | Start time (epoch) |
    | 48     | 32   | char[32] | Signal name        |
    | 80     | 8    | float64  | Latitude           |
    | 88     | 8    | float64  | Longitude          |
    """
    import struct

    header = bytearray(RFCAP_HEADER_SIZE)

    # Magic: "G20\0"
    header[0:4] = G20_MAGIC

    # Version (offset 4)
    struct.pack_into("<I", header, 4, metadata.get("version", 1))

    # Sample rate (offset 8)
    struct.pack_into("<d", header, 8, metadata["sample_rate"])

    # Center freq (offset 16)
    struct.pack_into("<d", header, 16, metadata["center_freq"])

    # Bandwidth (offset 24)
    struct.pack_into("<d", header, 24, metadata["bandwidth"])

    # Num samples (offset 32)
    struct.pack_into("<Q", header, 32, metadata["num_samples"])

    # Start time (offset 40)
    struct.pack_into("<d", header, 40, metadata.get("start_time", 0.0))

    # Signal name (offset 48, 32 bytes, null-terminated)
    signal_name = metadata.get("signal_name", "UNKNOWN")[:31]
    signal_bytes = signal_name.encode("utf-8")[:31]
    header[48 : 48 + len(signal_bytes)] = signal_bytes

    # Latitude (offset 80)
    struct.pack_into("<d", header, 80, metadata.get("latitude", 0.0))

    # Longitude (offset 88)
    struct.pack_into("<d", header, 88, metadata.get("longitude", 0.0))

    with open(filepath, "wb") as f:
        f.write(header)

    logger.info(
        f"[SubbandExtractor] Wrote G20 header: sr={metadata['sample_rate'] / 1e6:.2f}MHz, "
        f"cf={metadata['center_freq'] / 1e6:.2f}MHz, bw={metadata['bandwidth'] / 1e6:.2f}MHz, "
        f"samples={metadata['num_samples']}"
    )


# Alias for backwards compatibility
def _write_rfcap_header(filepath: str, metadata: dict):
    """Write header in G20 format (standard format)."""
    _write_g20_header(filepath, metadata)


def extract_subband_from_file(
    source_path: str,
    output_path: str,
    center_offset_hz: float,
    bandwidth_hz: float,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    stopband_db: float = 60.0,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Extract sub-band from RFCAP file using CHUNKED processing.

    Progress is reported accurately across 3 phases:
    - Phase 1 (RED):    0-30%  - Reading & frequency shifting
    - Phase 2 (YELLOW): 30-85% - Resampling (the slow part)
    - Phase 3 (GREEN):  85-100% - Finalizing & writing

    Args:
        source_path: Path to source RFCAP file
        output_path: Path for output RFCAP file
        center_offset_hz: Frequency offset from source center
        bandwidth_hz: Target signal bandwidth
        start_sec: Start time in source file
        duration_sec: Duration to extract (None = all)
        stopband_db: Filter stopband attenuation
        progress_callback: Optional callback(progress: 0-1, phase: str)

    Returns:
        Dict with extraction metadata
    """
    start_time = time.perf_counter()

    def report(pct: float, phase: str = "working"):
        if progress_callback:
            progress_callback(pct, phase)

    # ===== PHASE 1: READ & SHIFT (0-30%) =====
    report(0.0, "reading")

    # Read source header
    source_header = _read_rfcap_header(source_path)
    sample_rate = source_header["sample_rate"]

    report(0.02, "reading")

    # Calculate sample offsets
    offset_samples = int(start_sec * sample_rate)
    if duration_sec is not None:
        num_samples = int(duration_sec * sample_rate)
    else:
        num_samples = source_header["num_samples"] - offset_samples

    # Read IQ data in one go (memory efficient enough for most files)
    iq_data = _read_iq_data(source_path, offset_samples, num_samples)

    if len(iq_data) == 0:
        raise ValueError("No data to extract")

    report(0.10, "reading")
    logger.info(f"[SubbandExtractor] Read {len(iq_data) / 1e6:.2f}M samples")

    # DC removal
    iq_data = iq_data - np.mean(iq_data)

    report(0.15, "shifting")

    # Frequency shift (if needed)
    if center_offset_hz != 0:
        t = np.arange(len(iq_data), dtype=np.float64) / sample_rate
        nco = np.exp(-2j * np.pi * center_offset_hz * t).astype(np.complex64)
        iq_data = iq_data * nco

    report(0.30, "shifting")
    logger.info("[SubbandExtractor] Phase 1 done: frequency shifted")

    # ===== PHASE 2: RESAMPLE (30-85%) - CHUNKED =====
    report(0.30, "resampling")

    # Calculate resampling parameters
    target_rate = bandwidth_hz * 2.5  # 2.5x oversampling
    min_rate = bandwidth_hz * 2.0
    if target_rate < min_rate:
        target_rate = bandwidth_hz * 2.5

    ratio = Fraction(target_rate / sample_rate).limit_denominator(100)
    up = ratio.numerator
    down = ratio.denominator
    actual_target_rate = sample_rate * up / down

    logger.info(
        f"[SubbandExtractor] Resampling: {sample_rate / 1e6:.2f}MHz → {actual_target_rate / 1e6:.2f}MHz ({up}:{down})"
    )

    if up != 1 or down != 1:
        # Design filter for resample_poly
        cutoff_hz = bandwidth_hz / 2
        interp_rate = sample_rate * up
        transition_width = cutoff_hz * 0.1
        nyquist = interp_rate / 2
        normalized_transition = transition_width / nyquist

        try:
            numtaps, beta = kaiserord(stopband_db, normalized_transition)
        except ValueError:
            numtaps = 255
            beta = 8.6

        if numtaps % 2 == 0:
            numtaps += 1
        numtaps = max(numtaps, 63)
        numtaps = min(numtaps, 4095)

        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        filter_taps = firwin(numtaps, normalized_cutoff, window=("kaiser", beta)).astype(np.float32)

        logger.info(f"[SubbandExtractor] Filter: {len(filter_taps)} taps")

        # CHUNKED resampling for accurate progress
        # Chunk size: ~2M samples input (fits in L3 cache)
        chunk_size = 2_000_000
        num_chunks = max(1, int(np.ceil(len(iq_data) / chunk_size)))

        resampled_chunks = []
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, len(iq_data))
            chunk = iq_data[chunk_start:chunk_end]

            # Resample this chunk
            resampled = resample_poly(chunk, up, down, window=filter_taps)
            resampled_chunks.append(resampled.astype(np.complex64))

            # Update progress: 30% to 85% over all chunks
            chunk_progress = 0.30 + 0.55 * (i + 1) / num_chunks
            report(chunk_progress, "resampling")
            logger.info(
                f"[SubbandExtractor] Resampled chunk {i + 1}/{num_chunks} ({chunk_progress * 100:.0f}%)"
            )

        # Concatenate chunks
        iq_resampled = np.concatenate(resampled_chunks)
        del resampled_chunks  # Free memory
    else:
        iq_resampled = iq_data

    report(0.85, "resampling")
    logger.info(f"[SubbandExtractor] Phase 2 done: {len(iq_resampled) / 1e6:.2f}M output samples")

    # ===== PHASE 3: FINALIZE & WRITE (85-100%) =====
    report(0.85, "writing")

    # Normalize
    power = np.mean(np.abs(iq_resampled) ** 2)
    if power > 0:
        iq_resampled = iq_resampled / np.sqrt(power)

    report(0.90, "writing")

    # Prepare output metadata
    output_center_freq = source_header["center_freq"] + center_offset_hz
    output_bandwidth = actual_target_rate  # MUST match sample_rate!

    output_metadata = {
        "version": 1,
        "sample_rate": actual_target_rate,
        "center_freq": output_center_freq,
        "bandwidth": output_bandwidth,
        "num_samples": len(iq_resampled),
        "start_time": source_header.get("start_time", 0.0) + start_sec,
        "signal_name": source_header.get("signal_name", "UNKNOWN"),
        "latitude": source_header.get("latitude", 0.0),
        "longitude": source_header.get("longitude", 0.0),
    }

    # Write output file
    _write_rfcap_header(output_path, output_metadata)

    report(0.95, "writing")

    # Append IQ data
    with open(output_path, "ab") as f:
        iq_resampled.tofile(f)

    elapsed = time.perf_counter() - start_time
    report(1.0, "done")

    logger.info(
        f"[SubbandExtractor] COMPLETE in {elapsed:.1f}s: "
        f"sr={actual_target_rate / 1e6:.2f}MHz, cf={output_center_freq / 1e6:.2f}MHz"
    )

    return {
        "output_path": output_path,
        "source_rate": sample_rate,
        "output_rate": actual_target_rate,
        "bandwidth": bandwidth_hz,
        "center_offset": center_offset_hz,
        "output_center_freq": output_center_freq,
        "input_samples": len(iq_data),
        "output_samples": len(iq_resampled),
        "decimation_ratio": sample_rate / actual_target_rate,
        "filter_taps": len(filter_taps) if up != 1 or down != 1 else 0,
        "processing_time": elapsed,
    }
