#!/usr/bin/env python3
"""
Generate demo capture files for the training page.
Uses G20 RFCAP format: 512-byte header + complex64 IQ data

Properly resamples data when sub-band tuning to different bandwidths.
"""

import os
import struct
from datetime import datetime, timedelta

import numpy as np
from scipy import signal as scipy_signal

# ============================================================================
# G20 RFCAP FILE FORMAT (Blue file inspired)
# ============================================================================
#
# Total header: 512 bytes (fixed size)
#
# Offset  Size   Type      Field
# ------  ----   ----      -----
# 0       4      char[4]   Magic ("G20\x00")
# 4       4      uint32    Version (1)
# 8       8      float64   Sample rate (Hz)
# 16      8      float64   Center frequency (Hz)
# 24      8      float64   Bandwidth (Hz)
# 32      8      uint64    Number of samples
# 40      8      float64   Start time (Unix epoch seconds)
# 48      32     char[32]  Signal name (null-padded)
# 80      8      float64   Latitude (degrees)
# 88      8      float64   Longitude (degrees)
# 96      416    reserved  (zeros)
#
# After header: complex64 IQ data (float32 I, float32 Q pairs)
# ============================================================================

HEADER_SIZE = 512
MAGIC = b"G20\x00"
VERSION = 1


def write_rfcap_header(
    f,
    sample_rate: float,
    center_freq: float,
    bandwidth: float,
    num_samples: int,
    start_time: float,
    signal_name: str,
    latitude: float = 0.0,
    longitude: float = 0.0,
):
    """Write 512-byte RFCAP header."""
    # Prepare signal name (32 bytes, null-padded)
    name_bytes = signal_name.encode("utf-8")[:31].ljust(32, b"\x00")

    # Pack header
    header = struct.pack(
        "<4sI d d d Q d 32s d d",
        MAGIC,  # 4 bytes: Magic
        VERSION,  # 4 bytes: Version
        sample_rate,  # 8 bytes: Sample rate
        center_freq,  # 8 bytes: Center frequency
        bandwidth,  # 8 bytes: Bandwidth
        num_samples,  # 8 bytes: Number of samples
        start_time,  # 8 bytes: Start time
        name_bytes,  # 32 bytes: Signal name
        latitude,  # 8 bytes: Latitude
        longitude,  # 8 bytes: Longitude
    )

    # Pad to 512 bytes
    header = header.ljust(HEADER_SIZE, b"\x00")
    f.write(header)


def read_rfcap_header(filepath: str) -> dict:
    """Read and parse RFCAP header."""
    with open(filepath, "rb") as f:
        raw = f.read(HEADER_SIZE)

    # Unpack
    (
        magic,
        version,
        sample_rate,
        center_freq,
        bandwidth,
        num_samples,
        start_time,
        name_bytes,
        latitude,
        longitude,
    ) = struct.unpack("<4sI d d d Q d 32s d d", raw[:96])

    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic}")

    return {
        "version": version,
        "sample_rate": sample_rate,
        "center_freq_hz": center_freq,
        "center_freq_mhz": center_freq / 1e6,
        "bandwidth_hz": bandwidth,
        "bandwidth_mhz": bandwidth / 1e6,
        "num_samples": num_samples,
        "start_time": start_time,
        "signal_name": name_bytes.rstrip(b"\x00").decode("utf-8"),
        "latitude": latitude,
        "longitude": longitude,
        "duration_sec": num_samples / sample_rate if sample_rate > 0 else 0,
        "data_offset": HEADER_SIZE,
    }


def resample_iq(iq_complex: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """
    Resample complex IQ data from source_rate to target_rate.
    Uses polyphase resampling for efficiency.
    """
    if source_rate == target_rate:
        return iq_complex

    # Calculate rational resampling factors
    from math import gcd

    # Use integer approximation for resample ratio
    ratio = target_rate / source_rate

    # For rational resampling, find up/down factors
    # Limit to reasonable factors to avoid memory issues
    max_factor = 100
    up = int(round(ratio * max_factor))
    down = max_factor

    # Simplify
    g = gcd(up, down)
    up //= g
    down //= g

    # Cap factors to prevent memory explosion
    while up > 50 or down > 50:
        up = (up + 1) // 2
        down = (down + 1) // 2
        if up == 0:
            up = 1
        if down == 0:
            down = 1

    print(
        f"    Resampling: {source_rate/1e6:.1f} MHz -> {target_rate/1e6:.1f} MHz (up={up}, down={down})"
    )

    # Resample I and Q separately then recombine
    i_resampled = scipy_signal.resample_poly(iq_complex.real, up, down)
    q_resampled = scipy_signal.resample_poly(iq_complex.imag, up, down)

    return (i_resampled + 1j * q_resampled).astype(np.complex64)


def filter_and_extract_subband(
    iq_complex: np.ndarray, source_rate: float, center_offset_hz: float, target_bw_hz: float
) -> tuple:
    """
    Extract a sub-band from wideband IQ data.

    Args:
        iq_complex: Source complex IQ data
        source_rate: Source sample rate in Hz
        center_offset_hz: Offset from source center frequency (0 = centered)
        target_bw_hz: Desired output bandwidth in Hz

    Returns:
        (resampled_iq, new_sample_rate)
    """
    # Design lowpass filter for target bandwidth
    # Nyquist rate for the target bandwidth
    target_rate = target_bw_hz  # Sample rate = bandwidth for complex

    # First, shift the signal if needed (frequency translation)
    if center_offset_hz != 0:
        t = np.arange(len(iq_complex)) / source_rate
        shift = np.exp(-2j * np.pi * center_offset_hz * t).astype(np.complex64)
        iq_complex = iq_complex * shift

    # Design anti-aliasing lowpass filter
    cutoff = (target_bw_hz / 2) / (source_rate / 2)  # Normalized cutoff
    if cutoff >= 1.0:
        cutoff = 0.99  # Prevent filter design errors

    # Use a relatively sharp filter
    numtaps = min(255, len(iq_complex) // 10)
    if numtaps < 15:
        numtaps = 15
    if numtaps % 2 == 0:
        numtaps += 1  # Must be odd

    try:
        lpf = scipy_signal.firwin(numtaps, cutoff, window="hamming")

        # Apply filter to I and Q
        i_filtered = scipy_signal.lfilter(lpf, 1.0, iq_complex.real)
        q_filtered = scipy_signal.lfilter(lpf, 1.0, iq_complex.imag)
        iq_filtered = (i_filtered + 1j * q_filtered).astype(np.complex64)
    except Exception as e:
        print(f"    Warning: Filter failed ({e}), skipping filter")
        iq_filtered = iq_complex

    # Resample to target rate
    resampled = resample_iq(iq_filtered, source_rate, target_rate)

    return resampled, target_rate


# ============================================================================
# DEMO FILE GENERATION
# ============================================================================

SOURCE_IQ_PATH = "data/825MHz.sigmf-data"
OUTPUT_DIR = "data/captures"
SOURCE_SAMPLE_RATE = 20e6  # 20 MHz from sigmf-meta
CENTER_FREQ_HZ = 825e6  # 825 MHz
DURATION_SEC = 120  # 2 minutes each (will use what's available)

# Demo signals to create - different bandwidths require resampling
DEMO_SIGNALS = [
    {"name": "creamy_chicken", "offset_sec": 10, "bw_mhz": 5.0, "center_offset_mhz": 0},
    {"name": "hostile_burst", "offset_sec": 300, "bw_mhz": 2.5, "center_offset_mhz": 2.5},
    {"name": "lte_uplink", "offset_sec": 600, "bw_mhz": 10.0, "center_offset_mhz": -3.0},
    {"name": "narrowband_fsk", "offset_sec": 100, "bw_mhz": 1.0, "center_offset_mhz": 5.0},
]


def main():
    print("=" * 60)
    print("Generating G20 RFCAP demo captures with PROPER RESAMPLING")
    print("=" * 60)
    print(f"Source: {SOURCE_IQ_PATH}")
    print(f"Source Sample Rate: {SOURCE_SAMPLE_RATE/1e6:.1f} MHz")
    print(f"Target Duration: {DURATION_SEC}s (2 min) each")
    print(f"Output: {OUTPUT_DIR}/")
    print()

    # Check source exists
    if not os.path.exists(SOURCE_IQ_PATH):
        print(f"ERROR: Source file not found: {SOURCE_IQ_PATH}")
        return

    # Get source file size
    file_size = os.path.getsize(SOURCE_IQ_PATH)
    total_samples = file_size // 8  # complex64 = 8 bytes
    total_duration = total_samples / SOURCE_SAMPLE_RATE
    print(f"Source: {file_size / 1e9:.2f} GB ({total_samples:,} samples, {total_duration:.1f}s)")
    print()

    # Samples to read per capture (from source file)
    source_samples_per_capture = int(DURATION_SEC * SOURCE_SAMPLE_RATE)
    source_bytes_per_capture = source_samples_per_capture * 8

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, sig in enumerate(DEMO_SIGNALS, 1):
        print(f"[{i}/{len(DEMO_SIGNALS)}] {sig['name']} ({sig['bw_mhz']} MHz BW)")

        target_bw_hz = sig["bw_mhz"] * 1e6
        center_offset_hz = sig.get("center_offset_mhz", 0) * 1e6

        # Calculate byte offset into source file
        offset_samples = int(sig["offset_sec"] * SOURCE_SAMPLE_RATE)
        if offset_samples + source_samples_per_capture > total_samples:
            offset_samples = offset_samples % max(1, total_samples - source_samples_per_capture)
        offset_bytes = offset_samples * 8

        # Determine how much we can actually read
        available_bytes = file_size - offset_bytes
        bytes_to_read = min(source_bytes_per_capture, available_bytes)
        samples_to_read = bytes_to_read // 8

        print(
            f"    Reading {bytes_to_read / 1e6:.1f} MB ({samples_to_read:,} samples) from offset {offset_bytes / 1e6:.1f} MB..."
        )

        # Read source IQ data
        with open(SOURCE_IQ_PATH, "rb") as src:
            src.seek(offset_bytes)
            raw_bytes = src.read(bytes_to_read)

        # Convert to complex64
        iq_complex = np.frombuffer(raw_bytes, dtype=np.complex64)
        actual_samples_read = len(iq_complex)

        print(
            f"    Loaded {actual_samples_read:,} complex samples ({actual_samples_read / SOURCE_SAMPLE_RATE:.2f}s at source rate)"
        )

        # Sub-band extraction and resampling
        if target_bw_hz < SOURCE_SAMPLE_RATE:
            print(
                f"    Sub-band tuning: BW {SOURCE_SAMPLE_RATE/1e6:.1f} -> {target_bw_hz/1e6:.1f} MHz, offset {center_offset_hz/1e6:.1f} MHz"
            )
            resampled_iq, output_sample_rate = filter_and_extract_subband(
                iq_complex, SOURCE_SAMPLE_RATE, center_offset_hz, target_bw_hz
            )
        else:
            # Full bandwidth - no resampling needed
            resampled_iq = iq_complex
            output_sample_rate = SOURCE_SAMPLE_RATE

        # Calculate actual output parameters
        actual_num_samples = len(resampled_iq)
        actual_duration = actual_num_samples / output_sample_rate

        print(
            f"    Output: {actual_num_samples:,} samples at {output_sample_rate/1e6:.1f} MHz = {actual_duration:.2f}s"
        )

        # Convert to interleaved float32 bytes for storage
        iq_bytes = resampled_iq.view(np.float32).tobytes()

        # Generate filename with DTG
        fake_dtg = datetime.now() - timedelta(days=i)
        dtg_str = fake_dtg.strftime("%d%H%MZ%b%y").upper()
        filename = f"{sig['name'].upper()}_{dtg_str}.rfcap"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Calculate output center frequency (accounting for sub-band offset)
        output_center_freq = CENTER_FREQ_HZ + center_offset_hz

        # Write RFCAP file with CORRECT header values
        with open(filepath, "wb") as out:
            write_rfcap_header(
                out,
                sample_rate=output_sample_rate,  # Actual resampled rate
                center_freq=output_center_freq,  # Adjusted center frequency
                bandwidth=target_bw_hz,  # Target bandwidth
                num_samples=actual_num_samples,  # ACTUAL samples written
                start_time=fake_dtg.timestamp(),
                signal_name=sig["name"],
                latitude=39.7392,  # Denver
                longitude=-104.9903,
            )
            out.write(iq_bytes)

        file_size_mb = os.path.getsize(filepath) / 1e6
        print(f"    Saved: {filepath} ({file_size_mb:.1f} MB)")

        # Verify by reading header back
        header = read_rfcap_header(filepath)
        print(
            f"    Verified: {header['signal_name']}, sr={header['sample_rate']/1e6:.1f}MHz, "
            f"bw={header['bandwidth_mhz']:.1f}MHz, dur={header['duration_sec']:.2f}s, "
            f"cf={header['center_freq_mhz']:.2f}MHz"
        )
        print()

    print("=" * 60)
    print(f"Done! {len(DEMO_SIGNALS)} RFCAP files created in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
