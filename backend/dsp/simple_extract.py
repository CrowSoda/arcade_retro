"""
Simple subband extraction: shift → filter → decimate → save.

CHUNKED PROCESSING - does NOT load entire file into RAM!
For complex IQ: sample_rate = bandwidth
"""

import struct
import time

import numpy as np

# Logging
from logger_config import get_logger
from scipy.signal import firwin, lfilter, lfilter_zi

logger = get_logger("simple_extract")


HEADER_SIZE = 512


def read_rfcap_header(filepath: str) -> dict:
    """Read only the header from RFCAP file (512 bytes)."""
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)

    sample_rate = struct.unpack("<d", header_bytes[8:16])[0]
    center_freq = struct.unpack("<d", header_bytes[16:24])[0]
    bandwidth = struct.unpack("<d", header_bytes[24:32])[0]
    num_samples = struct.unpack("<Q", header_bytes[32:40])[0]

    # Read signal name at offset 48
    signal_name_raw = header_bytes[48:80]
    signal_name = signal_name_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    return {
        "sample_rate": sample_rate,
        "center_freq_hz": center_freq,
        "bandwidth_hz": bandwidth,
        "num_samples": num_samples,
        "signal_name": signal_name,
    }


def write_rfcap_header(
    f,
    center_freq_hz: float,
    sample_rate: float,
    bandwidth_hz: float,
    num_samples: int,
    signal_name: str = "SBT",
):
    """Write G20 RFCAP header to file (512 bytes)."""
    header = bytearray(HEADER_SIZE)

    # Magic: "G20\x00"
    header[0:4] = b"G20\x00"

    # Version: 1 (offset 4)
    struct.pack_into("<I", header, 4, 1)

    # Sample rate (double, offset 8)
    struct.pack_into("<d", header, 8, sample_rate)

    # Center freq (double, offset 16)
    struct.pack_into("<d", header, 16, center_freq_hz)

    # Bandwidth (double, offset 24)
    struct.pack_into("<d", header, 24, bandwidth_hz)

    # Num samples (uint64, offset 32)
    struct.pack_into("<Q", header, 32, num_samples)

    # Start time (double, offset 40)
    struct.pack_into("<d", header, 40, time.time())

    # Signal name (32 bytes at offset 48)
    name_bytes = signal_name.encode("utf-8")[:31]
    header[48 : 48 + len(name_bytes)] = name_bytes

    f.write(header)


def update_sample_count(filepath: str, num_samples: int):
    """Update the num_samples field in the header (offset 32)."""
    with open(filepath, "r+b") as f:
        f.seek(32)
        f.write(struct.pack("<Q", num_samples))


def extract_subband(
    input_path: str,
    output_path: str,
    original_center_hz: float,
    original_sample_rate: float,
    new_center_hz: float,
    new_bandwidth_hz: float,
    num_taps: int = 101,
    progress_callback=None,
    chunk_seconds: float = 0.5,  # Process 0.5 seconds at a time (~80 MB RAM)
) -> dict:
    """
    Extract a subband from wideband IQ capture using CHUNKED PROCESSING.

    Does NOT load entire file into RAM - processes in chunks!

    Args:
        input_path: Path to source RFCAP file
        output_path: Path for output RFCAP file
        original_center_hz: Center frequency of source file
        original_sample_rate: Sample rate of source file (Hz)
        new_center_hz: Center frequency for extracted subband
        new_bandwidth_hz: Bandwidth for extracted subband
        num_taps: Number of filter taps (odd, default 101)
        progress_callback: Callback function(progress: float) where progress is 0-1
        chunk_seconds: Seconds of data to process at a time (default 0.5s = ~80MB)

    Returns:
        Dict with extraction stats
    """

    def _progress(p, phase="processing"):
        if progress_callback:
            try:
                # Support both old (1 arg) and new (2 arg) callbacks
                import inspect

                sig = inspect.signature(progress_callback)
                if len(sig.parameters) >= 2:
                    progress_callback(p, phase)
                else:
                    progress_callback(p)
            except:
                pass

    logger.info(f"[extract_subband] CHUNKED processing: {chunk_seconds}s chunks")
    logger.info(f"[extract_subband] Input: {input_path}")
    logger.info(
        f"[extract_subband] Original: center={original_center_hz / 1e6:.3f} MHz, rate={original_sample_rate / 1e6:.3f} Msps",
    )
    logger.info(
        f"[extract_subband] Target: center={new_center_hz / 1e6:.3f} MHz, bw={new_bandwidth_hz / 1e6:.3f} MHz",
    )
    _progress(0.02)

    # Read source header to get total samples
    source_header = read_rfcap_header(input_path)
    total_input_samples = source_header["num_samples"]

    # Calculate decimation
    new_sample_rate = new_bandwidth_hz  # For complex IQ: sample_rate = bandwidth
    decimation = int(original_sample_rate / new_sample_rate)
    if decimation < 1:
        decimation = 1

    actual_output_rate = original_sample_rate / decimation

    # Calculate frequency shift
    shift_hz = new_center_hz - original_center_hz

    # Design FIR lowpass filter
    cutoff_hz = new_bandwidth_hz / 2
    nyquist_hz = original_sample_rate / 2
    cutoff_normalized = min(cutoff_hz / nyquist_hz, 0.99)

    logger.info(
        f"[extract_subband] Filter: cutoff={cutoff_hz / 1e6:.3f} MHz, normalized={cutoff_normalized:.4f}",
    )
    logger.info(
        f"[extract_subband] Decimation: {decimation}x (output rate: {actual_output_rate / 1e6:.2f} Msps)",
    )

    taps = firwin(num_taps, cutoff_normalized)
    _progress(0.05)

    # Initialize filter state for continuity between chunks
    zi = lfilter_zi(taps, 1.0) * 0  # Zero initial state (complex)
    zi = zi.astype(np.complex64)

    # Chunk parameters
    chunk_samples = int(chunk_seconds * original_sample_rate)
    total_chunks = int(np.ceil(total_input_samples / chunk_samples))

    logger.info(
        f"[extract_subband] Total: {total_input_samples} samples in {total_chunks} chunks of {chunk_samples} samples",
    )
    _progress(0.08)

    # Open files
    total_output_samples = 0
    time_offset = 0.0  # Track time for NCO phase continuity

    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        # Skip input header
        f_in.seek(HEADER_SIZE)

        # Write output header (will update sample count at end)
        write_rfcap_header(
            f_out,
            center_freq_hz=new_center_hz,
            sample_rate=actual_output_rate,
            bandwidth_hz=actual_output_rate,  # MUST match sample_rate!
            num_samples=0,  # Updated later
            signal_name=source_header.get("signal_name", "SBT"),
        )

        # Process chunks
        for chunk_idx in range(total_chunks):
            # Read chunk
            remaining = total_input_samples - (chunk_idx * chunk_samples)
            samples_to_read = min(chunk_samples, remaining)
            bytes_to_read = samples_to_read * 8  # 8 bytes per complex64

            raw = f_in.read(bytes_to_read)
            if len(raw) < 8:
                break

            # Parse as interleaved float32 -> complex64
            iq_float = np.frombuffer(raw, dtype=np.float32)
            iq = (iq_float[0::2] + 1j * iq_float[1::2]).astype(np.complex64)

            # Frequency shift with phase continuity
            t = np.arange(len(iq), dtype=np.float64) / original_sample_rate + time_offset
            nco = np.exp(-2j * np.pi * shift_hz * t).astype(np.complex64)
            iq = iq * nco
            time_offset += len(iq) / original_sample_rate

            # Filter with state continuity
            iq_filtered, zi = lfilter(taps, 1.0, iq, zi=zi)
            iq_filtered = iq_filtered.astype(np.complex64)

            # Decimate
            iq_decimated = iq_filtered[::decimation]

            # Write as interleaved float32
            interleaved = np.zeros(len(iq_decimated) * 2, dtype=np.float32)
            interleaved[0::2] = iq_decimated.real
            interleaved[1::2] = iq_decimated.imag
            f_out.write(interleaved.tobytes())

            total_output_samples += len(iq_decimated)

            # Progress
            progress = 0.1 + 0.85 * ((chunk_idx + 1) / total_chunks)
            _progress(progress)

            if chunk_idx % 10 == 0 or chunk_idx == total_chunks - 1:
                logger.info(
                    f"[extract_subband] Chunk {chunk_idx + 1}/{total_chunks}: {total_output_samples} output samples",
                )

    # Update header with actual sample count
    update_sample_count(output_path, total_output_samples)

    logger.info(f"[extract_subband] Complete! Output: {total_output_samples} samples")
    logger.info(
        f"[extract_subband] Output header: sr={actual_output_rate / 1e6:.2f}MHz, bw={actual_output_rate / 1e6:.2f}MHz",
    )
    logger.info(f"[extract_subband] Saved to: {output_path}")
    _progress(1.0)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "input_samples": total_input_samples,
        "output_samples": total_output_samples,
        "decimation": decimation,
        "original_center_hz": original_center_hz,
        "original_sample_rate": original_sample_rate,
        "new_center_hz": new_center_hz,
        "new_sample_rate": actual_output_rate,
        "new_bandwidth_hz": new_bandwidth_hz,
        "shift_hz": shift_hz,
        "filter_taps": num_taps,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 6:
        result = extract_subband(
            input_path=sys.argv[1],
            output_path=sys.argv[2],
            original_center_hz=float(sys.argv[3]),
            original_sample_rate=float(sys.argv[4]),
            new_center_hz=float(sys.argv[5]),
            new_bandwidth_hz=float(sys.argv[6]) if len(sys.argv) > 6 else 2e6,
        )
        logger.info(f"Result: {result}")
    else:
        logger.info(
            "Usage: python simple_extract.py <input> <output> <orig_center_hz> <orig_rate> <new_center_hz> [new_bw_hz]"
        )
