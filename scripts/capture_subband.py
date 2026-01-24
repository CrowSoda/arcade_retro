#!/usr/bin/env python3
"""
capture_subband.py - Extract subband IQ data from RFCAP files

This script extracts a frequency subband from a wideband RFCAP capture file.
It applies frequency shift, filtering, and resampling to isolate the signal.

Usage:
    python capture_subband.py source.rfcap output.rfcap \
        --start-sec 5.0 --duration-sec 10.0 \
        --freq-offset-hz 1e6 --bandwidth-hz 2e6

RFCAP Header Format (512 bytes):
    Offset  Size  Type      Field
    0       4     char[4]   Magic ("G20\0")
    4       4     uint32    Version
    8       8     float64   Sample rate (Hz)
    16      8     float64   Center freq (Hz)
    24      8     float64   Bandwidth (Hz)
    32      8     uint64    Number of samples
    40      8     float64   Start time (epoch)
    48      32    char[32]  Signal name
    80      8     float64   Latitude
    88      8     float64   Longitude
    96      416   reserved  (zeros)
"""

import argparse
import struct
import sys
import numpy as np
from scipy import signal
from pathlib import Path

RFCAP_HEADER_SIZE = 512
RFCAP_MAGIC = b'G20\x00'

def read_rfcap_header(filepath):
    """Read RFCAP header and return metadata dict"""
    with open(filepath, 'rb') as f:
        header = f.read(RFCAP_HEADER_SIZE)
    
    if header[:4] != RFCAP_MAGIC:
        raise ValueError(f"Invalid RFCAP magic: {header[:4]}")
    
    version = struct.unpack_from('<I', header, 4)[0]
    sample_rate = struct.unpack_from('<d', header, 8)[0]
    center_freq = struct.unpack_from('<d', header, 16)[0]
    bandwidth = struct.unpack_from('<d', header, 24)[0]
    num_samples = struct.unpack_from('<Q', header, 32)[0]
    start_time = struct.unpack_from('<d', header, 40)[0]
    signal_name = header[48:80].rstrip(b'\x00').decode('utf-8', errors='ignore')
    latitude = struct.unpack_from('<d', header, 80)[0]
    longitude = struct.unpack_from('<d', header, 88)[0]
    
    return {
        'version': version,
        'sample_rate': sample_rate,
        'center_freq': center_freq,
        'bandwidth': bandwidth,
        'num_samples': num_samples,
        'start_time': start_time,
        'signal_name': signal_name,
        'latitude': latitude,
        'longitude': longitude,
    }

def write_rfcap_header(filepath, metadata):
    """Write RFCAP header"""
    header = bytearray(RFCAP_HEADER_SIZE)
    
    # Magic
    header[0:4] = RFCAP_MAGIC
    
    # Version
    struct.pack_into('<I', header, 4, metadata.get('version', 1))
    
    # Sample rate
    struct.pack_into('<d', header, 8, metadata['sample_rate'])
    
    # Center frequency
    struct.pack_into('<d', header, 16, metadata['center_freq'])
    
    # Bandwidth
    struct.pack_into('<d', header, 24, metadata['bandwidth'])
    
    # Number of samples
    struct.pack_into('<Q', header, 32, metadata['num_samples'])
    
    # Start time
    struct.pack_into('<d', header, 40, metadata.get('start_time', 0))
    
    # Signal name (32 bytes, null-padded)
    name_bytes = metadata.get('signal_name', 'unknown')[:31].encode('utf-8')
    header[48:48+len(name_bytes)] = name_bytes
    
    # Latitude/Longitude
    struct.pack_into('<d', header, 80, metadata.get('latitude', 0))
    struct.pack_into('<d', header, 88, metadata.get('longitude', 0))
    
    with open(filepath, 'wb') as f:
        f.write(header)
    
    return header

def read_iq_data(filepath, offset_samples=0, num_samples=None):
    """Read complex64 IQ data from RFCAP file"""
    header = read_rfcap_header(filepath)
    
    with open(filepath, 'rb') as f:
        f.seek(RFCAP_HEADER_SIZE + offset_samples * 8)  # 8 bytes per complex64
        
        if num_samples is None:
            num_samples = header['num_samples'] - offset_samples
        
        # Clamp to available samples
        available = header['num_samples'] - offset_samples
        num_samples = min(num_samples, available)
        
        if num_samples <= 0:
            return np.array([], dtype=np.complex64), header
        
        # Read as float32 pairs (I, Q interleaved)
        raw = np.fromfile(f, dtype=np.float32, count=num_samples * 2)
        
        # Convert to complex
        iq = raw[0::2] + 1j * raw[1::2]
        return iq.astype(np.complex64), header

def frequency_shift(iq_data, freq_offset_hz, sample_rate):
    """Apply frequency shift to IQ data"""
    if freq_offset_hz == 0:
        return iq_data
    
    t = np.arange(len(iq_data)) / sample_rate
    shift = np.exp(-2j * np.pi * freq_offset_hz * t)
    return iq_data * shift.astype(np.complex64)

def lowpass_filter_and_resample(iq_data, source_rate, target_bandwidth, target_rate=None):
    """Apply lowpass filter and optionally resample"""
    if target_rate is None:
        target_rate = target_bandwidth * 1.25  # Nyquist + margin
    
    # Design lowpass filter
    cutoff = target_bandwidth / 2
    nyquist = source_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure cutoff is valid
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    
    # FIR lowpass filter
    numtaps = 101
    taps = signal.firwin(numtaps, normalized_cutoff)
    
    # Apply filter
    filtered = signal.lfilter(taps, 1.0, iq_data)
    
    # Resample if needed
    if abs(target_rate - source_rate) > 1:
        # Calculate rational resampling ratio
        gcd = np.gcd(int(target_rate), int(source_rate))
        up = int(target_rate) // gcd
        down = int(source_rate) // gcd
        
        # Limit ratio to prevent memory issues
        max_ratio = 100
        if up > max_ratio or down > max_ratio:
            # Fall back to simple resampling
            num_output = int(len(filtered) * target_rate / source_rate)
            resampled = signal.resample(filtered, num_output)
        else:
            resampled = signal.resample_poly(filtered, up, down)
        
        return resampled.astype(np.complex64), target_rate
    
    return filtered.astype(np.complex64), source_rate

def extract_subband(
    source_file,
    output_file,
    start_sec=0,
    duration_sec=None,
    freq_offset_hz=0,
    target_bandwidth_hz=None,
    signal_name=None,
    progress_callback=None
):
    """
    Extract a frequency subband from RFCAP file.
    
    Args:
        source_file: Path to source RFCAP file
        output_file: Path to output RFCAP file
        start_sec: Start time in seconds
        duration_sec: Duration in seconds (None = rest of file)
        freq_offset_hz: Frequency offset from center (Hz)
        target_bandwidth_hz: Target bandwidth (Hz), None = keep original
        signal_name: Name for output file
        progress_callback: Function called with progress (0-1)
    """
    # Read source header
    source_header = read_rfcap_header(source_file)
    sample_rate = source_header['sample_rate']
    
    if progress_callback:
        progress_callback(0.1)
        print("PROGRESS:0.1")
    
    # Calculate sample offsets
    offset_samples = int(start_sec * sample_rate)
    if duration_sec is not None:
        num_samples = int(duration_sec * sample_rate)
    else:
        num_samples = None
    
    # Read IQ data
    iq_data, _ = read_iq_data(source_file, offset_samples, num_samples)
    
    if len(iq_data) == 0:
        raise ValueError("No data to extract")
    
    if progress_callback:
        progress_callback(0.3)
        print("PROGRESS:0.3")
    
    # Apply frequency shift
    if freq_offset_hz != 0:
        iq_data = frequency_shift(iq_data, freq_offset_hz, sample_rate)
    
    if progress_callback:
        progress_callback(0.5)
        print("PROGRESS:0.5")
    
    # Filter and resample if bandwidth specified
    output_rate = sample_rate
    if target_bandwidth_hz is not None and target_bandwidth_hz < source_header['bandwidth']:
        iq_data, output_rate = lowpass_filter_and_resample(
            iq_data, sample_rate, target_bandwidth_hz
        )
    
    if progress_callback:
        progress_callback(0.7)
        print("PROGRESS:0.7")
    
    # Prepare output metadata
    output_center_freq = source_header['center_freq'] + freq_offset_hz
    output_bandwidth = target_bandwidth_hz if target_bandwidth_hz else source_header['bandwidth']
    
    output_metadata = {
        'version': 1,
        'sample_rate': output_rate,
        'center_freq': output_center_freq,
        'bandwidth': output_bandwidth,
        'num_samples': len(iq_data),
        'start_time': source_header['start_time'] + start_sec,
        'signal_name': signal_name or source_header['signal_name'],
        'latitude': source_header['latitude'],
        'longitude': source_header['longitude'],
    }
    
    # Write output file
    write_rfcap_header(output_file, output_metadata)
    
    # Append IQ data
    with open(output_file, 'ab') as f:
        # Convert complex to interleaved float32
        interleaved = np.zeros(len(iq_data) * 2, dtype=np.float32)
        interleaved[0::2] = iq_data.real
        interleaved[1::2] = iq_data.imag
        interleaved.tofile(f)
    
    if progress_callback:
        progress_callback(1.0)
        print("PROGRESS:1.0")
    
    return output_metadata

def main():
    parser = argparse.ArgumentParser(
        description='Extract frequency subband from RFCAP file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('source', help='Source RFCAP file')
    parser.add_argument('output', help='Output RFCAP file')
    parser.add_argument('--start-sec', type=float, default=0, help='Start time (seconds)')
    parser.add_argument('--duration-sec', type=float, default=None, help='Duration (seconds)')
    parser.add_argument('--freq-offset-hz', type=float, default=0, help='Frequency offset from center (Hz)')
    parser.add_argument('--bandwidth-hz', type=float, default=None, help='Target bandwidth (Hz)')
    parser.add_argument('--signal-name', type=str, default=None, help='Output signal name')
    parser.add_argument('--info', action='store_true', help='Show source file info and exit')
    
    args = parser.parse_args()
    
    if args.info:
        header = read_rfcap_header(args.source)
        print(f"Signal Name: {header['signal_name']}")
        print(f"Sample Rate: {header['sample_rate']/1e6:.2f} MHz")
        print(f"Center Freq: {header['center_freq']/1e6:.2f} MHz")
        print(f"Bandwidth:   {header['bandwidth']/1e6:.2f} MHz")
        print(f"Samples:     {header['num_samples']}")
        print(f"Duration:    {header['num_samples']/header['sample_rate']:.2f} s")
        return 0
    
    try:
        result = extract_subband(
            source_file=args.source,
            output_file=args.output,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            freq_offset_hz=args.freq_offset_hz,
            target_bandwidth_hz=args.bandwidth_hz,
            signal_name=args.signal_name,
        )
        
        print(f"SUCCESS: Extracted {result['num_samples']} samples")
        print(f"  Center: {result['center_freq']/1e6:.2f} MHz")
        print(f"  BW: {result['bandwidth']/1e6:.2f} MHz")
        print(f"  Rate: {result['sample_rate']/1e6:.2f} MHz")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
