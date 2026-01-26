# Part 3: Backend Implementation

## File Structure

```
g20_demo/backend/
├── dsp/                           # NEW: DSP module
│   ├── __init__.py
│   ├── subband_extractor.py       # Main extraction class
│   ├── filters.py                 # Filter design functions
│   ├── decimation.py              # Multi-stage decimation
│   └── gpu_subband.py             # GPU-accelerated version
│
├── capture_subband.py             # MODIFY: Use new DSP module
└── unified_pipeline.py            # MODIFY: Add live extraction
```

## New File: backend/dsp/subband_extractor.py

```python
"""
Sub-band extraction for CNN training data.

Implements proper DSP pipeline: mix → filter → decimate
with 60-80 dB stopband attenuation for clean training data.
"""

import numpy as np
from scipy.signal import firwin, kaiserord, resample_poly, lfilter
from fractions import Fraction
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ExtractionParams:
    """Parameters for sub-band extraction."""
    source_rate: float           # Hz, e.g., 20e6
    center_offset: float         # Hz from source center
    target_bandwidth: float      # Hz, signal bandwidth
    target_rate: Optional[float] # Hz, None = auto (2.5x BW)
    stopband_db: float = 60.0    # Stopband attenuation
    remove_dc: bool = True       # Remove DC offset
    normalize: bool = True       # Normalize to unit power


class SubbandExtractor:
    """
    Extract narrowband sub-band from wideband IQ data.
    
    Usage:
        extractor = SubbandExtractor(ExtractionParams(...))
        narrowband_iq = extractor.extract(wideband_iq)
    
    NOTE: Uses resample_poly with custom FIR taps - this combines
    filtering and decimation in ONE operation (no double filtering!).
    """
    
    def __init__(self, params: ExtractionParams):
        self.params = params
        
        # Calculate target rate if not specified
        if params.target_rate is None:
            self.target_rate = params.target_bandwidth * 2.5
        else:
            self.target_rate = params.target_rate
        
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
            print(f"[SubbandExtractor] WARNING: Rate approximation error {error_pct:.1f}%")
        self.actual_target_rate = actual_rate
        
        # Design anti-aliasing filter for resample_poly
        self.filter_taps = self._design_filter()
        
        print(f"[SubbandExtractor] Initialized:")
        print(f"  Source: {params.source_rate/1e6:.2f} MHz")
        print(f"  Target: {actual_rate/1e6:.2f} MHz ({self.up}:{self.down})")
        print(f"  Bandwidth: {params.target_bandwidth/1e6:.2f} MHz")
        print(f"  Filter: {len(self.filter_taps)} taps, {params.stopband_db} dB stopband")
    
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
        
        # Transition bandwidth = 10% of cutoff
        transition_width = cutoff_hz * 0.1
        nyquist = interp_rate / 2
        
        # Kaiser window design for specified attenuation
        numtaps, beta = kaiserord(p.stopband_db, transition_width / nyquist)
        
        # Ensure odd number (linear phase)
        if numtaps % 2 == 0:
            numtaps += 1
        
        # Minimum 63 taps
        numtaps = max(numtaps, 63)
        
        # Design FIR lowpass at interpolated rate
        taps = firwin(numtaps, cutoff_hz, window=('kaiser', beta), fs=interp_rate)
        
        return taps.astype(np.float32)
    
    def extract(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Extract sub-band from wideband IQ data.
        
        Pipeline: DC removal → frequency shift → (filter + decimate) → normalize
        
        NOTE: Filter and decimate are COMBINED in resample_poly to avoid
        double filtering. Custom filter taps passed via window parameter.
        """
        p = self.params
        
        # Ensure complex64
        if iq_data.dtype != np.complex64:
            iq_data = iq_data.astype(np.complex64)
        
        # Step 1: DC offset removal (before frequency shift!)
        if p.remove_dc:
            iq_data = iq_data - np.mean(iq_data)
        
        # Step 2: Frequency translation
        if p.center_offset != 0:
            t = np.arange(len(iq_data)) / p.source_rate
            nco = np.exp(-2j * np.pi * p.center_offset * t).astype(np.complex64)
            iq_data = iq_data * nco
        
        # Step 3+4 COMBINED: Anti-alias + decimate in ONE operation
        # resample_poly accepts custom FIR taps via window parameter
        # This avoids double-filtering (don't use separate lfilter!)
        if self.up != 1 or self.down != 1:
            iq_decimated = resample_poly(iq_data, self.up, self.down, window=self.filter_taps)
        else:
            iq_decimated = iq_data
        
        # Step 5: Normalize to unit power
        if p.normalize:
            power = np.mean(np.abs(iq_decimated) ** 2)
            if power > 0:
                iq_decimated = iq_decimated / np.sqrt(power)
        
        return iq_decimated.astype(np.complex64)
    
    def get_output_sample_count(self, input_samples: int) -> int:
        """Calculate output sample count for given input."""
        return int(input_samples * self.up / self.down)
    
    def get_filter_delay_samples(self) -> int:
        """Get filter group delay in output samples."""
        input_delay = (len(self.filter_taps) - 1) // 2
        return int(input_delay * self.up / self.down)
```

## Modify: backend/scripts/capture_subband.py

Replace `lowpass_filter_and_resample()` with:

```python
from backend.dsp.subband_extractor import SubbandExtractor, ExtractionParams

def extract_subband_improved(
    source_file,
    output_file,
    start_sec=0,
    duration_sec=None,
    freq_offset_hz=0,
    target_bandwidth_hz=None,
    signal_name=None,
    stopband_db=60.0,
    progress_callback=None
):
    """Improved sub-band extraction with proper DSP."""
    
    source_header = read_rfcap_header(source_file)
    sample_rate = source_header['sample_rate']
    
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
    
    # Create extractor with proper params
    params = ExtractionParams(
        source_rate=sample_rate,
        center_offset=freq_offset_hz,
        target_bandwidth=target_bandwidth_hz or source_header['bandwidth'],
        target_rate=None,  # Auto: 2.5x bandwidth
        stopband_db=stopband_db,
        remove_dc=True,
        normalize=True,
    )
    
    extractor = SubbandExtractor(params)
    iq_extracted = extractor.extract(iq_data)
    
    # Write output (rest of function same as before)
    # ...
```

## WebSocket Commands

Add to server.py:

```python
# Sub-band extraction commands
elif cmd == 'extract_subband':
    # Extract sub-band from capture file
    # {
    #   "command": "extract_subband",
    #   "source_file": "captures/MAN_123456Z_825MHz.rfcap",
    #   "output_file": "training_data/signals/unk/samples/0001.rfcap",
    #   "center_offset_hz": 500000,
    #   "bandwidth_hz": 2000000,
    #   "start_sec": 5.0,
    #   "duration_sec": 10.0
    # }
    try:
        result = await asyncio.to_thread(
            extract_subband_improved,
            source_file=data['source_file'],
            output_file=data['output_file'],
            freq_offset_hz=data.get('center_offset_hz', 0),
            target_bandwidth_hz=data.get('bandwidth_hz'),
            start_sec=data.get('start_sec', 0),
            duration_sec=data.get('duration_sec'),
        )
        await websocket.send(json.dumps({
            'type': 'subband_extracted',
            'output_file': data['output_file'],
            'output_rate': result['sample_rate'],
            'output_samples': result['num_samples'],
        }))
    except Exception as e:
        await websocket.send(json.dumps({
            'type': 'error',
            'command': 'extract_subband',
            'message': str(e)
        }))
```
