# Part 5: Testing and Validation

## Test Cases

### 1. Filter Design Tests

```python
# backend/dsp/tests/test_filters.py

def test_stopband_attenuation_60db():
    """Verify filter achieves 60 dB stopband."""
    from scipy.signal import freqz
    
    params = ExtractionParams(
        source_rate=20e6,
        center_offset=0,
        target_bandwidth=2e6,
        stopband_db=60.0,
    )
    extractor = SubbandExtractor(params)
    
    # Get frequency response
    w, h = freqz(extractor.filter_taps, worN=8000)
    freqs = w * params.source_rate / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    # Find stopband (> 1.2x cutoff)
    cutoff = params.target_bandwidth / 2
    stopband_mask = freqs > cutoff * 1.2
    
    # Verify attenuation
    stopband_max = mag_db[stopband_mask].max()
    assert stopband_max < -60, f"Stopband only {stopband_max:.1f} dB"


def test_passband_ripple():
    """Verify passband ripple < 0.1 dB."""
    params = ExtractionParams(
        source_rate=20e6,
        center_offset=0,
        target_bandwidth=2e6,
        stopband_db=60.0,
    )
    extractor = SubbandExtractor(params)
    
    w, h = freqz(extractor.filter_taps, worN=8000)
    freqs = w * params.source_rate / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    # Passband (< 0.8x cutoff)
    cutoff = params.target_bandwidth / 2
    passband_mask = freqs < cutoff * 0.8
    
    passband_ripple = mag_db[passband_mask].max() - mag_db[passband_mask].min()
    assert passband_ripple < 0.1, f"Passband ripple {passband_ripple:.2f} dB"
```

### 2. Frequency Translation Tests

```python
def test_frequency_shift_accuracy():
    """Verify frequency shift is accurate."""
    source_rate = 20e6
    freq_offset = 1e6  # 1 MHz offset
    duration = 0.01  # 10 ms
    
    # Create test tone at the offset frequency
    t = np.arange(int(source_rate * duration)) / source_rate
    test_signal = np.exp(2j * np.pi * freq_offset * t).astype(np.complex64)
    
    params = ExtractionParams(
        source_rate=source_rate,
        center_offset=freq_offset,
        target_bandwidth=500e3,
        normalize=False,
        remove_dc=False,
    )
    extractor = SubbandExtractor(params)
    output = extractor.extract(test_signal)
    
    # After shift, tone should be at DC (0 Hz)
    # Verify via FFT that peak is at center
    fft = np.fft.fftshift(np.fft.fft(output))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(output), 1/extractor.target_rate))
    
    peak_freq = freqs[np.argmax(np.abs(fft))]
    assert abs(peak_freq) < 100, f"Peak at {peak_freq} Hz, expected ~0 Hz"
```

### 3. Decimation Ratio Tests

```python
def test_decimation_ratio():
    """Verify output sample count matches expected."""
    params = ExtractionParams(
        source_rate=20e6,
        center_offset=0,
        target_bandwidth=2e6,  # 2.5x = 5 MHz output → 4:1 decimation
    )
    extractor = SubbandExtractor(params)
    
    input_samples = 1000000
    expected_output = extractor.get_output_sample_count(input_samples)
    
    test_data = np.random.randn(input_samples).astype(np.complex64)
    output = extractor.extract(test_data)
    
    # Allow 1% tolerance for filter edge effects
    ratio = len(output) / expected_output
    assert 0.99 < ratio < 1.01, f"Output ratio {ratio:.3f}, expected ~1.0"
```

### 4. Aliasing Detection Tests

```python
def test_no_aliasing():
    """Verify out-of-band signals don't alias into passband."""
    source_rate = 20e6
    
    # Create signal OUTSIDE target passband
    t = np.arange(int(source_rate * 0.01)) / source_rate
    out_of_band_freq = 3e6  # 3 MHz, outside 2 MHz target BW
    test_signal = np.exp(2j * np.pi * out_of_band_freq * t).astype(np.complex64)
    
    params = ExtractionParams(
        source_rate=source_rate,
        center_offset=0,
        target_bandwidth=2e6,
        stopband_db=60.0,
    )
    extractor = SubbandExtractor(params)
    output = extractor.extract(test_signal)
    
    # Check that output power is severely attenuated
    input_power = np.mean(np.abs(test_signal) ** 2)
    output_power = np.mean(np.abs(output) ** 2)
    
    attenuation_db = 10 * np.log10(output_power / input_power + 1e-12)
    assert attenuation_db < -50, f"Only {attenuation_db:.1f} dB attenuation"
```

### 5. DC Offset Removal Tests

```python
def test_dc_removal():
    """Verify DC offset is removed."""
    source_rate = 20e6
    
    # Create signal with large DC offset
    t = np.arange(int(source_rate * 0.01)) / source_rate
    signal = np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    signal_with_dc = (signal + 0.5 + 0.3j).astype(np.complex64)  # Add DC
    
    params = ExtractionParams(
        source_rate=source_rate,
        center_offset=0,
        target_bandwidth=5e6,
        remove_dc=True,
        normalize=False,
    )
    extractor = SubbandExtractor(params)
    output = extractor.extract(signal_with_dc)
    
    # Check DC component is near zero
    dc_component = np.abs(np.mean(output))
    assert dc_component < 0.01, f"DC component {dc_component:.3f}, expected ~0"
```

## Validation Script

```python
#!/usr/bin/env python3
"""validate_extraction.py - Validate sub-band extraction quality."""

import numpy as np
import matplotlib.pyplot as plt
from backend.dsp.subband_extractor import SubbandExtractor, ExtractionParams

def validate_extraction(source_file, params):
    """Generate validation plots for extraction."""
    
    # Read source data
    # ... load from RFCAP ...
    
    extractor = SubbandExtractor(params)
    output = extractor.extract(source_data)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 1. Source spectrogram
    # 2. Output spectrogram
    # 3. Filter frequency response
    # 4. Passband detail
    # 5. Input PSD
    # 6. Output PSD
    
    plt.tight_layout()
    plt.savefig('extraction_validation.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    params = ExtractionParams(
        source_rate=20e6,
        center_offset=1e6,
        target_bandwidth=2e6,
        stopband_db=60.0,
    )
    validate_extraction('test_capture.rfcap', params)
```

## Performance Benchmarks

```python
def benchmark_extraction():
    """Benchmark extraction performance."""
    import time
    
    params = ExtractionParams(
        source_rate=20e6,
        center_offset=0,
        target_bandwidth=2e6,
    )
    extractor = SubbandExtractor(params)
    
    # Test different input sizes
    for duration_sec in [1, 5, 10, 30, 60]:
        samples = int(20e6 * duration_sec)
        test_data = np.random.randn(samples).astype(np.complex64)
        
        start = time.perf_counter()
        output = extractor.extract(test_data)
        elapsed = time.perf_counter() - start
        
        mb_input = samples * 8 / 1e6
        mb_output = len(output) * 8 / 1e6
        rate = samples / elapsed / 1e6
        
        print(f"{duration_sec:3d}s ({mb_input:6.1f} MB → {mb_output:5.1f} MB): "
              f"{elapsed:.2f}s ({rate:.1f} Msamp/s)")
```

Expected results on Jetson Orin:
| Duration | Input | Output | Time | Rate |
|----------|-------|--------|------|------|
| 1s | 160 MB | 40 MB | 0.5s | 40 Msamp/s |
| 10s | 1.6 GB | 400 MB | 5s | 40 Msamp/s |
| 60s | 9.6 GB | 2.4 GB | 30s | 40 Msamp/s |

## Implementation Checklist

- [ ] Create backend/dsp/ directory
- [ ] Implement SubbandExtractor class
- [ ] Add 60 dB stopband Kaiser filter design
- [ ] Add DC offset removal
- [ ] Add unit tests for filter response
- [ ] Add aliasing detection tests
- [ ] Add WebSocket command for extraction
- [ ] Update capture_subband.py to use new DSP
- [ ] Add Flutter extraction params to CaptureRequest
- [ ] Add extraction progress UI
- [ ] Run benchmark on target hardware
- [ ] Validate against training data quality requirements
