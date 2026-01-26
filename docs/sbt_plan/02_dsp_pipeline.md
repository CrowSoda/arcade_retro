# Part 2: DSP Pipeline Design

## Canonical Pipeline: Mix → Filter → Decimate

**Order is NON-NEGOTIABLE.** Decimation without prior filtering causes aliasing that cannot be undone.

```
┌─────────────┐     ┌───────────────┐     ┌───────────┐
│  Freq Shift │ --> │ Lowpass Filter│ --> │ Decimate  │
│  (NCO Mix)  │     │ (Anti-alias)  │     │   (M:1)   │
└─────────────┘     └───────────────┘     └───────────┘
```

### Step 1: Frequency Translation

Shift target signal to baseband (DC):

```python
# Current implementation (correct)
t = np.arange(len(iq_data)) / sample_rate
iq_shifted = iq_data * np.exp(-2j * np.pi * freq_offset * t)
```

For GPU acceleration:
```python
# PyTorch version (batched)
t = torch.arange(len(iq_data), device='cuda') / sample_rate
nco = torch.exp(-2j * torch.pi * freq_offset * t)
iq_shifted = iq_data * nco
```

### Step 2: Anti-Aliasing Filter

**Critical parameters:**
- Cutoff: `target_bandwidth / 2`
- Stopband attenuation: **60-80 dB** (not the default 40 dB!)
- Transition bandwidth: 10-20% of passband

```python
from scipy.signal import firwin, kaiserord

def design_aa_filter(cutoff_hz, sample_rate, stopband_db=60):
    """Design anti-aliasing filter with specified stopband attenuation."""
    # Transition bandwidth = 10% of cutoff
    transition_width = cutoff_hz * 0.1
    
    # Kaiser window design for specified attenuation
    numtaps, beta = kaiserord(stopband_db, transition_width / (sample_rate / 2))
    
    # Ensure odd number of taps (linear phase)
    if numtaps % 2 == 0:
        numtaps += 1
    
    # Minimum 63 taps for reasonable response
    numtaps = max(numtaps, 63)
    
    taps = firwin(numtaps, cutoff_hz, window=('kaiser', beta), fs=sample_rate)
    return taps
```

### Step 3: Decimation

Keep every M-th sample where M = Fs_in / Fs_out:

```python
from scipy.signal import resample_poly

def decimate_iq(iq_filtered, source_rate, target_rate):
    """Rational resampling with anti-aliasing."""
    # Find rational approximation
    from fractions import Fraction
    ratio = Fraction(target_rate / source_rate).limit_denominator(100)
    up, down = ratio.numerator, ratio.denominator
    
    # resample_poly handles the combined interpolate-filter-decimate
    return resample_poly(iq_filtered, up, down)
```

## Multi-Stage Decimation (Large Ratios)

For decimation factors >10, use cascade:

```
┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│ CIC Decimate │ --> │ Halfband × 3   │ --> │ Compensation  │
│   (8-16x)    │     │ (2×2×2 = 8x)   │     │    FIR        │
└──────────────┘     └────────────────┘     └───────────────┘
```

Example: 20 MHz → 500 kHz (40:1 decimation)
1. CIC: 20 MHz → 2.5 MHz (8:1)
2. Halfband: 2.5 MHz → 1.25 MHz (2:1)
3. Halfband: 1.25 MHz → 625 kHz (2:1)
4. Compensation FIR: passband droop correction

## DC Offset Removal

Direct-conversion receivers have LO leakage creating DC spike:

```python
def remove_dc_offset(iq_data, alpha=0.01):
    """Exponential moving average DC removal."""
    dc_estimate = 0
    output = np.zeros_like(iq_data)
    for i, sample in enumerate(iq_data):
        dc_estimate = alpha * sample + (1 - alpha) * dc_estimate
        output[i] = sample - dc_estimate
    return output

# Or vectorized (block processing):
def remove_dc_block(iq_data):
    """Remove DC offset (mean of block)."""
    return iq_data - np.mean(iq_data)
```

## Sample Rate Margin

For bandwidth B, target sample rate = **2.5-3× B**:

```python
def calculate_target_rate(signal_bandwidth_hz):
    """Calculate target sample rate with margin."""
    # 2.5x gives margin for filter transition band
    return signal_bandwidth_hz * 2.5
```

The 80% rule: Only use central 80% of extracted bandwidth as "clean" signal.
