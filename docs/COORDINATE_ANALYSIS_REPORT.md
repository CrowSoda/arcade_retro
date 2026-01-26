# Coordinate System Analysis Report: TensorCade vs G20

## Executive Summary

**CRITICAL BUG IDENTIFIED**: G20 and TensorCade use fundamentally different approaches to label coordinates. The systems are incompatible without proper translation.

---

## Part 1: TensorCade Label System (WORKING)

### 1.1 Label File Format

TensorCade reads labels from `{prefix}_labels.json`:

```json
{
  "signals": [
    {
      "time_start": 0.5,       // Seconds from file start
      "time_stop": 0.7,        // Seconds from file start
      "freq_low": 820.5,       // MHz
      "freq_high": 825.0,      // MHz
      "label": "signal"
    }
  ]
}
```

**KEY: Labels are in REAL-WORLD UNITS (seconds and MHz), NOT normalized pixels.**

### 1.2 Spectrogram Generation (ChunkWorker.run())

```python
# Using scipy.signal.stft
f_vals, t_vals, Zxx = stft(
    chunk_data,
    fs=sr,                    # Sample rate
    nperseg=nfft,             # FFT size (e.g., 1024)
    noverlap=min(noverlap, nfft - 1),
    return_onesided=False,    # Full spectrum
)

# FFT shift to center DC
f_vals_shifted = np.fft.fftshift(f_vals)  # Now: negative freqs on left, positive on right
Zxx_shift = np.fft.fftshift(Zxx, axes=0)

# Power in dB
sxx = np.abs(Zxx_shift) ** 2
sxx_db = 10 * np.log10(sxx + 1e-12)

# Normalization
vmax = sxx_db.max()
vmin = vmax - dynamic_range
sxx_clamped = np.clip(sxx_db, vmin, vmax)

# Resize to output size
resized = cv2.resize(sxx_db_float, (out_size, out_size), interpolation=cv2.INTER_AREA)

# CRITICAL: FLIP VERTICALLY
resized = np.flipud(resized)
```

**KEY: `np.flipud(resized)` flips the Y-axis so that:**
- After flip: Y=0 is HIGH frequency (top of image)
- After flip: Y=out_size is LOW frequency (bottom of image)

### 1.3 Label-to-Pixel Coordinate Conversion

```python
def box_to_chunk_coords(sig, cstart, cend, tb, fb, freqs):
    """
    Args:
        sig: Signal dict with time_start, time_stop, freq_low, freq_high
        cstart, cend: Chunk time bounds in seconds
        tb: Number of time bins in spectrogram (sxx_db_float.shape[1])
        fb: Number of freq bins in spectrogram (sxx_db_float.shape[0])
        freqs: FFT-shifted frequency axis array
    """
    t_s = sig["time_start"]
    t_e = sig["time_stop"]
    
    # Clip to chunk bounds
    if t_e < cstart or t_s > cend:
        return None  # Signal not in this chunk
    
    is_tstart = max(t_s, cstart)
    is_tstop = min(t_e, cend)
    chunk_len = cend - cstart
    
    # Convert TIME to spectrogram X coordinate (bins)
    local_t_s = is_tstart - cstart  # Seconds from chunk start
    local_t_e = is_tstop - cstart
    x_min = (local_t_s / chunk_len) * tb  # Scale to time bins
    x_max = (local_t_e / chunk_len) * tb

    # Convert FREQUENCY to spectrogram Y coordinate (bins)
    f_l = sig["freq_low"] * 1e6   # MHz to Hz
    f_h = sig["freq_high"] * 1e6
    fmn = freqs[0]   # Lowest freq in shifted array
    fmx = freqs[-1]  # Highest freq in shifted array
    
    # Clip to frequency range
    if f_h < fmn or f_l > fmx:
        return None
    f_h = min(f_h, fmx)
    f_l = max(f_l, fmn)
    
    # Find bin indices using searchsorted
    def find_bin_for_freq(ff):
        i = np.searchsorted(freqs, ff, side="left")
        return max(0, min(i, len(freqs) - 1))

    y_min_idx = find_bin_for_freq(f_l)  # Bin index for low freq
    y_max_idx = find_bin_for_freq(f_h)  # Bin index for high freq
    
    return {
        "label": sig.get("label", "signal"),
        "x_min": float(x_min),   # Time bin (NOT normalized)
        "x_max": float(x_max),
        "y_min": float(y_min_idx),  # Freq bin (NOT normalized)
        "y_max": float(y_max_idx),
    }
```

### 1.4 Scaling to Final Image Size

```python
# After computing spectrogram: sxx_db_float.shape = (num_freq_bins, num_time_frames)
# After resize: out_size x out_size

scale_x = out_size / float(sxx_db_float.shape[1])  # Time scaling
scale_y = out_size / float(sxx_db_float.shape[0])  # Freq scaling

x1 = bc["x_min"] * scale_x
x2 = bc["x_max"] * scale_x
y1 = bc["y_min"] * scale_y
y2 = bc["y_max"] * scale_y
```

### 1.5 Image Saving

```python
# CRITICAL: origin='lower' makes Y=0 the BOTTOM of the saved image
plt.imsave(out_png, resized, cmap="gray", origin="lower")
```

**Combined with `np.flipud`, this means:**
- Low Y values (low freq bins) → bottom of image
- High Y values (high freq bins) → top of image
- **This matches standard spectrogram convention: high freq at top**

### 1.6 JSON Box Format Saved

```json
{
  "image": "chunk_xxx_001.png",
  "width": 1024,
  "height": 1024,
  "bboxes": [
    {
      "label": "signal",
      "x_min": 123.5,    // Pixel coordinates
      "y_min": 456.2,    // Pixel coordinates  
      "x_max": 234.1,
      "y_max": 567.8
    }
  ]
}
```

---

## Part 2: G20 Label System (CURRENT - BROKEN)

### 2.1 Flutter Label Input (training_spectrogram.dart)

```dart
// User draws box on spectrogram display
// Coordinates are NORMALIZED 0-1

void _createBox(double x1, double y1, double x2, double y2) {
    // Convert normalized coords to ABSOLUTE time
    final absTimeStart = _windowStartSec + min(x1, x2) * _windowLengthSec;
    final absTimeEnd = _windowStartSec + max(x1, x2) * _windowLengthSec;
    
    // Calculate frequency bounds (but using DIFFERENT formula)
    final bwHz = widget.header!.bandwidthHz;
    final cfHz = widget.header!.centerFreqHz;
    freqStartMHz = (cfHz - bwHz/2 + (1 - max(y1, y2)) * bwHz) / 1e6;
    freqEndMHz = (cfHz - bwHz/2 + (1 - min(y1, y2)) * bwHz) / 1e6;
    
    final box = LabelBox(
      x1: x1, y1: y1, x2: x2, y2: y2,  // NORMALIZED 0-1
      timeStartSec: absTimeStart,
      timeEndSec: absTimeEnd,
      freqStartMHz: freqStartMHz,
      freqEndMHz: freqEndMHz,
    );
}
```

**PROBLEM 1: Flutter sends NORMALIZED (0-1) coordinates directly, not real units.**

### 2.2 Flutter Spectrogram Generation

```dart
// From _computeSpectrogram()

// Compute FFT frames
for (int timeFrame = 0; timeFrame < numTimeFrames; timeFrame++) {
    // ... FFT computation ...
    
    // Power spectrum - store as column
    for (int freqBin = 0; freqBin < _spectrogramHeight; freqBin++) {
        // FFT shift
        final fftBin = (freqBin + fftSize ~/ 2) % fftSize;
        
        // Row = frequency (INVERTED so high freq at top)
        final row = _spectrogramHeight - 1 - freqBin;
        _spectrogramData![row * _spectrogramWidth + timeFrame] = dB;
    }
}
```

**KEY: Flutter already inverts Y so row=0 is HIGH frequency.**

### 2.3 Data Sent to Backend (training_provider.dart)

```dart
// For each box in trainFromFile():
final box = boxes[i];
final timeStartSec = box['time_start_sec'];  // Absolute time
final timeEndSec = box['time_end_sec'];

// Calculate IQ offsets
final offsetSamples = (timeStartSec * header.sampleRate).toInt();
final numSamples = (durationSec * header.sampleRate).toInt();

// Read IQ data
final iqData = await RfcapService.readIqDataRaw(
    rfcapPath,
    offsetSamples: offsetSamples,
    numSamples: numSamples,
);

// Convert box to normalized TrainingBox
final trainingBox = TrainingBox(
    x1: box['x1'],  // NORMALIZED 0-1
    y1: box['y1'],  // NORMALIZED 0-1
    x2: box['x2'],
    y2: box['y2'],
);

// Send to backend
_send({
    'command': 'save_sample',
    'iq_data': base64(iqData),
    'boxes': [trainingBox.toJson()],  // Normalized coords
    'metadata': {
        'time_offset_sec': timeStartSec,
        'duration_sec': durationSec,
    },
});
```

**PROBLEM 2: Backend receives NORMALIZED coordinates, not spectrogram bin indices.**

### 2.4 Backend Spectrogram Generation (sample_manager.py)

```python
def compute_spectrogram(self, iq_data: np.ndarray) -> np.ndarray:
    nfft = INFERENCE_FFT_SIZE      # e.g., 4096
    hop = INFERENCE_HOP_LENGTH     # e.g., 1024
    dynamic_range = INFERENCE_DYNAMIC_RANGE_DB  # e.g., 50
    
    # Compute STFT
    window = np.hanning(nfft)
    frames = []
    for i in range(num_frames):
        start = i * hop
        frame = iq_data[start:start + nfft] * window
        spectrum = np.fft.fftshift(np.fft.fft(frame))
        power = np.abs(spectrum) ** 2 + 1e-10
        power_db = 10 * np.log10(power)
        frames.append(power_db)
    
    spectrogram = np.stack(frames, axis=1)  # (nfft, num_frames)
    
    # Normalize
    max_val = spectrogram.max()
    min_val = max_val - dynamic_range
    spectrogram = np.clip(spectrogram, min_val, max_val)
    spectrogram = (spectrogram - min_val) / dynamic_range
    
    # Resize to output size
    from scipy.ndimage import zoom
    target_h, target_w = INFERENCE_OUTPUT_SIZE  # e.g., (1024, 1024)
    zoom_factors = (target_h / spectrogram.shape[0], target_w / spectrogram.shape[1])
    spectrogram = zoom(spectrogram, zoom_factors, order=1)
    
    # Convert to uint8
    spectrogram = (spectrogram * 255).astype(np.uint8)
    
    return spectrogram
```

**PROBLEM 3: NO `np.flipud`! Y=0 is LOW frequency in the output.**

### 2.5 Box Coordinate Conversion (sample_manager.py)

```python
# Convert normalized boxes to pixel coordinates
h, w = INFERENCE_OUTPUT_SIZE  # (1024, 1024)

for box in boxes:
    x1_px = int(box["x1"] * w)  # Direct multiplication
    y1_px = int(box["y1"] * h)  # Direct multiplication
    x2_px = int(box["x2"] * w)
    y2_px = int(box["y2"] * h)
    
    # Ensure min < max
    x_min = min(x1_px, x2_px)
    x_max = max(x1_px, x2_px)
    y_min = min(y1_px, y2_px)
    y_max = max(y1_px, y2_px)
    
    pixel_boxes.append({
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    })
```

**PROBLEM 4: No Y-flip applied when converting coordinates.**

### 2.6 Dataset Loading (dataset.py)

```python
def __getitem__(self, idx):
    # Load spectrogram
    data = np.load(npz_path)
    spec = data['spectrogram']  # uint8 (1024, 1024)
    
    # Read box from JSON
    with open(json_path) as f:
        metadata = json.load(f)
    
    for b in metadata.get("boxes", []):
        x_min = b["x_min"]
        y_min = b["y_min"]
        x_max = b["x_max"]
        y_max = b["y_max"]
        
        # Y-FLIP FIX (recently added)
        h = spec.shape[0]
        y_min_new = h - y_max
        y_max_new = h - y_min
        
        boxes.append([x_min, y_min_new, x_max, y_max_new])
```

**PARTIAL FIX: Y-flip in dataset.py, but spectrogram itself is NOT flipped.**

---

## Part 3: Detailed Comparison Table

| Aspect | TensorCade | G20 |
|--------|-----------|-----|
| **Label Units** | Real (seconds, MHz) | Normalized (0-1) |
| **Label Source** | JSON file | Flutter UI |
| **Spectrogram np.flipud** | ✅ YES | ❌ NO |
| **plt.imsave origin** | 'lower' | N/A (saves raw) |
| **Y=0 in saved image** | Low freq (bottom) | Low freq (TOP) |
| **Y=0 in display** | High freq (top) | High freq (top) |
| **Coord conversion** | Real → bins → pixels | Normalized → pixels |
| **Y-flip for boxes** | Not needed (image flipped) | Tried in dataset.py |

---

## Part 4: The Bug Chain

### Step 1: Flutter renders spectrogram with Y inverted
```
Flutter: row=0 → high frequency (matches visual expectation)
```

### Step 2: User draws box at visual position (near top = high freq)
```
Flutter box: y1=0.1, y2=0.3 (near top of display)
```

### Step 3: Box sent to Python as normalized coords
```
Python receives: y1=0.1, y2=0.3
```

### Step 4: Python generates spectrogram WITHOUT flipud
```
Python spectrogram: row=0 → LOW frequency
Python spectrogram: row=1023 → HIGH frequency
```

### Step 5: Python converts box directly
```
y_min = 0.1 * 1024 = 102
y_max = 0.3 * 1024 = 307
Box is at rows 102-307, which is LOW frequency region
```

### Step 6: The Y-flip fix in dataset.py
```
y_min_new = 1024 - 307 = 717
y_max_new = 1024 - 102 = 922
Box is now at rows 717-922
```

### Step 7: BUT the spectrogram data itself is NOT flipped!
```
Row 717-922 in the unflipped spectrogram = HIGH frequency
But the MODEL sees the image as-is without understanding the flip
```

### Result: MISMATCH
- Visual box (Flutter): High frequency region
- Training spectrogram: Row 717-922
- But visual appearance of row 717-922: LOW frequency (dark region)

---

## Part 5: SOLUTION

### Option A: Match TensorCade (Recommended)

1. **Add `np.flipud` to Python spectrogram generation:**
```python
spectrogram = zoom(spectrogram, zoom_factors, order=1)
spectrogram = np.flipud(spectrogram)  # ADD THIS
spectrogram = (spectrogram * 255).astype(np.uint8)
```

2. **Remove Y-flip from dataset.py:**
```python
# Remove this:
# y_min_new = h - y_max
# y_max_new = h - y_min

# Use raw coordinates:
boxes.append([x_min, y_min, x_max, y_max])
```

### Option B: Keep Python orientation, fix everywhere

1. Keep spectrogram as-is (no flipud)
2. Flip Y coordinates ONCE when saving sample (in sample_manager.py)
3. Remove flip from dataset.py

---

## Part 6: Verification Command

After fix, run:
```bash
cd g20_demo/backend
python verify_sample.py Creamy_Pork
```

If the RED BOX overlays the BRIGHT SIGNAL area → fix is correct.
If the RED BOX is in DARK area → coordinate mismatch still exists.
