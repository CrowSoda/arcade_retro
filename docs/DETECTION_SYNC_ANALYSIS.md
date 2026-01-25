# Detection Sync Analysis - Answers to Code Review Questions

## 1. FFT & STFT Configuration

| Parameter | Waterfall Display | Inference |
|-----------|-------------------|-----------|
| **fft_size** | `32768` | `4096` |
| **hop_size** | `16384` (50% overlap) | `2048` (50% overlap) |
| **dynamic_range** | `60 dB` | `80 dB` |

**Sample rate:** `20e6` (20 MHz, hardcoded in `UnifiedIQSource.__init__`)

**How many FFT rows per 33ms chunk?**
- At 20 MHz, 33ms = `20e6 * 0.033 = 660,000 samples`
- With `fft_size=32768` and `hop=16384`:
- `num_ffts = (660000 - 32768) / 16384 + 1 ≈ 38.3 → **38 rows**`

---

## 2. Buffer & Frame Timing

**How many rows added per frame?**
- ALL of `db_rows` (the full **38 rows** from `compute_waterfall_rows`) are added via:
```python
for db_row in db_rows:
    self.waterfall_buffer.add_row(db_row)  # Called 38 times!
```

**WaterfallBuffer.height:**
- Set to `time_span_seconds * video_fps` = `5.0 * 30 = **150 rows**`

**time_span_seconds:** `5.0` (default)

**video_fps:** `30`

### THE FUNDAMENTAL PROBLEM:

| What | Expected | Actual |
|------|----------|--------|
| Buffer height | 150 rows | 150 rows |
| Rows per frame | 1 row | **38 rows** |
| Time to fill buffer | 5 seconds (150 frames) | 150÷38 = **~4 frames = 0.13 seconds** |
| Scroll speed | Real-time | **~38× faster than real-time** |

---

## 3. IQ Source / File Reading

**Samples per read_chunk():**
```python
int(self.sample_rate * duration_ms / 1000) = int(20e6 * 33 / 1000) = 660,000 samples
```

**duration_ms:** `33ms`

**Throttling:** Yes, 30fps rate limiter:
```python
sleep_time = max(0.001, frame_interval - elapsed)
await asyncio.sleep(sleep_time)
```
But this only affects how often frames are SENT, not how many ROWS are added per frame.

---

## 4. Detection Pipeline

**Is inference on same chunk as waterfall?**
- NO. Inference batches 6 chunks together (`inference_chunk_count = 6`)
- Inference sees: `6 × 660,000 = 3,960,000 samples`

**Inference spectrogram size:** `1024 × 1024` (resized via `F.interpolate`)

**How is row_offset calculated?**
```python
'row_offset': int(d.x1 * rows_in_frame)
```
- `d.x1` is normalized position (0-1) on the 1024×1024 inference spectrogram
- This is **WRONG** - it doesn't map correctly to waterfall rows

---

## 5. PSD Specific

**Where does PSD come from?**
- `WaterfallBuffer.latest_psd` stores the most recent row
- Updated in `add_row()`: `self.latest_psd = db_row.astype(np.float32)`

**How often does PSD update?**
- 38× per video frame (once per `add_row()` call)

**What clears PSD boxes?**
- Nothing - they accumulate (the memory leak bug)

---

## 6. Code Snippets (Current State)

### compute_waterfall_rows()
```python
def compute_waterfall_rows(self, iq_data: np.ndarray) -> np.ndarray:
    fft_size = self.waterfall_fft_size  # 32768
    hop_size = fft_size // 2  # 16384
    num_ffts = (len(iq_data) - fft_size) // hop_size + 1  # ~38
    
    rows = np.zeros((num_ffts, fft_size), dtype=np.float32)
    for i in range(num_ffts):
        # ... FFT computation ...
    return rows  # Shape: (num_ffts, fft_size) = (38, 32768)
```

### Main loop in run_pipeline()
```python
db_rows = self.pipeline.compute_waterfall_rows(chunk.data)
self.rows_this_frame = len(db_rows)  # ~38
for db_row in db_rows:
    self.waterfall_buffer.add_row(db_row)  # Adds ALL 38 rows
self.total_rows_written += self.rows_this_frame
```

### WaterfallBuffer.__init__()
```python
self.db_buffer = np.full((height, width), -120.0, dtype=np.float32)
# height = time_span * fps = 5 * 30 = 150
# width = 2048
```

---

## What Was Implemented vs What's Still Wrong

### ✅ Implemented:

1. **Backend tracking:** Added `total_rows_written` and `rows_this_frame` counters
2. **Detection JSON format:** Added `base_row`, `rows_in_frame`, `row_offset`, `row_span`
3. **Flutter state:** Added `totalRowsReceived`, `rowsPerFrame` to `VideoStreamState`
4. **Flutter detection model:** Added `absoluteRow`, `rowSpan` to `VideoDetection`
5. **Flutter overlay:** Changed from PTS-based to row-based positioning

### ❌ Still Wrong:

1. **FUNDAMENTAL MISMATCH:**
   - Buffer height = 150 rows (for 5s display)
   - Rows per frame = 38
   - Buffer fills in ~4 frames (0.13 seconds), NOT 5 seconds

2. **row_offset calculation is incorrect:**
   - `d.x1` is from inference spectrogram (1024×1024)
   - It does NOT map to waterfall rows (which have different FFT params)

3. **visibleRows calculation is wrong in Flutter:**
   ```dart
   final visibleRows = (timeSpan * 30 * rowsPerFrame).round();
   // = 5 * 30 * 38 = 5700 rows
   // But buffer only holds 150 rows!
   ```

---

## The Real Fix Needed

The research article assumes `rows_per_frame` is 1 (or a known constant), and buffer height equals visible rows. The current code has:

- **Buffer height:** 150 rows
- **Rows added per frame:** 38 rows
- **These don't match!**

### Options:

**Option A:** Change waterfall FFT so only 1 row is produced per 33ms chunk
- Set `fft_size` much larger, or
- Downsample 38 rows → 1 row before adding to buffer

**Option B:** Change buffer height to match actual row rate
- `WaterfallBuffer.height = time_span * fps * rows_per_frame = 5 * 30 * 38 = 5700`
- This preserves all time resolution but uses more memory

**Option C:** Keep 1 row per frame for waterfall, separate from inference
- Waterfall: 1 large FFT per chunk → 1 row
- Inference: Many small FFTs for spectrogram

### The Core Insight:

The research says:
> "Each frame still contributes 38 rows. The relationship between detection position and waterfall position remains constant regardless of timing."

But the implementation assumes:
- `WaterfallBuffer.height` = number of visible rows
- `rows_per_frame` = 38

These two need to be synchronized: either buffer holds 5700 rows, or we add 1 row per frame.
