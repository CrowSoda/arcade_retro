# Box Positioning Investigation - Findings

## Investigation Summary

This document presents the findings from a comprehensive code analysis of the detection box positioning system. Debug logging has been added to both Flutter and Python backends to capture runtime values.

---

## Section A: Detection Data Values (Flutter Side)

### Debug Logging Added
Location: `_DetectionOverlayLayer.build()` in `video_waterfall_display.dart`

```dart
debugPrint('=== DETECTION DEBUG ===');
debugPrint('totalRowsReceived: $totalRowsReceived');
debugPrint('bufferHeight: $bufferHeight');
debugPrint('detections count: ${detections.length}');

for (int i = 0; i < detections.length && i < 5; i++) {
  final det = detections[i];
  final rowsAgo = totalRowsReceived - det.absoluteRow;
  debugPrint('Det[$i]: absoluteRow=${det.absoluteRow}, rowsAgo=$rowsAgo, '
      'y1=${det.y1}, y2=${det.y2}, x1=${det.x1}, x2=${det.x2}, rowSpan=${det.rowSpan}');
}
```

### Expected Values

| Field | Expected Range | Meaning |
|-------|---------------|---------|
| `totalRowsReceived` | Large, incrementing | Monotonic row counter |
| `bufferHeight` | ~5700 (5s × 30fps × 38rows) | Visible rows on screen |
| `absoluteRow` | < totalRowsReceived | Row when detection was made |
| `rowsAgo` | 0 to bufferHeight | 0=bottom, bufferHeight=top |
| `y1, y2` | 0.0 to 1.0 | Normalized frequency position |
| `x1, x2` | 0.0 to 1.0 | Normalized time position (inference) |
| `rowSpan` | 1 to ~50 | Detection height in rows |

---

## Section B: Coordinate System Analysis

### Backend Detection Creation (`unified_pipeline.py`)

```python
# In process_chunk():
det = Detection(
    box_id=i,
    x1=float(box[0]) / 1024,  # Normalized by inference image size
    y1=float(box[1]) / 1024,
    x2=float(box[2]) / 1024,
    y2=float(box[3]) / 1024,
    ...
)
```

### Coordinate System Confirmed

| Aspect | Value | Notes |
|--------|-------|-------|
| Normalization | **0.0 to 1.0** | Divided by 1024 (inference image size) |
| Raw box format | `[x1, y1, x2, y2]` | From Faster R-CNN output |
| x1/x2 meaning | **Time axis** | Horizontal on inference spectrogram |
| y1/y2 meaning | **Frequency axis** | Vertical on inference spectrogram |

### Inference Spectrogram Generation

```python
# In compute_spectrogram():
# 1. STFT produces: (frequency_bins, time_steps)
# 2. Resized to 1024×1024
# 3. torch.flip(resized, dims=[2]) - FLIPS TIME AXIS

Zxx = torch.stft(...)  # Shape: (4096, time_steps)
resized = F.interpolate(sxx_norm, size=(1024, 1024), ...)
resized = torch.flip(resized, dims=[2])  # Flip dimension 2 (time)
```

**Key Finding:** The `torch.flip(resized, dims=[2])` flips the time axis. This may affect interpretation of x1/x2 values.

---

## Section C: Axis Mapping Verification

### Inference Spectrogram Orientation
```
Inference Image (1024×1024):
┌─────────────────────────────────────┐
│  Y-axis (dim 2) = Frequency         │
│  ↑                                  │
│  │                                  │
│  │        (Model sees this)         │
│  │                                  │
│  └──────────────────────────────────→ X-axis (dim 3) = Time
│                                      (after flip)
└─────────────────────────────────────┘
```

### Model Output Coordinates
- `box[0]` (x1) = Time start (pixels, 0-1024)
- `box[1]` (y1) = Frequency start (pixels, 0-1024)
- `box[2]` (x2) = Time end (pixels, 0-1024)
- `box[3]` (y2) = Frequency end (pixels, 0-1024)

### Flutter Display Mapping
```dart
// In _DetectionOverlayLayer.build():

// FREQUENCY → Horizontal (X on screen)
final left = det.y1 * plotWidth;   // y1 → left edge
final right = det.y2 * plotWidth;  // y2 → right edge

// TIME → Vertical (Y on screen) via row index
final rowsAgo = totalRowsReceived - det.absoluteRow;
final boxBottom = plotHeight - (rowsAgo * pixelsPerRow);
```

**This mapping is CORRECT:**
- Model y1/y2 (frequency) → Screen horizontal position
- Model x1/x2 (time) → Converted to row offset → Screen vertical position

---

## Section D: Overflow Error Analysis

### Source of "BOTTOM OVERFLOWED BY X PIXELS"
This is a **Flutter runtime warning**, not code we wrote. It occurs when a widget's bounds exceed its parent's bounds.

### Potential Causes

1. **`boxTop` is negative:**
   ```dart
   final boxTop = boxBottom - boxHeight;
   // If boxBottom < boxHeight, boxTop becomes negative
   ```

2. **`boxBottom` exceeds `plotHeight`:**
   ```dart
   final boxBottom = plotHeight - (rowsAgo * pixelsPerRow);
   // If rowsAgo is 0 and boxHeight > 0, boxBottom = plotHeight
   // Then boxTop.clamp() may not prevent the overflow
   ```

3. **`rowsAgo` is negative:**
   ```dart
   final rowsAgo = totalRowsReceived - det.absoluteRow;
   // If det.absoluteRow > totalRowsReceived, rowsAgo is negative
   // This would place the box BELOW the screen bottom
   ```

### Current Safeguards (May Be Insufficient)
```dart
// Skip if outside visible range
if (rowsAgo < 0 || rowsAgo >= bufferHeight) continue;

// Clamping (but children may still overflow)
left: left.clamp(0.0, plotWidth - boxWidth),
top: boxTop.clamp(0.0, plotHeight - boxHeight),
```

### Likely Root Cause
The `_DetectionBoxWidget` contains a `Column` with children (label, debug text). Even if the `Positioned` widget is clamped, the **Column's children may overflow** if the box is too small to contain them.

```dart
child: Column(
  children: [
    Container(/* Label - has fixed height */),
    if (showDebug) Container(/* Debug text - additional height */),
  ],
),
```

---

## Section E: Backend Detection JSON Format

### Debug Logging Added
```python
if det_list:
    print(f"[DETECTION DEBUG] Frame {frame_id} - Sending {len(det_list)} detections:")
    print(f"  base_row={base_row}, rows_in_frame={rows_in_frame}")
    for i, d in enumerate(det_list[:3]):
        print(f"  [{i}] {json.dumps(d)}")
        absolute_row = base_row + d['row_offset']
        print(f"      -> absoluteRow={absolute_row}")
```

### JSON Message Format
```json
{
  "type": "detection_frame",
  "frame_id": 42,
  "pts": 1.234,
  "inference_ms": 45.6,
  "base_row": 15200,
  "rows_in_frame": 38,
  "detections": [
    {
      "detection_id": 0,
      "x1": 0.123,
      "y1": 0.456,
      "x2": 0.234,
      "y2": 0.567,
      "confidence": 0.95,
      "class_id": 1,
      "class_name": "creamy_chicken",
      "row_offset": 4,
      "row_span": 8
    }
  ]
}
```

---

## Section F: Row Calculation Documentation

### Backend Side (Python)

```python
# In VideoStreamServer:
self.total_rows_written = 0  # Monotonic counter

# In run_pipeline():
self.total_rows_written += self.rows_this_frame  # Updated AFTER sending strip

# In _run_inference_async():
base_row = self.total_rows_written  # Captured BEFORE inference

# Detection row calculation:
row_offset = int(d.x1 * rows_in_frame)  # x1 is time position (0-1)
row_span = max(1, int((d.x2 - d.x1) * rows_in_frame))
```

### Flutter Side (Dart)

```dart
// In video_stream_provider.dart _handleDetection():
final baseRow = data['base_row'] as int? ?? state.totalRowsReceived;
final rowsInFrame = data['rows_in_frame'] as int? ?? 38;

// VideoDetection.fromJson():
absoluteRow: baseRow + rowOffset,
rowSpan: rowSpan > 0 ? rowSpan : 1,

// In _handleStrip():
state = state.copyWith(
  totalRowsReceived: totalRows + rowsInStrip,  // ⚠️ POTENTIAL BUG!
);
```

### ⚠️ CRITICAL BUG FOUND

In `_handleStrip()`:
```dart
// totalRows from header is the CURRENT count before this strip
// Then we ADD rowsInStrip to get new count
totalRowsReceived: totalRows + rowsInStrip,
```

**But in the backend:**
```python
# Header contains total_rows_written BEFORE adding this strip
header = struct.pack('...', ..., self.total_rows_written, ...)

# THEN we add:
self.total_rows_written += self.rows_this_frame
```

**This means:**
- Backend sends `total_rows_written` = 1000 (before strip)
- Flutter receives, calculates `totalRowsReceived = 1000 + 38 = 1038`
- Backend then sets `total_rows_written = 1038`
- **They match!** ✓

Actually, this is correct. Let me verify once more...

Looking more carefully:
- `totalRows` from header = backend's `total_rows_written` (BEFORE sending current strip)
- Flutter sets `totalRowsReceived = totalRows + rowsInStrip`
- Backend then does `total_rows_written += rows_this_frame`

This is **correct and synchronized**.

---

## Section G: Visual Quality Issues

### Issue 1: Box Height Calculation
```dart
final boxHeight = (det.rowSpan * pixelsPerRow).clamp(8.0, plotHeight * 0.3);
```

- If `rowSpan` is very small (1-2), box is ~0.1-0.2 pixels tall
- Clamped to minimum 8.0 pixels, which is reasonable
- Maximum 30% of plot height seems arbitrary

### Issue 2: Label Overflow
```dart
child: Column(
  children: [
    Container(/* ~12px height for label */),
    if (showDebug) Container(/* ~10px height for debug */),
  ],
)
```

If `boxHeight` is clamped to 8.0 but label needs ~12px, **the label overflows**.

### Issue 3: Boxes Too Small
With `bufferHeight = 5700` and `plotHeight` typically ~400-600 pixels:
```
pixelsPerRow = 500 / 5700 = 0.088 pixels per row
```

A detection spanning 10 rows would only be **0.88 pixels tall**, clamped to 8.0.

---

## Data Collection Template (Code Analysis)

```
=== COLLECTED DATA ===

FLUTTER SIDE:
- totalRowsReceived: Increments by ~38 per frame (monotonic)
- bufferHeight: 5700 (default for 5s @ 30fps × 38 rows)
- plotWidth (pixels): Layout-dependent (~1600-1900)
- plotHeight (pixels): Layout-dependent (~400-600)
- pixelsPerRow: plotHeight / bufferHeight ≈ 0.07-0.1

SAMPLE DETECTION (expected):
- absoluteRow: baseRow + row_offset (e.g., 15200 + 4 = 15204)
- rowsAgo: totalRowsReceived - absoluteRow (should be 0-5700)
- y1: 0.0-1.0 (normalized frequency)
- y2: 0.0-1.0 (normalized frequency, y2 > y1)
- x1: 0.0-1.0 (normalized time)
- x2: 0.0-1.0 (normalized time, x2 > x1)
- rowSpan: (x2-x1) × rows_in_frame (typically 5-50)
- Calculated boxTop: plotHeight - (rowsAgo + rowSpan) × pixelsPerRow
- Calculated boxBottom: plotHeight - rowsAgo × pixelsPerRow
- Calculated boxLeft: y1 × plotWidth
- Calculated boxWidth: (y2 - y1) × plotWidth

BACKEND SIDE:
- total_rows_written: Monotonic, increments by ~38 per frame
- rows_per_frame: ~38 (depends on FFT settings)
- inference_chunk_count: 6 frames combined for inference
- Raw detection box from model: [x1, y1, x2, y2] in pixels (0-1024)
- Normalized detection: x1, y1, x2, y2 all 0.0-1.0
- absolute_row calculation: base_row + (x1 × rows_in_frame)

COORDINATE SYSTEM:
- Inference spectrogram dimensions: 1024×1024
- Waterfall width: 2048 pixels
- Waterfall height (buffer): 5700 rows (default)
- X axis on inference = TIME (after flip)
- Y axis on inference = FREQUENCY
- y1/y2 in detection = FREQUENCY (maps to horizontal)
- x1/x2 in detection = TIME (maps to row offset)
```

---

## Identified Issues

### 1. Label Overflow in Small Boxes
**Problem:** Box height clamped to 8px minimum, but label content needs ~12-22px.
**Fix:** Conditionally hide labels when box is too small, or use tooltip instead.

### 2. Very Small Pixel-per-Row Ratio
**Problem:** With 5700 rows in ~500px, each row is ~0.088 pixels.
**Impact:** Boxes appear as thin horizontal lines.
**Fix:** Consider using time-based height instead of row-based, or limit visible detection range.

### 3. Potential Race Condition
**Problem:** Inference runs asynchronously. By the time detection arrives, more strips have been received.
**Symptom:** `base_row` might be stale relative to current `totalRowsReceived`.
**Fix:** This is partially mitigated by capturing `base_row` before inference, but verify with debug logs.

### 4. Time Axis Flip
**Note:** The `torch.flip(resized, dims=[2])` in spectrogram generation may need corresponding adjustment in `row_offset` calculation if boxes appear inverted.

---

## Proposed Fixes

### Fix 1: Handle Label Overflow
```dart
// In _DetectionBoxWidget.build():
final showLabel = constraints.maxHeight >= 16;  // Need ~16px for label
final showDebug = debugInfo != null && constraints.maxHeight >= 24;

return Container(
  // ...
  child: showLabel ? Column(
    children: [
      /* label */,
      if (showDebug) /* debug */,
    ],
  ) : const SizedBox.shrink(),
);
```

### Fix 2: Use ConstrainedBox
```dart
child: ConstrainedBox(
  constraints: BoxConstraints(maxHeight: boxHeight),
  child: _DetectionBoxWidget(...),
),
```

### Fix 3: Clip Children
```dart
return ClipRect(
  child: Container(
    // existing code
  ),
);
```

---

## Next Steps

1. **Run the app** with debug logging enabled to capture actual runtime values
2. **Compare** captured values with expected ranges in the table above
3. **Identify** which value is out of expected range
4. **Apply** the appropriate fix from the proposed fixes section

The debug logging added to both Flutter and Python will output to console when the app runs, providing the actual runtime data needed to pinpoint the exact cause of the overflow errors.
