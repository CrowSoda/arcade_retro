# Detection Box Pipeline Validation

## Date: 2026-01-25

## Pipeline Summary

```
IQ Data
    ↓
Backend: FFT → 1024×1024 spectrogram
    ↓
Backend: Faster R-CNN → box coords (pixels 0-1024)
    ↓
Backend: Normalize to 0-1, calculate row_offset
    ↓
WebSocket JSON
    ↓
Flutter: Parse, calculate absoluteRow
    ↓
Flutter: Render on waterfall (boxes track vertically)
    ↓
Flutter: Render on PSD (frequency bands)
```

## Coordinate System Analysis

### Backend (unified_pipeline.py)

1. **Spectrogram Generation:**
   - STFT creates `[freq_bins, time_frames]` tensor
   - `fftshift` on dim=0 (freq axis) centers DC
   - Resized to 1024×1024 via bilinear interpolation
   - `torch.flip(resized, dims=[2])` flips frequency axis

2. **Detection Coordinates:**
   - Model outputs boxes in pixel coords [0-1024]
   - Normalized to [0-1]: `x1=box[0]/1024, y1=box[1]/1024, ...`
   - **x1/x2 = TIME axis** (columns = time frames)
   - **y1/y2 = FREQUENCY axis** (rows = freq bins)

3. **Row Tracking:**
   ```python
   'row_offset': int(d.x1 * rows_in_frame),
   'row_span': max(1, int((d.x2 - d.x1) * rows_in_frame)),
   ```

### Flutter Parsing (video_stream_provider.dart)

1. **Detection JSON Received:**
   - `base_row`: Absolute row at time of inference
   - `row_offset`: Offset within frame (from x1)
   - `row_span`: Number of rows (from x2-x1)
   - `y1, y2`: Frequency position [0-1]

2. **absoluteRow Calculation:**
   ```dart
   absoluteRow: baseRow + rowOffset,
   ```

### Waterfall Rendering (video_waterfall_display.dart)

1. **Frequency (X axis):**
   ```dart
   left = det.y1 * plotWidth
   right = det.y2 * plotWidth
   ```

2. **Time (Y axis):**
   ```dart
   rowsAgo = totalRowsReceived - det.absoluteRow
   boxTop = plotHeight - (rowsAgo * pixelsPerRow) - boxHeight
   ```

### PSD Rendering (psd_chart.dart)

1. **Frequency bands use same y1/y2:**
   ```dart
   left = det.y1 * plotWidth
   width = (det.y2 - det.y1) * plotWidth
   ```

## Fixes Applied

### 1. 20% Padding on Waterfall Boxes

**Frequency axis (horizontal):**
```dart
final detWidth = det.y2 - det.y1;
final freqPadding = detWidth * 0.2;  // 20% padding each side
final paddedY1 = (det.y1 - freqPadding).clamp(0.0, 1.0);
final paddedY2 = (det.y2 + freqPadding).clamp(0.0, 1.0);
final left = paddedY1 * plotWidth;
final right = paddedY2 * plotWidth;
final boxWidth = (right - left).abs().clamp(8.0, plotWidth);
```

**Time axis (vertical):**
```dart
final baseHeight = det.rowSpan * pixelsPerRow;
final boxHeight = (baseHeight * 1.4).clamp(12.0, plotHeight * 0.3);  // 40% total padding

// Center the padding
final heightPadding = (boxHeight - baseHeight) / 2;
final boxBottom = plotHeight - (rowsAgo * pixelsPerRow) + heightPadding;
final boxTop = boxBottom - boxHeight;
```

### 2. 20% Padding on PSD Bands

```dart
final detWidth = det.y2 - det.y1;
final freqPadding = detWidth * 0.2;  // 20% padding each side
final paddedY1 = (det.y1 - freqPadding).clamp(0.0, 1.0);
final paddedY2 = (det.y2 + freqPadding).clamp(0.0, 1.0);

final left = paddedY1 * plotWidth;       
final width = (paddedY2 - paddedY1) * plotWidth;
```

### 3. Simplified Box Widget

```dart
class _DetectionBoxWidget extends StatelessWidget {
  final VideoDetection detection;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: color, width: 2),
        color: color.withOpacity(0.15),
      ),
    );
  }
}
```

## Debug Logging Added

### Backend (unified_pipeline.py)
- `[DET RAW]` - Raw detection coordinates from model
- `[DETECTION DEBUG]` - Detection JSON being sent

### Flutter (video_stream_provider.dart)
- `[VideoStream] DETECTION RECEIVED` - Incoming detection data
- `[VideoStream] Parsed` - Parsed detection values

### Flutter (video_waterfall_display.dart)
- `[WATERFALL]` - Plot dimensions and detection count
- `[WATERFALL BOX]` - Individual box positioning

### Flutter (psd_chart.dart)
- `[PSD]` - Plot dimensions and detection count
- `[PSD BAND]` - Individual band positioning

## How to Verify

1. **Run the app and observe debug output**
2. **Look for detection boxes appearing on waterfall**
3. **Verify boxes track with waterfall scroll (move upward)**
4. **Check PSD bands align with waterfall boxes horizontally**
5. **Compare PSD bands to peaks in the spectrum graph**

## Expected Log Output Example

```
[DET RAW] box_id=0 x1=0.234 y1=0.456 x2=0.312 y2=0.589 class=creamy_chicken conf=0.87 pts=12.345
[DETECTION DEBUG] Frame 100 - Sending 1 detections:
  base_row=3800, rows_in_frame=38, total_rows_written=3838
  [0] {"detection_id": 0, "x1": 0.234, "y1": 0.456, ...}
      -> absoluteRow=3808 (base_row + row_offset = 3800 + 8)

[VideoStream] DETECTION RECEIVED: pts=12.345, baseRow=3800, rowsInFrame=38, count=1
[VideoStream]   First raw: x1=0.234, y1=0.456, x2=0.312, y2=0.589
[VideoStream]   row_offset=8, row_span=3
[VideoStream]   Parsed: y1=0.456, y2=0.589, absoluteRow=3808, rowSpan=3

[WATERFALL] plotWidth=950, plotHeight=400, pixelsPerRow=0.070
[WATERFALL] totalRowsReceived=3850, bufferHeight=5700, detections=1
[WATERFALL BOX] det_id=0 class=creamy_chicken
  y1=0.456 y2=0.589 → left=407.0 width=147.9
  absoluteRow=3808 rowSpan=3 rowsAgo=42
  boxTop=397.1 boxHeight=2.9

[PSD] plotWidth=950, plotHeight=120, detections=1
[PSD BAND] class=creamy_chicken
  y1=0.456 y2=0.589 → left=407.0 width=147.9
```

## Success Criteria

- ✅ Boxes appear on the waterfall where signals are visible
- ✅ Boxes scroll up with the waterfall as new data arrives
- ✅ Boxes are horizontally centered on their signals
- ✅ PSD frequency bands align with waterfall boxes
- ✅ PSD bands align with peaks in the spectrum
- ✅ Boxes are ~20% larger than the actual detection
- ✅ No overflow errors or visual glitches
