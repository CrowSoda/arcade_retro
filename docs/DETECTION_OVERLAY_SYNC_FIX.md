# Detection Overlay Synchronization Fix

## Problem Statement

Detection boxes painted over the waterfall display do not track their position accurately as the waterfall scrolls. If a detection is at position (row 1, col 1), then when we update with a new row, it should move to (row 2, col 1).

**Root Cause:** PTS-based positioning fails because:
1. Backend sends one PTS per detection batch, but each batch corresponds to **~38 rows** of waterfall data
2. A single timestamp cannot encode which specific row contains the signal
3. Playback speed variations cause drift between PTS time and display time

---

## The Fix in One Sentence

**Replace PTS-based positioning with row indices** — each detection knows "I belong to row N", and the display calculates position from `(current_row - detection_row)`.

---

## Current Architecture Analysis

### Waterfall Scroll Mechanism

The waterfall scrolls **per-buffer-fill**, rate-limited to ~30fps.

From `unified_pipeline.py`:
```python
async def run_pipeline(self, websocket):
    frame_interval = 1.0 / self.video_fps  # 1/30 = 0.033s
    
    while self.is_running:
        frame_start = time.perf_counter()
        
        # Read IQ chunk (33ms of data at 20MHz = ~660k samples)
        chunk = self.iq_source.read_chunk(duration_ms=33)
        self.current_pts = chunk.pts  # PTS from FILE POSITION, not wall clock
        
        # Compute waterfall rows - THIS CREATES ~38 ROWS PER CHUNK!
        db_rows = self.pipeline.compute_waterfall_rows(chunk.data)
        for db_row in db_rows:
            self.waterfall_buffer.add_row(db_row)  # SCROLL HAPPENS HERE
        
        # Render frame and send
        rgb_frame = self.waterfall_buffer.get_frame_with_detections(...)
        video_bytes = self.encoder.encode(rgb_frame)
        await websocket.send(bytes([self.MSG_VIDEO]) + video_bytes)
        
        # Rate limit to 30fps
        elapsed = time.perf_counter() - frame_start
        sleep_time = max(0.001, frame_interval - elapsed)
        await asyncio.sleep(sleep_time)
```

### Multiple Rows Per Frame Problem

With `waterfall_fft_size = 32768` and 50% overlap:
- One 33ms chunk at 20MHz = 660,000 samples
- Number of FFTs = `(660000 - 32768) / 16384 + 1 ≈ 38 rows` per frame

So **each JPEG frame contains ~38 new rows**, but detections only get **one PTS per batch**.

### Current Detection Positioning (Flutter)

From `video_waterfall_display.dart`:
```dart
// PROBLEM: Uses PTS age for positioning
final detectionAge = currentPts - det.pts;  
final normalizedAge = detectionAge / timeSpan;
final boxBottom = (1.0 - normalizedAge) * plotHeight;
```

This fails because PTS represents "when the batch was processed" rather than "which row of the waterfall contains the signal".

---

## Solution: Row-Index Synchronization

### Pattern Used by Professional SDR Software

- **GQRX:** Age-based row indexing with `getLine(unsigned int age)`
- **Inspectrum:** Sample indices as primary reference (SigMF annotations)
- **GNU Radio, CubicSDR:** Row/line indices relative to current position

The universal pattern: **row indices relative to current position, not absolute timestamps**.

### Core Formula

```dart
detection_row = frame_id * rows_per_frame + row_offset_within_frame
display_y = (current_bottom_row - detection.start_row) * row_height
```

When `display_y` exceeds waterfall height, the detection has scrolled off and should be culled.

---

## Implementation Roadmap

### Phase 1: Backend Changes (Python)

**File:** `g20_demo/backend/unified_pipeline.py`

#### 1. Add Global Row Counter

```python
class VideoStreamServer:
    def __init__(self, ...):
        # ... existing code ...
        self.total_rows_written = 0  # Monotonic counter, never resets
```

#### 2. Track Rows Per Frame

```python
async def run_pipeline(self, websocket):
    while self.is_running:
        # ... existing chunk read code ...
        
        db_rows = self.pipeline.compute_waterfall_rows(chunk.data)
        rows_this_frame = len(db_rows)
        for db_row in db_rows:
            self.waterfall_buffer.add_row(db_row)
        self.total_rows_written += rows_this_frame
        
        # ... rest of loop ...
```

#### 3. Send Row Info with Detections

```python
async def _run_inference_async(self, websocket, iq_data, pts, frame_id):
    result = self.pipeline.process_chunk(iq_data, pts)
    
    det_list = [{
        'detection_id': d.box_id,
        'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2,
        'confidence': d.confidence,
        'class_name': d.class_name,
        'row_offset': int(d.x1 * rows_this_frame),  # NEW: 0-37 within this frame
    } for d in result['detections']]
    
    msg = json.dumps({
        'type': 'detection_frame',
        'frame_id': frame_id,
        'base_row': self.total_rows_written,    # NEW: absolute row position
        'rows_in_frame': rows_this_frame,       # NEW: typically 38
        'pts': result['pts'],                   # Keep for backward compat
        'detections': det_list,
    })
    
    await websocket.send(bytes([self.MSG_DETECTION]) + msg.encode())
```

#### 4. Modify Inference to Output row_offset

The inference sees a spectrogram chunk where detection's `x1/x2` (time axis) maps to `row_offset`:
```python
row_offset = int(det.x1 * rows_this_frame)
```

---

### Phase 2: Frontend Changes (Flutter)

**File:** `g20_demo/lib/features/live_detection/providers/video_stream_provider.dart`

#### 1. Update State Model

```dart
class VideoStreamState {
  final bool isConnected;
  final bool isConnecting;
  final StreamMetadata? metadata;
  final Uint8List? currentFrame;
  final List<VideoDetection> detections;
  final int totalRowsReceived;  // NEW: replaces currentPts for positioning
  final int rowsPerFrame;       // NEW: typically 38
  final double currentPts;      // Keep for backward compat
  final int frameCount;
  final String? error;
  final double fps;
  
  // ... constructor and copyWith ...
}
```

#### 2. Update Detection Model

```dart
class VideoDetection {
  final int detectionId;
  final double x1, y1, x2, y2;  // Keep for frequency axis (y1/y2)
  final double confidence;
  final int classId;
  final String className;
  final int absoluteRow;  // NEW: base_row + row_offset
  final int rowSpan;      // NEW: number of rows detection spans
  final bool isSelected;
  
  // ... constructor ...
  
  factory VideoDetection.fromJson(Map<String, dynamic> json, int baseRow, int rowsInFrame) {
    final rowOffset = json['row_offset'] ?? 0;
    return VideoDetection(
      detectionId: json['detection_id'] ?? 0,
      x1: (json['x1'] ?? 0).toDouble(),
      y1: (json['y1'] ?? 0).toDouble(),
      x2: (json['x2'] ?? 0).toDouble(),
      y2: (json['y2'] ?? 0).toDouble(),
      confidence: (json['confidence'] ?? 0).toDouble(),
      classId: json['class_id'] ?? 0,
      className: json['class_name'] ?? 'unknown',
      absoluteRow: baseRow - rowsInFrame + rowOffset,  // Exact row
      rowSpan: max(1, ((json['x2'] ?? 0) - (json['x1'] ?? 0)) * rowsInFrame).round(),
    );
  }
}
```

#### 3. Simplify Detection Handling

```dart
void _handleDetection(Uint8List jsonData) {
  try {
    final jsonStr = utf8.decode(jsonData);
    final data = json.decode(jsonStr) as Map<String, dynamic>;

    final baseRow = data['base_row'] as int? ?? state.totalRowsReceived;
    final rowsInFrame = data['rows_in_frame'] as int? ?? 38;
    final detList = (data['detections'] as List<dynamic>?) ?? [];

    // Parse new detections with absolute row indices
    final newDetections = detList
        .map((d) => VideoDetection.fromJson(d as Map<String, dynamic>, baseRow, rowsInFrame))
        .toList();

    // Add new detections to buffer
    _detectionBuffer.addAll(newDetections);
    
    // Cull old detections (simple integer comparison)
    final visibleRows = state.totalRowsReceived;
    final displayRows = (_currentTimeSpan * 30 * rowsInFrame).round();
    _detectionBuffer.removeWhere((d) => d.absoluteRow < visibleRows - displayRows);

    state = state.copyWith(
      detections: List.from(_detectionBuffer),
      rowsPerFrame: rowsInFrame,
    );
  } catch (e) {
    debugPrint('[VideoStream] Detection parse error: $e');
  }
}

void _handleVideoFrame(Uint8List frameData) {
  // ... existing FPS calculation ...
  
  // Increment row counter
  final rowsInFrame = state.rowsPerFrame > 0 ? state.rowsPerFrame : 38;
  
  state = state.copyWith(
    currentFrame: frameData,
    totalRowsReceived: state.totalRowsReceived + rowsInFrame,
    frameCount: state.frameCount + 1,
    fps: _measuredFps,
  );
}
```

---

**File:** `g20_demo/lib/features/live_detection/widgets/video_waterfall_display.dart`

#### 4. Rewrite Overlay Positioning

```dart
class _DetectionOverlayLayer extends ConsumerWidget {
  final int totalRowsReceived;
  final int rowsPerFrame;
  final List<VideoDetection> detections;

  const _DetectionOverlayLayer({
    required this.totalRowsReceived,
    required this.rowsPerFrame,
    required this.detections,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (detections.isEmpty) {
      return const SizedBox.shrink();
    }

    // Get the time span to calculate visible rows
    final timeSpan = ref.watch(waterfallTimeSpanProvider);
    final displayRows = (timeSpan * 30 * rowsPerFrame).round();

    return LayoutBuilder(
      builder: (context, constraints) {
        final plotWidth = constraints.maxWidth;
        final plotHeight = constraints.maxHeight;
        final rowHeight = plotHeight / displayRows;
        
        final visibleBoxes = <Widget>[];
        
        for (final det in detections) {
          // Check visibility toggle
          final isVisible = ref.watch(soiVisibilityProvider(det.className));
          if (!isVisible) continue;
          
          // === ROW-BASED TIME AXIS (Y) ===
          // How many rows ago was this detection?
          final rowsAgo = totalRowsReceived - det.absoluteRow;
          
          // Skip if outside visible range
          if (rowsAgo < 0 || rowsAgo >= displayRows) continue;
          
          // Y position: rowsAgo=0 → bottom, rowsAgo=displayRows → top
          final boxBottom = plotHeight - rowsAgo * rowHeight;
          final boxTop = boxBottom - (det.rowSpan * rowHeight).clamp(8.0, plotHeight * 0.3);
          
          // Skip if outside visible area
          if (boxTop > plotHeight || boxBottom < 0) continue;
          
          // === FREQUENCY AXIS (X) - unchanged ===
          final left = det.y1 * plotWidth;
          final right = det.y2 * plotWidth;
          final boxWidth = (right - left).abs();
          
          if (boxWidth < 4) continue;
          
          final color = getSOIColor(det.className);
          
          visibleBoxes.add(
            Positioned(
              left: left.clamp(0.0, plotWidth - boxWidth),
              top: boxTop.clamp(0.0, plotHeight - 8),
              width: boxWidth.clamp(4.0, plotWidth),
              height: (boxBottom - boxTop).clamp(8.0, plotHeight),
              child: _DetectionBoxWidget(detection: det, color: color),
            ),
          );
        }
        
        return Stack(
          clipBehavior: Clip.hardEdge,
          children: visibleBoxes,
        );
      },
    );
  }
}
```

#### 5. Update Widget Call

```dart
// In VideoWaterfallDisplay.build():
child: _DetectionOverlayLayer(
  totalRowsReceived: streamState.totalRowsReceived,
  rowsPerFrame: streamState.rowsPerFrame,
  detections: streamState.detections,
),
```

---

### Phase 3: Fix PSD Memory Leak

The issue: PSD boxes accumulate instead of replacing.

**Option A:** Clear previous PSD detections before adding new ones:
```dart
void _handlePsdDetection(data) {
  // Remove ALL previous PSD detections
  _detectionBuffer.removeWhere((d) => d.source == 'psd');
  
  // Add new ones
  _detectionBuffer.addAll(newPsdDetections);
}
```

**Option B:** If PSD should only show current frame, don't add to buffer:
```dart
// Don't add to buffer at all - render separately
state = state.copyWith(currentPsdBoxes: newPsdDetections);
```

---

## Summary

| What | Before | After |
|------|--------|-------|
| Position reference | PTS (seconds) | Row index (integer) |
| Sync dependency | Wall clock timing | Frame count only |
| Playback speed | Causes drift | No effect |
| Culling logic | PTS comparison | Simple integer math |
| PSD boxes | Accumulate forever | Replace each frame |

**Total changes:** ~50 lines backend, ~80 lines frontend. No architectural changes needed.

---

## Files to Modify

### Backend
- `g20_demo/backend/unified_pipeline.py` - Add row counter, send row info with detections

### Frontend
- `g20_demo/lib/features/live_detection/providers/video_stream_provider.dart` - Update state model, detection parsing
- `g20_demo/lib/features/live_detection/widgets/video_waterfall_display.dart` - Rewrite overlay positioning

---

## References

- GQRX FftBuffer implementation: age-based row indexing
- Inspectrum: sample indices as primary reference (SigMF v0.3.0)
- GNU Radio / CubicSDR: row/line indices relative to current position
