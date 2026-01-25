# Fact-Finding Investigation for Junior Dev

## 1. Current Video Encoding

### What encoder is being used?
**Priority order (checked in `create_encoder()`):**
1. NVENC H.264 (if GPU available) - `h264_nvenc`
2. libx264 software H.264 (if FFmpeg available)
3. JPEG fallback (always works)

The encoder selection is logged at startup:
```
[create_encoder] NVENC available, using H.264 hardware encoding
# or
[create_encoder] Using libx264 software encoding
# or
[create_encoder] Falling back to JPEG encoding
```

### Quality/compression settings:

**H.264 (NVENC/libx264):**
```python
'-preset', 'p1' / 'ultrafast'   # Lowest latency
'-tune', 'll' / 'zerolatency'   # Low latency tuning
'-b:v', '4000000'               # 4 Mbps bitrate
'-g', '30'                      # Keyframe every 30 frames (1 sec)
'-bf', '0'                      # No B-frames
```

**JPEG:**
```python
quality = 80  # Pillow quality setting
```

### Frame byte size: **NOT CURRENTLY LOGGED**
**⚠️ ACTION REQUIRED:** Add logging to `run_pipeline()`:
```python
if video_bytes:
    print(f"[FRAME] {len(video_bytes)} bytes")
```

### Encoding time per frame: **NOT CURRENTLY LOGGED**
**⚠️ ACTION REQUIRED:** Add timing measurement to `run_pipeline()`.

---

## 2. Current Frame Dimensions

### video_width:
```python
video_width = 2048  # Hardcoded in VideoStreamServer.__init__
```

### video_height:
```python
video_height = int(time_span_seconds * video_fps)
# Default: 5.0 * 30 = 150 pixels tall
```

### Does height change when time_span changes?
**YES** - The `set_time_span` command handler:
1. Recreates `WaterfallBuffer` with new height
2. Closes old encoder
3. Creates new encoder with new dimensions
4. Sends updated metadata to client

```python
new_height = max(30, min(900, int(seconds * 30)))  # 1s-30s range
```

### Flutter display size:
The widget uses `LayoutBuilder` and displays at whatever size the parent allows:
```dart
Positioned(
  left: plotRect.left,    // 50.0 (left margin)
  top: plotRect.top,      // 8.0 (top margin)
  width: plotRect.width,  // size.width - 58 (minus margins)
  height: plotRect.height, // size.height - 33 (minus margins)
  child: Image.memory(frame, fit: BoxFit.fill)
)
```
**Actual screen pixels depend on widget layout** - not fixed.

---

## 3. WebSocket Message Sizes

### Message format:
| Type | Prefix | Content |
|------|--------|---------|
| Video frame | `0x01` | JPEG or H.264 NAL bytes |
| Detection | `0x02` | JSON with detection list |
| Metadata | `0x03` | JSON with stream params |

### **NOT CURRENTLY LOGGED**
**⚠️ ACTION REQUIRED:** Add to `run_pipeline()`:
```python
# After send
print(f"[MSG_VIDEO] {len(video_bytes)} bytes")

# In _run_inference_async:
print(f"[MSG_DETECTION] {len(msg.encode())} bytes")
```

### Bandwidth estimate (theoretical):
- Video: ~150 rows × (JPEG ~30KB per 2048×1 row) × 30fps = **~135 MB/s** (way too high!)
- But actually: Full frame is 2048×150 pixels → JPEG ~30-80KB per frame
- At 30fps: **~1-2.5 MB/s** for video alone

---

## 4. Flutter Rendering

### How is waterfall rendered?
**Video mode (current):**
```dart
Image.memory(
  frame,                          // Uint8List JPEG bytes
  fit: BoxFit.fill,               // Stretch to fill
  filterQuality: FilterQuality.high,
  gaplessPlayback: true,          // No flicker between frames
)
```

**Row-by-row mode (exists but not used for video):**
```dart
// In WaterfallNotifier._onWaterfallRow():
_pixelBuffer.setRange(0, _totalBytes - _rowBytes, _pixelBuffer.sublist(_rowBytes));  // Scroll
_pixelBuffer.setRange(bottomRowStart, _totalBytes, row.rgbaPixels);  // Insert new row
```

### Does Flutter maintain local pixel buffer?
**Video mode: NO** - Just displays JPEG bytes from WebSocket.

**Row-by-row mode: YES** - `WaterfallNotifier` maintains:
```dart
late Uint8List _pixelBuffer;  // Size: 2048 × height × 4 bytes (RGBA)
```

### Widget size on screen:
**Depends on parent layout.** The waterfall is inside a `LayoutBuilder` and fills available space minus margins:
- Left margin: 50px (time axis)
- Top margin: 8px
- Right margin: 8px
- Bottom margin: 25px (frequency axis)

---

## 5. Network Constraints

**NOT DOCUMENTED IN CODE** - Questions for user:
- [ ] WiFi hardware: ESP32? Raspberry Pi? Phone hotspot?
- [ ] Target device: Phone model? Tablet?
- [ ] Latency requirements: ms from signal-in-air to box-on-screen?

---

## 6. Compute Hardware

**NOT DOCUMENTED IN CODE** - Questions for user:
- [ ] Backend device: Pi 4? Pi 5? Jetson? Laptop?
- [ ] CPU usage during streaming: (run `htop`)
- [ ] GPU availability for encoding: (NVENC check logs at startup)

---

## 7. Quick Test - Diagnostic Logging

**Add this to `run_pipeline()` in `unified_pipeline.py`:**

```python
# At the top of run_pipeline():
import time
log_interval = 30  # Log every 30 frames (1 second)

# Inside the main loop, after video_bytes = self.encoder.encode(rgb_frame):
if frame_count % log_interval == 0:
    print(f"[STATS] Frame {frame_count}: "
          f"Video={len(video_bytes) if video_bytes else 0} bytes, "
          f"Rows/frame={self.rows_this_frame}, "
          f"Buffer={self.waterfall_buffer.height} rows, "
          f"Dims={self.video_width}×{self.video_height}", flush=True)
```

---

## 8. Existing Row-Streaming Code

### Is there existing row-by-row code?
**YES** - Two separate systems exist:

### A. Backend: `UnifiedServer` (legacy row-by-row)
```python
# In unified_pipeline.py - legacy handler
class UnifiedServer:
    """Row-by-row streaming via binary WebSocket"""
```

### B. Flutter: `WaterfallNotifier` with `_pixelBuffer`
Location: `lib/features/live_detection/providers/waterfall_provider.dart`

```dart
class WaterfallNotifier extends StateNotifier<WaterfallState> {
  late Uint8List _pixelBuffer;  // RGBA buffer, scrolled in-place
  
  void _onWaterfallRow(WaterfallRow row) {
    // SCROLL: Move all rows UP by 1 row (memmove)
    _pixelBuffer.setRange(0, _totalBytes - _rowBytes, 
        _pixelBuffer.sublist(_rowBytes));
    
    // INSERT: Copy new row to bottom
    _pixelBuffer.setRange(bottomRowStart, _totalBytes, row.rgbaPixels);
  }
}
```

### What format are rows sent in?
**Binary protocol:**
```
Byte 0: Message type (0x01)
Bytes 1-4: Sequence ID (uint32 LE)
Bytes 5-12: PTS (float64 LE)
Bytes 13-16: Width (uint32 LE)
Bytes 17 - (17 + width*4): RGBA pixels (width × 4 bytes)
Remaining: Float32 dB values for PSD (width × 4 bytes)
```

### Is WaterfallNotifier currently used?
**YES, but for a different purpose!** 

It's connected to `UnifiedPipelineManager` (row-by-row mode), NOT to the video stream:
```dart
void connectToPipeline(UnifiedPipelineManager pipeline) {
  _waterfallSub = pipeline.waterfallRows.listen(_onWaterfallRow);
}
```

The video mode uses `VideoWaterfallDisplay` + `videoStreamProvider` which is completely separate.

---

## Summary: Two Parallel Systems

| System | Backend | Flutter | Currently Used? |
|--------|---------|---------|-----------------|
| **Row-by-row** | `UnifiedServer` | `WaterfallNotifier` + `_pixelBuffer` | Unclear |
| **Video stream** | `VideoStreamServer` | `VideoWaterfallDisplay` + `Image.memory` | YES |

The video stream approach sends complete JPEG frames.
The row-by-row approach sends individual RGBA rows.

**Both exist, both have code, but they're different paths.**
