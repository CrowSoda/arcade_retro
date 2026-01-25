# Waterfall Source Selection

## Overview

The waterfall display shows a viewport into the most important RF stream. The system intelligently selects which receiver (RX) stream to display based on recording status:

```
if (any RX is saving IQ to disk):
    show that stream
else:
    show RX1 scanning
```

**User benefit:** You always know "Am I looking at saved data or just hunting?"

## Source Priority

Sources are selected by priority (highest to lowest):

| Priority | Source | Indicator | Color |
|----------|--------|-----------|-------|
| 1 | Manual collection | `MANUAL` | Red (with dot) |
| 2 | RX2 collecting detection | `RX2 REC` | Red (with dot) |
| 3 | RX1 collecting detection | `RX1 REC` | Red (with dot) |
| 4 | RX1 scanning (default) | `SCANNING` | Grey |

## System States

| System State | Waterfall Shows | Indicator |
|-------------|-----------------|-----------|
| RX1 scanning, no detections | RX1 stream | "SCANNING" (grey) |
| RX1 detected, RX2 collecting | RX2 stream | "RX2 REC" (red dot) |
| RX1 detected, RX1 collecting (single RX mode) | RX1 stream | "RX1 REC" (red dot) |
| Manual collection on any RX | That RX's stream | "MANUAL" (red dot) |

## Architecture

```
┌─────────────┐     ┌─────────────┐
│    RX1      │     │    RX2      │
│  (scanner)  │     │ (collector) │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────┐
│       Stream Source Selector      │
│                                  │
│  if (manual_active):             │
│      output = manual_rx.stream   │
│  elif (rx2.isRecording):         │
│      output = rx2.stream         │
│  elif (rx1.isRecording):         │
│      output = rx1.stream         │
│  else:                           │
│      output = rx1.stream (scan)  │
└──────────────────┬───────────────┘
                   │
                   ▼
            ┌──────────────┐
            │  Waterfall   │
            │   Display    │
            └──────────────┘
```

## Implementation Details

### Backend (unified_pipeline.py)

#### WaterfallSource Enum
```python
class WaterfallSource:
    """Which RX stream is feeding the waterfall display."""
    RX1_SCANNING = 0   # Default: RX1 is scanning, no detection/recording
    RX1_RECORDING = 1  # RX1 detected something and is recording
    RX2_RECORDING = 2  # RX2 is collecting (RX1 handed off detection)
    MANUAL = 3         # Manual collection on any RX
```

#### StreamSourceSelector Class
```python
class StreamSourceSelector:
    """
    Determines which IQ stream should feed the waterfall display.
    
    Priority (highest to lowest):
    1. Manual collection (user explicitly recording)
    2. RX2 recording (detector handed off to collector)
    3. RX1 recording (single RX mode, detector is also collecting)
    4. RX1 scanning (default: just hunting for signals)
    """
    
    def update(self, rx1_recording=None, rx2_recording=None, 
               manual_active=None, manual_rx=None) -> int:
        """Update source state and return which source should feed the waterfall."""
        # ... priority-based selection logic
```

#### Strip Message Header (17 bytes)
```python
header = struct.pack('<I I H H f B',
    frame_count,           # uint32 - Frame ID
    total_rows_written,    # uint32 - Monotonic row counter
    rows_this_frame,       # uint16 - Rows in this strip
    video_width,           # uint16 - Strip width (2048)
    current_pts,           # float32 - Presentation timestamp
    source_selector.current_source,  # uint8 - Source ID (0-3) **NEW**
)
```

### Flutter (video_stream_provider.dart)

#### WaterfallSource Enum
```dart
enum WaterfallSource {
  rx1Scanning,   // 0 - RX1 scanning, no detection/recording
  rx1Recording,  // 1 - RX1 detected something and is recording
  rx2Recording,  // 2 - RX2 is collecting (handoff from RX1)
  manual,        // 3 - Manual collection on any RX
}
```

#### Source Label Extension
```dart
extension WaterfallSourceExtension on WaterfallSource {
  String get label {
    switch (this) {
      case WaterfallSource.rx1Scanning: return 'SCANNING';
      case WaterfallSource.rx1Recording: return 'RX1 REC';
      case WaterfallSource.rx2Recording: return 'RX2 REC';
      case WaterfallSource.manual: return 'MANUAL';
    }
  }
  
  bool get isRecording => this != WaterfallSource.rx1Scanning;
}
```

#### Buffer Clear on Source Change
When the source changes, the pixel buffer is cleared to prevent mixing old/new data:
```dart
if (newSource != state.waterfallSource && _pixelBuffer != null) {
  debugPrint('[VideoStream] Source changed: ${state.waterfallSource.label} -> ${newSource.label}, clearing buffer');
  _clearPixelBuffer();  // Fill with viridis dark purple
  _detectionBuffer.clear();
}
```

### Flutter (video_waterfall_display.dart)

#### Stats Overlay Display
The stats overlay shows the current source with visual indicators:
- **Scanning state:** Grey "SCANNING" text
- **Recording states:** Red dot + red text showing source (e.g., "RX1 REC", "RX2 REC", "MANUAL")

## Usage

### For Detection System Integration
When a detection starts recording:
```python
# In your detection handler:
async def handle_detection(self, detection):
    collecting_rx = self.assign_collector(detection)
    
    if collecting_rx == 2:
        self.source_selector.update(rx1_recording=False, rx2_recording=True)
        await self.start_recording(rx=2, detection=detection)
    else:
        self.source_selector.update(rx1_recording=True, rx2_recording=False)
        await self.start_recording(rx=1, detection=detection)

async def handle_recording_complete(self):
    self.source_selector.update(rx1_recording=False, rx2_recording=False)
    # Waterfall automatically switches back to scanning
```

### For Manual Collection
```python
# Start manual collection
self.source_selector.update(manual_active=True, manual_rx=1)

# Stop manual collection
self.source_selector.update(manual_active=False)
```

## Files Modified

| File | Changes |
|------|---------|
| `backend/unified_pipeline.py` | Added `WaterfallSource` enum, `StreamSourceSelector` class, updated header to 17 bytes |
| `lib/features/live_detection/providers/video_stream_provider.dart` | Added `WaterfallSource` enum, parse source from header, clear buffer on switch |
| `lib/features/live_detection/widgets/video_waterfall_display.dart` | Updated stats overlay with SCANNING/RECORDING indicator |

## Validation Checklist

| Check | Expected |
|-------|----------|
| No detections | Waterfall shows scanning, "SCANNING" label (grey) |
| Detection starts, RX2 collecting | Waterfall switches to RX2, "RX2 REC" label (red dot) |
| Detection ends | Waterfall switches back to RX1 scanning |
| Source switch | Buffer clears, no mixed old/new data |
| Manual collection | Waterfall shows manual target, "MANUAL" label |

## Future: Dual IQ Source

When you have two physical RX streams:
```python
class DualRXPipeline:
    def __init__(self):
        self.rx1_source = UnifiedIQSource(...)  # Scanner
        self.rx2_source = UnifiedIQSource(...)  # Collector
        self.active_source = self.rx1_source
        self.source_selector = StreamSourceSelector()
    
    def get_next_chunk(self):
        # Always return from the currently selected source
        return self.active_source.read_chunk()
    
    def switch_to_rx2(self):
        self.active_source = self.rx2_source
        # Buffer will clear on Flutter side when it sees source change
```
