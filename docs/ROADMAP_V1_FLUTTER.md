# G20/NV100 Flutter/UI Implementation Guide
## ROADMAP_V1_FLUTTER.md

---

## Overview

This document contains detailed implementation code and guidance for the Flutter UI components of the G20 RF Waterfall Detection System.

**Related Documents:**
- [ROADMAP_V1_OVERVIEW.md](ROADMAP_V1_OVERVIEW.md) - System architecture and overview
- [ROADMAP_V1_BACKEND.md](ROADMAP_V1_BACKEND.md) - Backend implementation
- [ROADMAP_V1_DEPLOYMENT.md](ROADMAP_V1_DEPLOYMENT.md) - Deployment guide

---

## Phase 3: Frame-ID Synchronization

**Goal:** Replace time-based scrolling with frame-indexed mapping.

### Task 3.1: Frame Buffer Manager (Python Backend)

```python
# frame_buffer.py
from collections import OrderedDict
from threading import Lock

class FrameBuffer:
    """
    Manages spectrogram frames for synchronized display.
    Frames are indexed by frame_id, not time.
    """
    
    def __init__(self, max_frames: int = 300):
        self.max_frames = max_frames
        self.frames: OrderedDict[int, SpectrogramFrame] = OrderedDict()
        self.lock = Lock()
    
    def add_frame(self, frame: SpectrogramFrame):
        with self.lock:
            self.frames[frame.frame_id] = frame
            
            # Prune old frames
            while len(self.frames) > self.max_frames:
                self.frames.popitem(last=False)
    
    def get_frame(self, frame_id: int) -> Optional[SpectrogramFrame]:
        with self.lock:
            return self.frames.get(frame_id)
    
    def get_display_range(self, rows: int = 256) -> list[SpectrogramFrame]:
        """Get the most recent N frames for display."""
        with self.lock:
            frame_ids = list(self.frames.keys())[-rows:]
            return [self.frames[fid] for fid in frame_ids]
    
    def frame_id_to_row(self, frame_id: int, display_rows: int = 256) -> Optional[int]:
        """
        Convert frame_id to display row using buffer position, not ID arithmetic.
        Handles dropped frames correctly.
        
        CRITICAL: Row position is determined by position in the visible deque,
        NOT by subtracting frame IDs. Using arithmetic difference (newest_id - frame_id)
        breaks when frames are dropped, causing boxes to jump.
        
        Returns None if frame is outside display range.
        """
        with self.lock:
            frame_ids = list(self.frames.keys())
            
            if frame_id not in frame_ids:
                return None
            
            # Get visible range (most recent N frames)
            visible_ids = frame_ids[-display_rows:]
            
            if frame_id not in visible_ids:
                return None
            
            # Row is position in visible buffer, not ID difference
            # Row 0 = newest (top of waterfall)
            position_from_end = visible_ids[::-1].index(frame_id)
            return position_from_end
```

### Task 3.2: Update Flutter Display

```dart
// waterfall_display.dart
class WaterfallDisplayState extends State<WaterfallDisplay> {
  final FrameBuffer _frameBuffer = FrameBuffer();
  
  /// Convert track box to pixel coordinates using frame_id mapping
  Rect trackBoxToPixels(Track track, Size displaySize, int displayRows) {
    // X: frequency axis (direct mapping)
    final x1 = track.displayBox[0] * displaySize.width;
    final x2 = track.displayBox[2] * displaySize.width;
    
    // Y: time axis (frame_id based)
    final row = _frameBuffer.frameIdToRow(
      track.lastSeenFrameId, 
      displayRows
    );
    
    if (row == null) {
      // Track is outside visible range
      return Rect.zero;
    }
    
    final rowHeight = displaySize.height / displayRows;
    final y1 = row * rowHeight;
    final y2 = y1 + (track.displayBox[3] - track.displayBox[1]) * displaySize.height;
    
    return Rect.fromLTRB(x1, y1.toDouble(), x2, y2.clamp(0, displaySize.height));
  }
}
```

### Task 3.3: Dart Frame Buffer Implementation

```dart
// lib/models/frame_buffer.dart
import 'dart:collection';

class FrameBuffer {
  final int maxFrames;
  final LinkedHashMap<int, SpectrogramFrame> _frames = LinkedHashMap();
  
  FrameBuffer({this.maxFrames = 300});
  
  void addFrame(SpectrogramFrame frame) {
    _frames[frame.frameId] = frame;
    
    // Prune old frames
    while (_frames.length > maxFrames) {
      _frames.remove(_frames.keys.first);
    }
  }
  
  SpectrogramFrame? getFrame(int frameId) => _frames[frameId];
  
  List<SpectrogramFrame> getDisplayRange(int rows) {
    final frameIds = _frames.keys.toList();
    final startIdx = (frameIds.length - rows).clamp(0, frameIds.length);
    return frameIds
        .sublist(startIdx)
        .map((id) => _frames[id]!)
        .toList();
  }
  
  /// Convert frame_id to display row using buffer position, not ID arithmetic.
  /// Handles dropped frames correctly.
  ///
  /// CRITICAL: Row position is determined by position in the visible list,
  /// NOT by subtracting frame IDs. Using arithmetic difference (newestId - frameId)
  /// breaks when frames are dropped, causing boxes to jump.
  int? frameIdToRow(int frameId, int displayRows) {
    if (!_frames.containsKey(frameId)) return null;
    
    final frameIds = _frames.keys.toList();
    final visibleIds = frameIds.sublist(
      (frameIds.length - displayRows).clamp(0, frameIds.length)
    );
    
    if (!visibleIds.contains(frameId)) return null;
    
    // Row is position in visible buffer, not ID difference
    // Row 0 = newest (top of waterfall)
    final reversedIds = visibleIds.reversed.toList();
    final positionFromEnd = reversedIds.indexOf(frameId);
    
    return positionFromEnd >= 0 ? positionFromEnd : null;
  }
  
  int get newestFrameId => _frames.keys.isEmpty ? 0 : _frames.keys.last;
  int get oldestFrameId => _frames.keys.isEmpty ? 0 : _frames.keys.first;
  int get length => _frames.length;
}
```

### Task 3.4: Track Model

```dart
// lib/models/track.dart
class Track {
  final int trackId;
  final int classId;
  final String className;
  final List<double> truthBox;     // [x1, y1, x2, y2] normalized 0-1
  final List<double> displayBox;   // Smoothed for rendering
  final double confidence;
  final String state;              // 'tentative' | 'confirmed' | 'lost'
  final int ageFrames;
  final int hits;
  final String motionMode;         // 'stationary' | 'drifting'
  final double driftRateHzPerSec;
  final double driftConfidence;
  final double freqCenterHz;
  final double freqBandwidthHz;
  final int firstSeenFrameId;
  final int lastSeenFrameId;
  
  const Track({
    required this.trackId,
    required this.classId,
    required this.className,
    required this.truthBox,
    required this.displayBox,
    required this.confidence,
    required this.state,
    required this.ageFrames,
    required this.hits,
    required this.motionMode,
    required this.driftRateHzPerSec,
    required this.driftConfidence,
    required this.freqCenterHz,
    required this.freqBandwidthHz,
    required this.firstSeenFrameId,
    required this.lastSeenFrameId,
  });
  
  factory Track.fromBinary(ByteData data, int offset) {
    return Track(
      trackId: data.getUint32(offset, Endian.little),
      classId: data.getUint16(offset + 4, Endian.little),
      confidence: data.getFloat32(offset + 6, Endian.little),
      state: _stateFromInt(data.getUint8(offset + 10)),
      displayBox: [
        data.getFloat32(offset + 11, Endian.little),
        data.getFloat32(offset + 15, Endian.little),
        data.getFloat32(offset + 19, Endian.little),
        data.getFloat32(offset + 23, Endian.little),
      ],
      freqCenterHz: data.getFloat64(offset + 27, Endian.little),
      freqBandwidthHz: data.getFloat64(offset + 35, Endian.little),
      motionMode: data.getUint8(offset + 43) == 1 ? 'drifting' : 'stationary',
      // Fill defaults for fields not in binary protocol
      className: '',
      truthBox: const [0, 0, 0, 0],
      ageFrames: 0,
      hits: 0,
      driftRateHzPerSec: 0,
      driftConfidence: 0,
      firstSeenFrameId: 0,
      lastSeenFrameId: 0,
    );
  }
  
  static String _stateFromInt(int value) {
    switch (value) {
      case 0: return 'tentative';
      case 1: return 'confirmed';
      case 2: return 'lost';
      default: return 'unknown';
    }
  }
}
```

**Acceptance criteria:**
- [ ] Track boxes scroll with waterfall at exact same rate
- [ ] No drift between box and signal over time
- [ ] Boxes disappear cleanly when scrolling off screen
- [ ] System can replay from logged frames deterministically

---

## Phase 4: Overlay Rendering

**Goal:** Single-pass rendering with minimal occlusion.

### Task 4.1: Track Overlay Painter

```dart
// track_overlay.dart
import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';

class TrackOverlayPainter extends CustomPainter {
  final List<Track> tracks;
  final int? selectedTrackId;
  final FrameBuffer frameBuffer;
  final int displayRows;
  
  // Pre-allocated paint objects (avoid per-frame allocation)
  final Paint _strokePaint = Paint()
    ..style = PaintingStyle.stroke
    ..strokeCap = StrokeCap.round;
  
  final Paint _fillPaint = Paint()
    ..style = PaintingStyle.fill;
  
  // Class colors
  static const Map<int, Color> classColors = {
    0: Color(0xFF4CAF50),  // Green
    1: Color(0xFF2196F3),  // Blue
    2: Color(0xFFFF9800),  // Orange
    3: Color(0xFF9C27B0),  // Purple
    4: Color(0xFF00BCD4),  // Cyan
  };
  
  TrackOverlayPainter({
    required this.tracks,
    required this.frameBuffer,
    this.selectedTrackId,
    this.displayRows = 256,
  });
  
  @override
  void paint(Canvas canvas, Size size) {
    for (final track in tracks) {
      final rect = _trackToRect(track, size);
      if (rect == Rect.zero) continue;
      
      _paintTrack(canvas, track, rect);
    }
  }
  
  Rect _trackToRect(Track track, Size size) {
    // X axis: direct normalized mapping
    final x1 = track.displayBox[0] * size.width;
    final x2 = track.displayBox[2] * size.width;
    
    // Y axis: frame_id to row mapping
    final row = frameBuffer.frameIdToRow(track.lastSeenFrameId, displayRows);
    if (row == null) return Rect.zero;
    
    final rowHeight = size.height / displayRows;
    
    // Box height from normalized coords
    final boxHeightNorm = track.displayBox[3] - track.displayBox[1];
    final boxHeightRows = (boxHeightNorm * displayRows).clamp(1.0, displayRows.toDouble());
    
    final y1 = row * rowHeight;
    final y2 = y1 + boxHeightRows * rowHeight;
    
    return Rect.fromLTRB(x1, y1.toDouble(), x2, y2.clamp(0, size.height));
  }
  
  void _paintTrack(Canvas canvas, Track track, Rect rect) {
    final isSelected = track.trackId == selectedTrackId;
    
    // Base color from class
    final baseColor = classColors[track.classId % classColors.length] 
        ?? const Color(0xFFFFEB3B);
    
    // Opacity from confidence and age
    final ageDecay = pow(0.92, track.ageFrames).toDouble();
    final opacity = (track.confidence * ageDecay).clamp(0.4, 1.0);
    
    // Stroke width from state
    double strokeWidth;
    switch (track.state) {
      case 'tentative':
        strokeWidth = 1.5;
        break;
      case 'confirmed':
        strokeWidth = isSelected ? 3.0 : 2.0;
        break;
      default:
        strokeWidth = 1.0;
    }
    
    _strokePaint
      ..color = baseColor.withOpacity(opacity)
      ..strokeWidth = strokeWidth;
    
    if (isSelected) {
      // Selected: full rectangle + subtle fill
      canvas.drawRect(rect, _strokePaint);
      _fillPaint.color = baseColor.withOpacity(0.1);
      canvas.drawRect(rect, _fillPaint);
    } else {
      // Default: corner markers only
      _drawCornerMarkers(canvas, rect, _strokePaint);
    }
    
    // Drift indicator (optional)
    if (track.motionMode == 'drifting') {
      _drawDriftIndicator(canvas, rect, track, baseColor.withOpacity(opacity));
    }
  }
  
  void _drawCornerMarkers(Canvas canvas, Rect rect, Paint paint) {
    final cornerLen = min(rect.width, rect.height) * 0.25;
    final len = max(cornerLen, 6.0);
    
    final path = Path()
      // Top-left
      ..moveTo(rect.left, rect.top + len)
      ..lineTo(rect.left, rect.top)
      ..lineTo(rect.left + len, rect.top)
      // Top-right
      ..moveTo(rect.right - len, rect.top)
      ..lineTo(rect.right, rect.top)
      ..lineTo(rect.right, rect.top + len)
      // Bottom-right
      ..moveTo(rect.right, rect.bottom - len)
      ..lineTo(rect.right, rect.bottom)
      ..lineTo(rect.right - len, rect.bottom)
      // Bottom-left
      ..moveTo(rect.left + len, rect.bottom)
      ..lineTo(rect.left, rect.bottom)
      ..lineTo(rect.left, rect.bottom - len);
    
    canvas.drawPath(path, paint);
  }
  
  void _drawDriftIndicator(Canvas canvas, Rect rect, Track track, Color color) {
    // Small arrow showing drift direction
    final arrowPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    
    final centerX = rect.center.dx;
    final top = rect.top - 8;
    
    // Arrow pointing in drift direction
    final direction = track.driftRateHzPerSec > 0 ? 1.0 : -1.0;
    
    canvas.drawLine(
      Offset(centerX - 4 * direction, top + 4),
      Offset(centerX + 4 * direction, top),
      arrowPaint,
    );
    canvas.drawLine(
      Offset(centerX - 4 * direction, top - 4),
      Offset(centerX + 4 * direction, top),
      arrowPaint,
    );
  }
  
  @override
  bool shouldRepaint(TrackOverlayPainter oldDelegate) {
    // Repaint if tracks changed or selection changed
    return tracks != oldDelegate.tracks ||
           selectedTrackId != oldDelegate.selectedTrackId;
  }
}
```

### Task 4.2: Waterfall Widget with Overlay

```dart
// lib/widgets/waterfall_with_overlay.dart
import 'package:flutter/material.dart';

class WaterfallWithOverlay extends StatefulWidget {
  final Stream<Uint8List> waterfallStream;
  final Stream<List<Track>> trackStream;
  final int displayRows;
  
  const WaterfallWithOverlay({
    super.key,
    required this.waterfallStream,
    required this.trackStream,
    this.displayRows = 256,
  });
  
  @override
  State<WaterfallWithOverlay> createState() => _WaterfallWithOverlayState();
}

class _WaterfallWithOverlayState extends State<WaterfallWithOverlay> {
  final FrameBuffer _frameBuffer = FrameBuffer();
  List<Track> _tracks = [];
  int? _selectedTrackId;
  ui.Image? _waterfallImage;
  
  @override
  void initState() {
    super.initState();
    _subscribeToStreams();
  }
  
  void _subscribeToStreams() {
    widget.waterfallStream.listen(_onWaterfallFrame);
    widget.trackStream.listen(_onTracksUpdate);
  }
  
  void _onWaterfallFrame(Uint8List rgbaData) {
    // Decode and update waterfall image
    // Add frame to buffer for sync
    ui.decodeImageFromPixels(
      rgbaData,
      _waterfallWidth,
      _waterfallHeight,
      ui.PixelFormat.rgba8888,
      (image) {
        if (mounted) {
          setState(() {
            _waterfallImage = image;
          });
        }
      },
    );
  }
  
  void _onTracksUpdate(List<Track> tracks) {
    if (mounted) {
      setState(() {
        _tracks = tracks;
      });
    }
  }
  
  void _onTrackTap(Track track) {
    setState(() {
      _selectedTrackId = _selectedTrackId == track.trackId ? null : track.trackId;
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: _handleTapDown,
      child: CustomPaint(
        painter: WaterfallPainter(image: _waterfallImage),
        foregroundPainter: TrackOverlayPainter(
          tracks: _tracks,
          frameBuffer: _frameBuffer,
          selectedTrackId: _selectedTrackId,
          displayRows: widget.displayRows,
        ),
        size: Size.infinite,
      ),
    );
  }
  
  void _handleTapDown(TapDownDetails details) {
    final size = context.size;
    if (size == null) return;
    
    // Find track under tap
    for (final track in _tracks) {
      final rect = _trackToRect(track, size);
      if (rect.contains(details.localPosition)) {
        _onTrackTap(track);
        return;
      }
    }
    
    // Tap outside any track - deselect
    setState(() {
      _selectedTrackId = null;
    });
  }
  
  Rect _trackToRect(Track track, Size size) {
    final x1 = track.displayBox[0] * size.width;
    final x2 = track.displayBox[2] * size.width;
    
    final row = _frameBuffer.frameIdToRow(track.lastSeenFrameId, widget.displayRows);
    if (row == null) return Rect.zero;
    
    final rowHeight = size.height / widget.displayRows;
    final boxHeightNorm = track.displayBox[3] - track.displayBox[1];
    final boxHeightRows = (boxHeightNorm * widget.displayRows).clamp(1.0, widget.displayRows.toDouble());
    
    final y1 = row * rowHeight;
    final y2 = y1 + boxHeightRows * rowHeight;
    
    return Rect.fromLTRB(x1, y1, x2, y2.clamp(0, size.height));
  }
  
  @override
  void dispose() {
    _waterfallImage?.dispose();
    super.dispose();
  }
}

class WaterfallPainter extends CustomPainter {
  final ui.Image? image;
  
  WaterfallPainter({this.image});
  
  @override
  void paint(Canvas canvas, Size size) {
    if (image == null) return;
    
    canvas.drawImageRect(
      image!,
      Rect.fromLTWH(0, 0, image!.width.toDouble(), image!.height.toDouble()),
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint(),
    );
  }
  
  @override
  bool shouldRepaint(WaterfallPainter oldDelegate) {
    return image != oldDelegate.image;
  }
}
```

### Task 4.3: Signal Info Panel

```dart
// lib/widgets/signal_info_panel.dart
import 'package:flutter/material.dart';

class SignalInfoPanel extends StatelessWidget {
  final Track? selectedTrack;
  
  const SignalInfoPanel({
    super.key,
    this.selectedTrack,
  });
  
  @override
  Widget build(BuildContext context) {
    if (selectedTrack == null) {
      return const SizedBox.shrink();
    }
    
    final track = selectedTrack!;
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildRow('Track ID', '${track.trackId}'),
          _buildRow('Class', track.className.isEmpty ? 'Class ${track.classId}' : track.className),
          _buildRow('State', track.state),
          _buildRow('Confidence', '${(track.confidence * 100).toStringAsFixed(1)}%'),
          const Divider(color: Colors.white24, height: 16),
          _buildRow('Center Freq', _formatFrequency(track.freqCenterHz)),
          _buildRow('Bandwidth', _formatFrequency(track.freqBandwidthHz)),
          if (track.motionMode == 'drifting') ...[
            const Divider(color: Colors.white24, height: 16),
            _buildRow('Motion', 'Drifting'),
            _buildRow('Drift Rate', '${track.driftRateHzPerSec.toStringAsFixed(1)} Hz/s'),
          ],
        ],
      ),
    );
  }
  
  Widget _buildRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '$label: ',
            style: const TextStyle(
              color: Colors.white60,
              fontSize: 12,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
  
  String _formatFrequency(double hz) {
    if (hz >= 1e9) {
      return '${(hz / 1e9).toStringAsFixed(3)} GHz';
    } else if (hz >= 1e6) {
      return '${(hz / 1e6).toStringAsFixed(3)} MHz';
    } else if (hz >= 1e3) {
      return '${(hz / 1e3).toStringAsFixed(3)} kHz';
    } else {
      return '${hz.toStringAsFixed(1)} Hz';
    }
  }
}
```

### Task 4.4: WebSocket Provider

```dart
// lib/providers/websocket_provider.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';

class WebSocketProvider {
  WebSocketChannel? _channel;
  final _waterfallController = StreamController<Uint8List>.broadcast();
  final _trackController = StreamController<List<Track>>.broadcast();
  
  Stream<Uint8List> get waterfallStream => _waterfallController.stream;
  Stream<List<Track>> get trackStream => _trackController.stream;
  
  bool _isConnected = false;
  bool get isConnected => _isConnected;
  
  Future<void> connect(String url) async {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(url));
      _isConnected = true;
      
      _channel!.stream.listen(
        _onMessage,
        onError: _onError,
        onDone: _onDone,
      );
    } catch (e) {
      _isConnected = false;
      rethrow;
    }
  }
  
  void _onMessage(dynamic data) {
    if (data is! Uint8List) return;
    
    // Parse message header (8 bytes)
    final byteData = ByteData.sublistView(data);
    final messageType = byteData.getUint8(0);
    final frameId = byteData.getUint32(4, Endian.little);
    
    switch (messageType) {
      case 0x01: // Frame
        // Extract RGBA payload after header
        final rgbaData = data.sublist(8);
        _waterfallController.add(rgbaData);
        break;
        
      case 0x02: // TrackUpdate
        final tracks = _parseTrackUpdate(data, frameId);
        _trackController.add(tracks);
        break;
        
      case 0x03: // Status
        // Handle status messages
        break;
    }
  }
  
  List<Track> _parseTrackUpdate(Uint8List data, int frameId) {
    final byteData = ByteData.sublistView(data);
    final trackCount = byteData.getUint16(8, Endian.little);
    final inferenceLatency = byteData.getFloat32(10, Endian.little);
    final trackerLatency = byteData.getFloat32(14, Endian.little);
    
    final tracks = <Track>[];
    int offset = 18; // Start of track array
    
    for (int i = 0; i < trackCount; i++) {
      final track = Track.fromBinary(byteData, offset);
      tracks.add(track);
      offset += 44; // Size of one track record
    }
    
    return tracks;
  }
  
  void _onError(Object error) {
    _isConnected = false;
    // Handle reconnection logic
  }
  
  void _onDone() {
    _isConnected = false;
  }
  
  void disconnect() {
    _channel?.sink.close();
    _isConnected = false;
  }
  
  void dispose() {
    disconnect();
    _waterfallController.close();
    _trackController.close();
  }
}
```

**Acceptance criteria:**
- [ ] All tracks render in single paint call
- [ ] Corner markers only (no fill) by default
- [ ] Selected track shows full rectangle with subtle fill
- [ ] Opacity reflects confidence and age
- [ ] No frame drops at 50+ tracks
- [ ] Signal texture remains visible through overlays

---

## Widget Hierarchy

```
WaterfallWithOverlay
├── CustomPaint
│   ├── WaterfallPainter (background)
│   └── TrackOverlayPainter (foreground)
├── GestureDetector (tap handling)
└── SignalInfoPanel (overlay, positioned)
```

---

## Performance Optimization Tips

### 1. Avoid Per-Frame Allocations

```dart
// BAD: Creates new Paint every frame
void paint(Canvas canvas, Size size) {
  final paint = Paint()..color = Colors.red;  // ❌ Allocation
}

// GOOD: Reuse pre-allocated Paint
final Paint _paint = Paint();  // ✅ Allocated once

void paint(Canvas canvas, Size size) {
  _paint.color = Colors.red;  // Just modify
}
```

### 2. Use shouldRepaint Effectively

```dart
@override
bool shouldRepaint(TrackOverlayPainter oldDelegate) {
  // Only repaint when necessary
  return tracks != oldDelegate.tracks ||
         selectedTrackId != oldDelegate.selectedTrackId;
}
```

### 3. Batch Track Updates

```dart
// Receive tracks, but only trigger rebuild once
void _onTracksUpdate(List<Track> tracks) {
  _pendingTracks = tracks;
  _scheduleRebuild();
}

void _scheduleRebuild() {
  if (_rebuildScheduled) return;
  _rebuildScheduled = true;
  
  SchedulerBinding.instance.addPostFrameCallback((_) {
    _rebuildScheduled = false;
    setState(() {
      _tracks = _pendingTracks;
    });
  });
}
```

### 4. Use RepaintBoundary

```dart
// Isolate waterfall from overlay repaints
RepaintBoundary(
  child: CustomPaint(
    painter: WaterfallPainter(image: _image),
  ),
)
```

### 5. CRITICAL: Explicit ui.Image Disposal

```dart
// NOTE: ui.Image is a GPU resource. Unlike most Dart objects, it is NOT
// automatically garbage collected. Failing to call dispose() will cause
// GPU memory to grow unbounded, leading to crashes after hours of operation.
// This is the #1 cause of memory leaks in Flutter apps with custom rendering.

// lib/providers/waterfall_provider.dart
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';

class WaterfallProvider extends ChangeNotifier {
  ui.Image? _currentImage;
  final int _waterfallWidth;
  final int _waterfallHeight;
  
  WaterfallProvider({
    required int waterfallWidth,
    required int waterfallHeight,
  }) : _waterfallWidth = waterfallWidth,
       _waterfallHeight = waterfallHeight;
  
  ui.Image? get currentImage => _currentImage;
  
  void updateWaterfallImage(Uint8List rgba) {
    // CRITICAL: Dispose previous image to prevent GPU memory leak
    _currentImage?.dispose();
    
    ui.decodeImageFromPixels(
      rgba,
      _waterfallWidth,
      _waterfallHeight,
      ui.PixelFormat.rgba8888,
      (ui.Image img) {
        _currentImage = img;
        notifyListeners();
      },
    );
  }
  
  @override
  void dispose() {
    // Clean up on provider disposal
    _currentImage?.dispose();
    super.dispose();
  }
}
```

**Why this matters:**
- `ui.Image` wraps a GPU texture handle
- Dart's garbage collector does NOT track GPU memory
- Without explicit `dispose()`, GPU memory grows continuously
- Symptoms: App slows down, then crashes after 1-4 hours
- This is especially critical for waterfall displays updating 30+ times/second

---

## Testing: Frame Buffer with Dropped Frames

Verify the frame_id→row mapping handles dropped frames correctly:

```dart
// test/frame_buffer_test.dart
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FrameBuffer', () {
    test('frame_id_to_row handles dropped frames correctly', () {
      final buffer = FrameBuffer(maxFrames: 100);
      
      // Simulate frames with gaps (dropped frames 4 and 7)
      final frameIds = [1, 2, 3, 5, 6, 8, 9, 10];
      for (final fid in frameIds) {
        buffer.addFrame(SpectrogramFrame(frameId: fid));
      }
      
      // Row should be position in buffer, not ID difference
      expect(buffer.frameIdToRow(10, 8), equals(0));  // newest
      expect(buffer.frameIdToRow(9, 8), equals(1));
      expect(buffer.frameIdToRow(8, 8), equals(2));
      expect(buffer.frameIdToRow(6, 8), equals(3));   // NOT 4 (10-6)
      expect(buffer.frameIdToRow(5, 8), equals(4));   // NOT 5 (10-5)
      expect(buffer.frameIdToRow(3, 8), equals(5));
      
      // Frame 4 was dropped - should return null, not a wrong row
      expect(buffer.frameIdToRow(4, 8), isNull);
      
      // Frame 7 was dropped - should return null
      expect(buffer.frameIdToRow(7, 8), isNull);
    });
    
    test('frame_id_to_row returns null for frame outside visible range', () {
      final buffer = FrameBuffer(maxFrames: 100);
      
      for (int i = 1; i <= 20; i++) {
        buffer.addFrame(SpectrogramFrame(frameId: i));
      }
      
      // Display only shows 8 rows (frames 13-20)
      expect(buffer.frameIdToRow(20, 8), equals(0));
      expect(buffer.frameIdToRow(13, 8), equals(7));
      expect(buffer.frameIdToRow(12, 8), isNull);  // Outside visible range
      expect(buffer.frameIdToRow(1, 8), isNull);   // Way outside
    });
  });
}
```
