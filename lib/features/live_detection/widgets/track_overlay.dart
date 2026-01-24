/// TrackOverlayPainter - Single-pass rendering of all track overlays.
/// 
/// INVARIANT: Minimal occlusion - corner markers only, no fill.
/// Selected track gets full rectangle with subtle fill.

import 'dart:math';
import 'package:flutter/material.dart';
import '../models/track.dart';

/// Converts class name to consistent color.
Color getTrackColor(String className) {
  // Hash class name for consistent color
  final hash = className.hashCode;
  final hue = (hash % 360).abs().toDouble();
  return HSLColor.fromAHSL(1.0, hue, 0.7, 0.5).toColor();
}

/// Custom painter for all track overlays.
/// 
/// Renders ALL tracks in a single paint call for efficiency.
/// Uses corner markers for minimal occlusion.
class TrackOverlayPainter extends CustomPainter {
  final List<Track> tracks;
  final int? selectedTrackId;
  final int displayRows;
  
  /// Map of frameId → row in display (0 = newest/top)
  final Map<int, int> frameIdToRow;
  
  /// Current newest frame ID
  final int newestFrameId;
  
  /// Corner marker length (pixels)
  final double cornerLength;
  
  TrackOverlayPainter({
    required this.tracks,
    required this.frameIdToRow,
    required this.newestFrameId,
    this.selectedTrackId,
    this.displayRows = 256,
    this.cornerLength = 8.0,
  });
  
  @override
  void paint(Canvas canvas, Size size) {
    if (tracks.isEmpty) return;
    
    for (final track in tracks) {
      _paintTrack(canvas, size, track);
    }
  }
  
  void _paintTrack(Canvas canvas, Size size, Track track) {
    // Map track's lastSeenFrameId to row
    final row = _frameIdToRow(track.lastSeenFrameId);
    if (row == null) return; // Track not visible
    
    // Calculate pixel coordinates
    // X: frequency (normalized 0-1)
    final x1 = track.x1 * size.width;
    final x2 = track.x2 * size.width;
    
    // Y: row position (0 = top/newest)
    final rowHeight = size.height / displayRows;
    final trackHeight = track.height * size.height; // Approximate height
    final y1 = row * rowHeight;
    final y2 = y1 + max(trackHeight, rowHeight); // At least one row
    
    // Clamp to bounds
    if (y2 < 0 || y1 > size.height) return;
    if (x2 < 0 || x1 > size.width) return;
    
    final rect = Rect.fromLTRB(
      x1.clamp(0, size.width),
      y1.clamp(0, size.height),
      x2.clamp(0, size.width),
      y2.clamp(0, size.height),
    );
    
    // Determine color and style
    final color = getTrackColor(track.className);
    final isSelected = track.trackId == selectedTrackId;
    
    // Opacity based on confidence and state
    final opacity = _calculateOpacity(track);
    final strokeWidth = _calculateStrokeWidth(track, isSelected);
    
    final paint = Paint()
      ..color = color.withOpacity(opacity)
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;
    
    if (isSelected) {
      // Selected track: full rectangle with subtle fill
      canvas.drawRect(rect, paint);
      
      final fillPaint = Paint()
        ..color = color.withOpacity(0.1)
        ..style = PaintingStyle.fill;
      canvas.drawRect(rect, fillPaint);
    } else {
      // Normal track: corner markers only
      _paintCornerMarkers(canvas, rect, paint);
    }
  }
  
  /// Paint only corner markers (minimal occlusion).
  void _paintCornerMarkers(Canvas canvas, Rect rect, Paint paint) {
    final len = min(cornerLength, min(rect.width, rect.height) / 3);
    
    final path = Path();
    
    // Top-left corner
    path.moveTo(rect.left, rect.top + len);
    path.lineTo(rect.left, rect.top);
    path.lineTo(rect.left + len, rect.top);
    
    // Top-right corner
    path.moveTo(rect.right - len, rect.top);
    path.lineTo(rect.right, rect.top);
    path.lineTo(rect.right, rect.top + len);
    
    // Bottom-right corner
    path.moveTo(rect.right, rect.bottom - len);
    path.lineTo(rect.right, rect.bottom);
    path.lineTo(rect.right - len, rect.bottom);
    
    // Bottom-left corner
    path.moveTo(rect.left + len, rect.bottom);
    path.lineTo(rect.left, rect.bottom);
    path.lineTo(rect.left, rect.bottom - len);
    
    canvas.drawPath(path, paint);
  }
  
  /// Calculate opacity based on confidence and age.
  double _calculateOpacity(Track track) {
    // Base opacity from confidence
    var opacity = track.confidence.clamp(0.3, 1.0);
    
    // Reduce for tentative tracks
    if (track.isTentative) {
      opacity *= 0.6;
    }
    
    // Reduce for lost tracks (fade out)
    if (track.isLost) {
      opacity *= max(0.2, 1.0 - track.ageFrames * 0.1);
    }
    
    return opacity.clamp(0.2, 1.0);
  }
  
  /// Calculate stroke width based on state and selection.
  double _calculateStrokeWidth(Track track, bool isSelected) {
    if (isSelected) return 3.0;
    if (track.isConfirmed) return 2.0;
    if (track.isTentative) return 1.5;
    return 1.0; // Lost
  }
  
  /// Convert frame ID to row index using the provided mapping.
  /// 
  /// CRITICAL: Uses the frameIdToRow map which is based on deque position,
  /// not arithmetic difference. This handles dropped frames correctly.
  int? _frameIdToRow(int frameId) {
    return frameIdToRow[frameId];
  }
  
  @override
  bool shouldRepaint(TrackOverlayPainter oldDelegate) {
    // NOTE: List identity comparison (tracks != oldDelegate.tracks) will 
    // almost always be true since we rebuild the list each update.
    // This is acceptable for v1 - track overlay is lightweight.
    // 
    // For v2 optimization, consider:
    // - Using listEquals() for deep comparison
    // - Caching track list in provider with proper equality
    // - Comparing only track count + newest track timestamp
    
    if (selectedTrackId != oldDelegate.selectedTrackId) return true;
    if (newestFrameId != oldDelegate.newestFrameId) return true;
    if (tracks.length != oldDelegate.tracks.length) return true;
    
    // v1: Always repaint when tracks list is rebuilt (common case)
    // This is acceptable overhead for <50 tracks at 30fps
    return tracks != oldDelegate.tracks;
  }
}

/// Widget wrapper for TrackOverlayPainter.
class TrackOverlay extends StatelessWidget {
  final List<Track> tracks;
  final Map<int, int> frameIdToRow;
  final int newestFrameId;
  final int? selectedTrackId;
  final int displayRows;
  
  const TrackOverlay({
    super.key,
    required this.tracks,
    required this.frameIdToRow,
    required this.newestFrameId,
    this.selectedTrackId,
    this.displayRows = 256,
  });
  
  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        painter: TrackOverlayPainter(
          tracks: tracks,
          frameIdToRow: frameIdToRow,
          newestFrameId: newestFrameId,
          selectedTrackId: selectedTrackId,
          displayRows: displayRows,
        ),
        size: Size.infinite,
      ),
    );
  }
}

/// FrameBuffer for Dart side - maintains frame_id → row mapping.
/// 
/// CRITICAL: Uses deque position, not arithmetic difference.
/// This handles dropped frames correctly.
class DartFrameBuffer {
  final int maxFrames;
  final List<int> _frameIds = [];
  final Map<int, int> _frameIdToIndex = {};
  
  DartFrameBuffer({this.maxFrames = 300});
  
  /// Add a frame ID to the buffer.
  void addFrame(int frameId) {
    // Remove oldest if at capacity
    if (_frameIds.length >= maxFrames) {
      final oldest = _frameIds.removeAt(0);
      _frameIdToIndex.remove(oldest);
    }
    
    _frameIds.add(frameId);
    _rebuildIndex();
  }
  
  void _rebuildIndex() {
    _frameIdToIndex.clear();
    for (var i = 0; i < _frameIds.length; i++) {
      _frameIdToIndex[_frameIds[i]] = i;
    }
  }
  
  /// Convert frame_id to row index.
  /// 
  /// CRITICAL: Uses deque position, not arithmetic difference.
  int? frameIdToRow(int frameId, int displayRows) {
    final idx = _frameIdToIndex[frameId];
    if (idx == null) return null;
    
    // Calculate row (0 = newest)
    final framesFromNewest = _frameIds.length - 1 - idx;
    
    if (framesFromNewest < displayRows) {
      return framesFromNewest;
    }
    
    return null; // Too old
  }
  
  /// Build the complete frameId → row map for current display.
  Map<int, int> buildFrameIdToRowMap(int displayRows) {
    final map = <int, int>{};
    
    for (var i = 0; i < min(_frameIds.length, displayRows); i++) {
      final idx = _frameIds.length - 1 - i;
      if (idx >= 0) {
        map[_frameIds[idx]] = i;
      }
    }
    
    return map;
  }
  
  int get newestFrameId => _frameIds.isEmpty ? -1 : _frameIds.last;
  int get oldestFrameId => _frameIds.isEmpty ? -1 : _frameIds.first;
  int get frameCount => _frameIds.length;
  
  void clear() {
    _frameIds.clear();
    _frameIdToIndex.clear();
  }
}
