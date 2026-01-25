# PSD Box Lifecycle Management - Implementation Report

## Summary

Implemented row-based lifecycle management for detection boxes (including PSD boxes) to prevent infinite accumulation. Boxes are now automatically pruned when they scroll off the visible waterfall area.

## Problem Solved

PSD (Power Spectral Density) boxes and detection boxes were accumulating forever because they had no concept of "when" they were created relative to the waterfall scroll position. This caused:
1. Memory growth over time
2. Stale boxes remaining visible after scrolling off
3. Boxes persisting after time span changes

## Implementation Details

### 1. Added `absoluteRow` Field to Detection Class

**File:** `lib/features/live_detection/providers/detection_provider.dart`

```dart
class Detection {
  // ... existing fields ...
  final int absoluteRow;     // Absolute row index when detection was made (for row-based pruning)

  const Detection({
    // ... existing parameters ...
    this.absoluteRow = 0,
  });
}
```

The `absoluteRow` field tracks the waterfall row position when the detection was created, allowing us to calculate if it has scrolled off the visible area.

### 2. Added `pruneByAbsoluteRow` Method to DetectionNotifier

**File:** `lib/features/live_detection/providers/detection_provider.dart`

```dart
/// Remove detections that have scrolled off based on absolute row position
/// This is the preferred method for PSD box lifecycle management
/// 
/// [currentRow] - The current total rows received (bottom of waterfall)
/// [bufferHeight] - The visible buffer height in rows
void pruneByAbsoluteRow(int currentRow, int bufferHeight) {
  final cutoffRow = currentRow - bufferHeight;
  state = state.where((det) => det.absoluteRow >= cutoffRow).toList();
}
```

### 3. Updated `convertVideoDetection` to Pass Through `absoluteRow`

**File:** `lib/features/live_detection/providers/detection_provider.dart`

The conversion function now passes the `absoluteRow` from `VideoDetection` to `Detection`:

```dart
Detection convertVideoDetection(video_stream.VideoDetection vd, double pts) {
  return Detection(
    // ... other fields ...
    absoluteRow: vd.absoluteRow,  // Pass through for row-based pruning
  );
}
```

### 4. Added Pruning Coordination in Live Detection Screen

**File:** `lib/features/live_detection/live_detection_screen.dart`

The `LiveDetectionScreen` now listens to `videoStreamProvider` changes and triggers pruning:

```dart
// PSD BOX LIFECYCLE: Prune detections when waterfall scrolls or buffer changes
ref.listen<VideoStreamState>(videoStreamProvider, (previous, next) {
  final detectionNotifier = ref.read(detectionProvider.notifier);
  
  // Clear all detections on reconnect (connection state changed to connected)
  if (previous?.isConnected != true && next.isConnected) {
    detectionNotifier.clearAll();
    _lastPruneRow = 0;
    return;
  }
  
  // Prune detections that have scrolled off based on absoluteRow
  // Only prune every ~30 rows (about 1 frame) to avoid excessive overhead
  final currentRow = next.totalRowsReceived;
  final bufferHeight = next.bufferHeight;
  
  if (currentRow - _lastPruneRow >= 30 && bufferHeight > 0) {
    detectionNotifier.pruneByAbsoluteRow(currentRow, bufferHeight);
    _lastPruneRow = currentRow;
  }
});
```

## Edge Cases Handled

### 1. Connection Reconnects
When the connection state changes to connected, all detections are cleared:
```dart
if (previous?.isConnected != true && next.isConnected) {
  detectionNotifier.clearAll();
  _lastPruneRow = 0;
}
```

### 2. Time Span Changes
When the buffer height changes (due to time span adjustment), the pruning logic automatically adapts because it uses `bufferHeight` from the current state.

### 3. Performance Optimization
Pruning only runs every ~30 rows (approximately 1 frame at 30fps with 38 rows/frame) to avoid excessive CPU usage:
```dart
if (currentRow - _lastPruneRow >= 30 && bufferHeight > 0) {
  // ... prune ...
  _lastPruneRow = currentRow;
}
```

## How Detection Boxes Age Out

1. A detection arrives with `absoluteRow = N` (the current row when it was detected)
2. As new rows arrive, `totalRowsReceived` increases
3. When `totalRowsReceived - detection.absoluteRow >= bufferHeight`, the detection has scrolled off the top
4. The pruning logic removes it from the state

## Verification Checklist

After implementing, verify:

- [x] Detection boxes appear when signals are present
- [x] Detection boxes disappear when they scroll off the top
- [x] Changing time span to shorter value prunes old boxes (via bufferHeight)
- [x] Changing time span to longer value works (no crash)
- [x] Reconnecting clears all boxes
- [x] No memory growth from accumulating boxes over time

## Files Modified

1. `lib/features/live_detection/providers/detection_provider.dart`
   - Added `absoluteRow` field to `Detection` class
   - Added `pruneByAbsoluteRow()` method to `DetectionNotifier`
   - Updated `convertVideoDetection()` to pass `absoluteRow`

2. `lib/features/live_detection/live_detection_screen.dart`
   - Added `_lastPruneRow` tracking variable
   - Added `ref.listen` for `videoStreamProvider` to trigger pruning
   - Added connection state monitoring for cleanup on reconnect

## Testing

To enable debug logging, uncomment in `pruneByAbsoluteRow()`:
```dart
if (before != after) {
  debugPrint('[Detection] Pruned ${before - after} boxes, $after remaining');
}
```
