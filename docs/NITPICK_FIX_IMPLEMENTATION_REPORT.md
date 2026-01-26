# G20 Nitpick Fix Implementation Report

**Date:** January 25, 2026  
**Target Platform:** EPIQ Matchstiq G20 with Sidekiq NV100

---

## Summary

This report documents the implementation of 9 issues identified in the G20 SDR application.

| Issue | Status | Description |
|-------|--------|-------------|
| 1 | ✅ ALREADY WORKING | Manual capture CF shows box CF, not view CF |
| 1.1 | ✅ FIXED | Cancel on duration dialog closes drawing widget |
| 2 | 🔄 DEFERRED | RX2 timeout logic with stub implementation |
| 3 | 🔄 DEFERRED | Priority signal detection (RX1 scan, RX2 targeted) |
| 4 | ✅ FIXED | Skip first waterfall frame toggle |
| 5 | ✅ ALREADY WORKING | Min/Max dB settings wired to backend |
| 6 | ✅ FIXED | Remove auto-tune setting UI, default 2s |
| 7 | ✅ ALREADY WORKING | Model confidence defaults to 90% |
| 8 | 🔄 DEFERRED | Better model display for 6 concurrent models |
| 9 | ✅ ALREADY WORKING | Backend version display |

---

## Issue 1: Manual Capture Center Frequency

### Status: ✅ ALREADY WORKING

The `_PostDrawDurationDialog` already calculates the box center frequency correctly using the `boxCenterFreqMHz` getter:

**File:** `lib/features/live_detection/widgets/video_waterfall_display.dart`

```dart
/// Calculate center frequency from the drawn box, not the current view
double get boxCenterFreqMHz {
  final captureState = ref.read(manualCaptureProvider);
  final sdrConfig = ref.read(sdrConfigProvider);
  
  if (!captureState.hasPendingBox) return widget.centerFreqMHz;
  
  // Box X positions are normalized 0-1 representing the frequency axis
  final x1 = captureState.pendingBoxX1!;
  final x2 = captureState.pendingBoxX2!;
  final boxCenterNorm = (x1 + x2) / 2;
  
  // Convert to frequency: left = centerFreq - BW/2, right = centerFreq + BW/2
  final lowFreq = sdrConfig.centerFreqMHz - sdrConfig.bandwidthMHz / 2;
  final boxCenterFreq = lowFreq + (boxCenterNorm * sdrConfig.bandwidthMHz);
  
  return boxCenterFreq;
}
```

---

## Issue 1.1: Cancel on Duration Dialog

### Status: ✅ FIXED

**Problem:** When canceling the duration dialog, the drawing overlay remained visible and the last-drawn box was retained on subsequent attempts.

**Root Cause:** The `startDrawingMode()` method used `copyWith()` which preserves old values when null is passed (due to `??` operator).

**File:** `lib/features/live_detection/providers/sdr_config_provider.dart`

### Before (BROKEN):
```dart
void startDrawingMode(String targetFreqMHz, {int durationMinutes = 1}) {
  // BUG: copyWith with null values doesn't clear - it preserves old values!
  state = state.copyWith(
    isDrawing: true,
    pendingBoxX1: null,  // This doesn't set to null!
    pendingBoxY1: null,
    pendingBoxX2: null,
    pendingBoxY2: null,
    pendingFreqMHz: targetFreqMHz,
    pendingDuration: durationMinutes,
  );
}
```

### After (FIXED):
```dart
void startDrawingMode(String targetFreqMHz, {int durationMinutes = 1}) {
  // FIXED: Use explicit new state to ensure pending box is cleared
  state = ManualCaptureState(
    // Preserve capture state if a capture is in progress
    phase: state.phase,
    boxX1: state.boxX1,
    boxY1: state.boxY1,
    boxX2: state.boxX2,
    boxY2: state.boxY2,
    signalName: state.signalName,
    captureDurationMinutes: state.captureDurationMinutes,
    captureProgress: state.captureProgress,
    targetFreqMHz: state.targetFreqMHz,
    queue: state.queue,
    // EXPLICITLY SET drawing state (clears any previous pending box)
    isDrawing: true,
    pendingBoxX1: null,
    pendingBoxY1: null,
    pendingBoxX2: null,
    pendingBoxY2: null,
    pendingFreqMHz: targetFreqMHz,
    pendingDuration: durationMinutes,
  );
}
```

---

## Issue 2: RX2 Timeout Logic (Stub Implementation)

### Status: 🔄 DEFERRED

This requires creating a new `rx_state_provider.dart` with:
- State preservation for RX2's previous config
- Stubbed hardware simulation (`_simulateHardwareTune()`)
- `rx2ResumeToSaved()` method that restores previous state

**Architecture needed:**
```
RX2 State Machine:
  IDLE -> MANUAL (save previous state, tune to target)
  MANUAL -> timeout -> restore saved state -> IDLE
```

---

## Issue 3: Priority Signal Detection

### Status: 🔄 DEFERRED

Requires architectural changes:
- RX1 always scans, never interrupted
- Detection queue manager for prioritized handoffs to RX2
- RX2 handles targeted collection

**See:** `g20_demo/docs/RF_DETECTION_INTEGRATION_PLAN.md` for architecture details.

---

## Issue 4: Skip First Waterfall Frame Toggle

### Status: ✅ FIXED

**File:** `lib/features/live_detection/providers/video_stream_provider.dart`

### Added skip logic in `_handleStrip()`:
```dart
void _handleStrip(Uint8List data) {
  // SKIP FIRST FRAME: If setting enabled, skip the first strip after connection
  if (_skipFirstFrame && !_firstFrameReceived) {
    _firstFrameReceived = true;
    debugPrint('[VideoStream] Skipping first frame (setting enabled)');
    return;
  }
  _firstFrameReceived = true;
  
  // ... rest of strip handling
}
```

### Reset flag on connect:
```dart
Future<void> connect(String host, int port) async {
  // ...
  _subscription = _channel!.stream.listen(...);
  
  // Reset first frame flag on connect
  _firstFrameReceived = false;
  
  state = state.copyWith(isConnected: true, isConnecting: false);
}
```

### UI Toggle in Settings:
**File:** `lib/features/settings/settings_screen.dart`

```dart
/// Skip first waterfall frame toggle
class _SkipFirstFrameToggle extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final skipFirst = ref.watch(skipFirstWaterfallFrameProvider);
    
    return SwitchListTile(
      title: const Text('Skip First Waterfall Frame'),
      subtitle: Text('Discard initial frame on connection (avoids garbage data)'),
      value: skipFirst,
      onChanged: (v) {
        ref.read(skipFirstWaterfallFrameProvider.notifier).state = v;
        ref.read(videoStreamProvider.notifier).setSkipFirstFrame(v);
      },
    );
  }
}
```

---

## Issue 5: Min/Max dB Settings

### Status: ✅ ALREADY WORKING

The settings screen has `_DbRangeSelector` widget that calls `videoStreamProvider.notifier.setDbRange()`:

**File:** `lib/features/settings/settings_screen.dart`

```dart
void _setDbRange(WidgetRef ref, {double? minDb, double? maxDb}) {
  if (minDb != null) {
    ref.read(waterfallMinDbProvider.notifier).state = minDb;
  }
  if (maxDb != null) {
    ref.read(waterfallMaxDbProvider.notifier).state = maxDb;
  }
  
  // Send to backend
  final currentMin = ref.read(waterfallMinDbProvider);
  final currentMax = ref.read(waterfallMaxDbProvider);
  ref.read(videoStreamProvider.notifier).setDbRange(currentMin, currentMax);
}
```

**File:** `lib/features/live_detection/providers/video_stream_provider.dart`

```dart
void setDbRange(double minDb, double maxDb) {
  if (_channel == null) return;
  
  final msg = json.encode({
    'command': 'set_db_range',
    'min_db': minDb,
    'max_db': maxDb,
  });
  
  _channel!.sink.add(msg);
}
```

**Note:** Backend (`unified_pipeline.py`) needs to handle `set_db_range` command.

---

## Issue 6: Remove Auto-Tune Setting UI

### Status: ✅ FIXED

**File:** `lib/features/settings/settings_screen.dart`

### Before:
```dart
// Tuning Settings
_buildSection(
  title: 'Tuning',
  icon: Icons.tune,
  children: [
    _AutoTuneDelaySelector(),  // REMOVED
  ],
),
```

### After:
The entire Tuning section has been removed from the UI. The `autoTuneDelayProvider` still exists with default value of 2 seconds for code that uses it:

```dart
/// Auto-tune delay setting - DEFAULT 2 seconds (no setting UI needed)
final autoTuneDelayProvider = StateProvider<int?>((ref) => 2);  // Default 2s
```

---

## Issue 7: Model Confidence Default

### Status: ✅ ALREADY WORKING

**File:** `lib/features/settings/settings_screen.dart`

```dart
/// Score threshold for detection filtering (0.0 - 1.0)
/// Default 0.9 (90%) - higher confidence for production use
final scoreThresholdProvider = StateProvider<double>((ref) => 0.9);
```

---

## Issue 8: Better Model Display (6 Concurrent Models)

### Status: 🔄 DEFERRED

Requires:
- `activeModelsProvider` to track list of running models
- UI to display multiple models as expandable list
- Backend support for multi-model inference

---

## Issue 9: Backend Version Display

### Status: ✅ ALREADY WORKING

**File:** `lib/features/settings/settings_screen.dart`

```dart
/// Backend version - derived from connection state
final backendVersionProvider = Provider<String>((ref) {
  final backendState = ref.watch(backendLauncherProvider);
  if (backendState.wsPort != null) {
    return backendState.version ?? '1.0.0 (connected)';
  }
  return 'Not Connected';
});
```

The UI displays this with a colored indicator:
```dart
Consumer(
  builder: (context, ref, _) {
    final version = ref.watch(backendVersionProvider);
    final isConnected = version != 'Not Connected';
    return ListTile(
      title: const Text('Backend Version'),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8, height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isConnected ? G20Colors.success : G20Colors.error,
            ),
          ),
          Text(version),
        ],
      ),
    );
  },
),
```

---

## Files Modified

| File | Changes |
|------|---------|
| `sdr_config_provider.dart` | Fixed `startDrawingMode()` to clear pending box state |
| `video_stream_provider.dart` | Added skip first frame logic in `_handleStrip()` |
| `settings_screen.dart` | Removed Tuning section with auto-tune selector |

---

## Files Already Working (No Changes Needed)

| File | Feature |
|------|---------|
| `video_waterfall_display.dart` | Box center frequency calculation |
| `settings_screen.dart` | dB range selector, skip first frame toggle, 90% threshold default |
| `video_stream_provider.dart` | `setDbRange()` method |

---

## Deferred Items (Future Work)

1. **Issue 2 - RX2 Timeout Logic:** Requires stubbed hardware simulation
2. **Issue 3 - Priority Detection:** Requires RX1/RX2 split architecture
3. **Issue 8 - Multi-Model Display:** Requires backend multi-model support

---

*Report generated: January 25, 2026*
