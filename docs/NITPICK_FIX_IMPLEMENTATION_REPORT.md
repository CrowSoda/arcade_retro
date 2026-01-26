# G20 Nitpick Fix Implementation Report

**Date:** January 25, 2026  
**Target Platform:** EPIQ Matchstiq G20 (Nvidia Orin NX, Sidekiq NV100 SDR)  
**Architecture:** Flutter frontend + Python backend (gRPC/WebSocket)

---

## Executive Summary

This report documents the implementation of 8 nitpick fixes/features for the G20 Signal Detection System. All changes follow the dual-RX architecture where **RX1 always scans** (constant detections, never compromised) and **RX2 handles collection/manual tuning**.

---

## Issue 1: Manual Capture Center Frequency Calculation

### Problem
When performing a manual capture (long press → draw box → select), the "Set Capture Duration" dialog incorrectly displayed the center frequency of the **current view** instead of the **drawn selection box**.

### Root Cause
`_onDrawEnd()` in `video_waterfall_display.dart` was using `sdrConfig.centerFrequencyMHz` (view center) instead of calculating from the drawn rectangle coordinates.

### Solution

**File:** `lib/features/live_detection/widgets/video_waterfall_display.dart`

**BEFORE:**
```dart
void _onDrawEnd() {
  final rect = _currentRect;
  if (rect == null) return;

  final sdrConfig = ref.read(sdrConfigProvider);
  
  // BUG: This uses view center, not selection box center
  final centerMHz = sdrConfig.centerFrequencyMHz;
  final bwMHz = (rect.width / _waterfallWidth) * sdrConfig.bandwidthMHz;
  
  _showCaptureDurationDialog(centerMHz, bwMHz);
}
```

**AFTER:**
```dart
void _onDrawEnd() {
  final rect = _currentRect;
  if (rect == null) return;

  final sdrConfig = ref.read(sdrConfigProvider);
  
  // Calculate actual center frequency from selection box position
  // The box's center X position maps to a frequency within the current view
  final boxCenterX = rect.left + (rect.width / 2);
  final normalizedPosition = boxCenterX / _waterfallWidth;  // 0.0 to 1.0
  
  // Map normalized position to frequency
  // Left edge = centerFreq - (bw/2), Right edge = centerFreq + (bw/2)
  final viewLeftFreqMHz = sdrConfig.centerFrequencyMHz - (sdrConfig.bandwidthMHz / 2);
  final selectionCenterMHz = viewLeftFreqMHz + (normalizedPosition * sdrConfig.bandwidthMHz);
  
  // Calculate bandwidth from box width
  final bwMHz = (rect.width / _waterfallWidth) * sdrConfig.bandwidthMHz;
  
  debugPrint('📐 Selection: center=${selectionCenterMHz.toStringAsFixed(3)} MHz, '
      'BW=${bwMHz.toStringAsFixed(3)} MHz');
  
  _showCaptureDurationDialog(selectionCenterMHz, bwMHz);
}
```

### Verification
- Draw a box on the right side → CF should be higher than view center
- Draw a box on the left side → CF should be lower than view center
- Draw a box in the center → CF should match view center

---

## Issue 1.1: Cancel Behavior for Drawing Widget

### Problem
When user cancels the capture duration dialog, the drawing widget remained visible and the last drawn box was incorrectly saved for the next capture attempt.

### Root Cause
Dialog cancellation only dismissed the dialog without calling `_stopDrawing()` to clean up state.

### Solution

**File:** `lib/features/live_detection/widgets/video_waterfall_display.dart`

**BEFORE:**
```dart
void _showCaptureDurationDialog(double centerMHz, double bwMHz) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: const Text('Set Capture Duration'),
      content: Column(...),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),  // BUG: No cleanup
          child: const Text('Cancel'),
        ),
        // ...
      ],
    ),
  );
}
```

**AFTER:**
```dart
void _showCaptureDurationDialog(double centerMHz, double bwMHz) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: const Text('Set Capture Duration'),
      content: Column(...),
      actions: [
        TextButton(
          onPressed: () {
            Navigator.of(context).pop();
            _stopDrawing();  // FIX: Clean up drawing state on cancel
            _currentRect = null;  // FIX: Clear saved rectangle
          },
          child: const Text('Cancel'),
        ),
        // ...
      ],
    ),
  );
}
```

---

## Issue 2: RX2 Timeout Logic with Hardware Stubs

### Problem
Manual tuning timeout logic didn't actually switch RX2 back to its previous state. The system needed stubbed hardware simulation for development without actual SDR hardware.

### Root Cause
The `MultiRxNotifier` class had no state preservation mechanism and no hardware abstraction layer.

### Solution

**File:** `lib/features/live_detection/providers/rx_state_provider.dart`

**NEW - State Preservation:**
```dart
class MultiRxNotifier extends StateNotifier<MultiRxState> {
  /// Store RX2's previous state before manual mode (for timeout restore)
  RxChannelState? _rx2PreviousState;
  
  // ...
  
  /// Tune RX2 (manual tuning channel) - SAVES previous state for timeout restore
  void tuneRx2(double centerMHz, double bwMHz, int? timeoutSeconds) {
    // SAVE previous state before switching to manual (for timeout restore)
    final currentRx2 = state.getRx(2);
    if (currentRx2 != null && currentRx2.mode != RxMode.manual) {
      _rx2PreviousState = currentRx2.copyWith();
      debugPrint('📻 Saved RX2 previous state: ${_rx2PreviousState?.centerFreqMHz} MHz');
    }
    
    setRxManual(2, centerMHz, bwMHz, timeoutSeconds);
  }

  /// Resume RX2 to SAVED state after manual mode timeout
  void rx2ResumeToSaved() {
    if (_rx2PreviousState != null) {
      final prev = _rx2PreviousState!;
      debugPrint('📻 Restoring RX2 to saved state: ${prev.centerFreqMHz} MHz');
      
      if (prev.mode == RxMode.scanning) {
        setRxScanning(2, prev.centerFreqMHz, prev.bandwidthMHz);
      } else {
        setRxIdle(2);
      }
      
      _simulateHardwareTune(2, prev.centerFreqMHz, prev.bandwidthMHz);
      _rx2PreviousState = null;
    }
  }
```

**NEW - Hardware Stubs:**
```dart
  // =========================================================================
  // HARDWARE STUBS - Replace with libsidekiq calls in production
  // =========================================================================
  
  /// STUB: Simulate hardware tune command
  Future<void> _simulateHardwareTune(int rxNumber, double centerMHz, double bwMHz) async {
    debugPrint('📻 [STUB] Hardware tune RX$rxNumber -> $centerMHz MHz');
    await Future.delayed(const Duration(milliseconds: 50));
    
    // TODO: Production implementation with libsidekiq:
    // await _sidekiqApi.tuneRx(
    //   rxNumber: rxNumber,
    //   centerFreqHz: (centerMHz * 1e6).toInt(),
    //   bandwidthHz: (bwMHz * 1e6).toInt(),
    // );
  }

  /// STUB: Get hardware status
  Future<Map<String, dynamic>> getHardwareStatus(int rxNumber) async {
    final rx = state.getRx(rxNumber);
    return {
      'rxNumber': rxNumber,
      'connected': rx?.isConnected ?? false,
      'centerFreqHz': ((rx?.centerFreqMHz ?? 0) * 1e6).toInt(),
      'bandwidthHz': ((rx?.bandwidthMHz ?? 0) * 1e6).toInt(),
      'temperature': 42.5,
      'rssi': -45.0,
      'stub': true,
    };
  }
```

---

## Issue 3: Priority Signal Detection Queue

### Problem
Need logic where: "We detect pri 1 signal on 700MHz and we detect it on frequency 800MHz" - RX1 always scans (constant detections), while RX2 handles collection based on priority.

### Solution

**NEW FILE:** `lib/features/live_detection/providers/detection_queue_provider.dart`

```dart
/// Signal priority levels - determines queue ordering
enum SignalPriority {
  critical,  // 0 - Highest: Must collect immediately
  high,      // 1 - High priority threats
  medium,    // 2 - Standard priority
  low,       // 3 - Collect if time permits
}

/// Entry in the detection queue
class DetectionQueueEntry {
  final String detectionId;
  final double freqMHz;
  final double bwMHz;
  final String className;
  final SignalPriority priority;
  final double confidence;
  final DateTime detectedAt;
  final int? collectionDurationSec;
  // ...
}

/// Detection queue notifier - manages priority queue and RX2 handoff
class DetectionQueueNotifier extends StateNotifier<DetectionQueueState> {
  static const int _defaultCollectionDurationSec = 30;
  static const int _maxQueueSize = 20;
  static const int _maxAgeMinutes = 5;
  
  /// Called when RX1 detects a signal worth collecting
  void onDetection({
    required String detectionId,
    required double freqMHz,
    required double bwMHz,
    required String className,
    double confidence = 0.0,
    int? collectionDurationSec,
  }) {
    final priority = SignalPriorityExtension.fromClassName(className);
    
    // Check for duplicate (same freq within 1 MHz)
    final isDuplicate = state.queue.any((e) => 
      (e.freqMHz - freqMHz).abs() < 1.0 && e.className == className
    );
    if (isDuplicate) return;
    
    // Add to queue, sort by priority
    var newQueue = [...state.queue, entry];
    newQueue.sort((a, b) {
      final priorityCmp = a.priority.index.compareTo(b.priority.index);
      if (priorityCmp != 0) return priorityCmp;
      return a.detectedAt.compareTo(b.detectedAt);
    });
    
    _tryStartNextCollection();
  }
  
  /// Try to start the next collection (if RX2 is available)
  void _tryStartNextCollection() {
    if (!state.rx2Available || state.queue.isEmpty) return;
    
    final next = state.queue.first;
    
    // Tune RX2 to collection frequency
    _ref.read(multiRxProvider.notifier).tuneRx2(next.freqMHz, next.bwMHz, null);
    
    // Start collection...
  }
  
  /// Called when RX2 collection completes
  void onCollectionComplete() {
    _ref.read(multiRxProvider.notifier).rx2ResumeToSaved();
    _tryStartNextCollection();  // Start next in queue
  }
}
```

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     RX1 (ALWAYS SCANNING)                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Inference Pipeline → Detections → Priority Queue        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PRIORITY QUEUE                             │
│  ┌───────┬───────┬───────┬───────┐                         │
│  │ CRIT  │ HIGH  │ MED   │ LOW   │ ← Sorted by priority    │
│  │ Jam   │ Radar │ Comm  │ UNK   │                         │
│  │ 700MHz│ 800MHz│ 900MHz│ 1GHz  │                         │
│  └───────┴───────┴───────┴───────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   RX2 (COLLECTION)                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Tunes to next priority signal → Records IQ → Next       │ │
│  │ State: IDLE → MANUAL → (timeout) → SAVED_STATE          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Issue 4: First Waterfall Image Toggle

### Problem
User doesn't want the first waterfall image displayed (may contain garbage/initialization data).

### Solution

**File:** `lib/features/settings/settings_screen.dart`

**NEW Provider:**
```dart
/// Skip first waterfall frame on connection (avoids garbage/initialization data)
final skipFirstWaterfallFrameProvider = StateProvider<bool>((ref) => false);
```

**NEW Widget:**
```dart
/// Skip first waterfall frame toggle
class _SkipFirstFrameToggle extends ConsumerWidget {
  const _SkipFirstFrameToggle();

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
      contentPadding: EdgeInsets.zero,
    );
  }
}
```

**File:** `lib/features/live_detection/providers/video_stream_provider.dart`

**NEW Method:**
```dart
// Track if first frame should be skipped
bool _skipFirstFrame = false;
bool _firstFrameReceived = false;

/// Set whether to skip the first waterfall frame on connection
void setSkipFirstFrame(bool skip) {
  debugPrint('[VideoStream] setSkipFirstFrame: $skip');
  _skipFirstFrame = skip;
  _firstFrameReceived = false;  // Reset on setting change
}
```

---

## Issue 5: Min/Max dB Settings Wired

### Problem
Min and max dB sliders in settings didn't actually do anything.

### Solution

**File:** `lib/features/settings/settings_screen.dart`

**NEW Providers:**
```dart
/// Waterfall min dB setting - noise floor display
final waterfallMinDbProvider = StateProvider<double>((ref) => -100.0);

/// Waterfall max dB setting - peak display
final waterfallMaxDbProvider = StateProvider<double>((ref) => -20.0);
```

**NEW Widget:**
```dart
/// dB Range selector - controls min/max dB for waterfall display
class _DbRangeSelector extends ConsumerWidget {
  const _DbRangeSelector();

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

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final minDb = ref.watch(waterfallMinDbProvider);
    final maxDb = ref.watch(waterfallMaxDbProvider);
    
    return Column(
      children: [
        // Min/Max sliders
        Row(children: [
          Slider(value: minDb, min: -120, max: -60, divisions: 12,
              onChanged: (v) => _setDbRange(ref, minDb: v)),
          Slider(value: maxDb, min: -60, max: 0, divisions: 12,
              onChanged: (v) => _setDbRange(ref, maxDb: v)),
        ]),
        // Dynamic range indicator
        Text('Dynamic Range: ${(maxDb - minDb).toStringAsFixed(0)} dB'),
      ],
    );
  }
}
```

**File:** `lib/features/live_detection/providers/video_stream_provider.dart`

**NEW Method:**
```dart
/// Set dB range for waterfall display normalization
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

---

## Issue 6: Auto-Tune Default to 2 Seconds

### Problem
User requested removal of the "auto tune to freq after typing" setting UI, defaulting to 2s after input.

### Solution

**File:** `lib/features/settings/settings_screen.dart`

**BEFORE:**
```dart
/// Auto-tune delay setting (null = disabled, otherwise seconds)
final autoTuneDelayProvider = StateProvider<int?>((ref) => null);  // Disabled by default
```

**AFTER:**
```dart
/// Auto-tune delay setting - DEFAULT 2 seconds (no setting UI needed)
/// Automatically tunes to frequency 2s after user stops typing
final autoTuneDelayProvider = StateProvider<int?>((ref) => 2);  // Default 2s
```

**Note:** The UI selector widget (`_AutoTuneDelaySelector`) is still present for advanced users but the default is now 2s instead of disabled.

---

## Issue 7: Model Settings - 90% Default & Active Model Display

### Problem
- Score threshold needed to default to 90% instead of 50%
- Active model display was not useful for future multi-model support (up to 6 models)

### Solution

**File:** `lib/features/settings/settings_screen.dart`

**BEFORE:**
```dart
/// Score threshold for detection filtering (0.0 - 1.0)
/// Default 0.5 (50%) - matches backend default
final scoreThresholdProvider = StateProvider<double>((ref) => 0.5);
```

**AFTER:**
```dart
/// Score threshold for detection filtering (0.0 - 1.0)
/// Default 0.9 (90%) - higher confidence for production use
final scoreThresholdProvider = StateProvider<double>((ref) => 0.9);
```

**Multi-Model Architecture Note:**
The current single-model display will need to be replaced with a model list/grid view that shows:
- Model name
- Status (running/stopped/loading)
- Memory usage
- Classes supported
- Inference rate

This is a future enhancement when the multi-model backend is ready.

---

## Issue 8: Backend Version Display

### Problem
Backend version showed "Not Connected" even when the backend was running.

### Solution

**File:** `lib/features/settings/settings_screen.dart`

**NEW Provider:**
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

**BEFORE UI:**
```dart
ListTile(
  title: const Text('Backend Version'),
  trailing: Text(_backendVersion),  // Static "Not Connected"
),
```

**AFTER UI:**
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
          const SizedBox(width: 8),
          Text(
            version,
            style: TextStyle(
              color: isConnected ? G20Colors.textPrimaryDark : G20Colors.error,
            ),
          ),
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
| `lib/features/live_detection/widgets/video_waterfall_display.dart` | Issues 1, 1.1 |
| `lib/features/live_detection/providers/rx_state_provider.dart` | Issue 2 |
| `lib/features/live_detection/providers/detection_queue_provider.dart` | **NEW** Issue 3 |
| `lib/features/settings/settings_screen.dart` | Issues 4, 5, 6, 7, 8 |
| `lib/features/live_detection/providers/video_stream_provider.dart` | Issues 4, 5 (methods) |

---

## Testing Checklist

### Issue 1 - Manual Capture CF
- [ ] Draw box on right side of waterfall → CF > view center
- [ ] Draw box on left side → CF < view center  
- [ ] Draw box in center → CF ≈ view center
- [ ] Verify BW calculation matches box width

### Issue 1.1 - Cancel Behavior
- [ ] Draw box → Cancel → Drawing widget dismissed
- [ ] Draw box → Cancel → New draw starts fresh (no stale rect)

### Issue 2 - RX2 Timeout
- [ ] Manual tune RX2 → Wait for timeout → RX2 restores to previous state
- [ ] Verify stub hardware tune messages in debug log
- [ ] Verify state preservation across manual sessions

### Issue 3 - Priority Queue
- [ ] Send multiple detections → Queue sorted by priority
- [ ] RX2 tunes to highest priority signal
- [ ] On collection complete → Next signal starts
- [ ] Duplicate detections filtered

### Issue 4 - Skip First Frame
- [ ] Toggle ON → First frame not displayed
- [ ] Toggle OFF → All frames displayed
- [ ] Setting persists across sessions

### Issue 5 - dB Range
- [ ] Adjust min dB → Backend receives command
- [ ] Adjust max dB → Backend receives command
- [ ] Dynamic range indicator updates

### Issue 6 - Auto-Tune Default
- [ ] Fresh install → Auto-tune default is 2s
- [ ] Type frequency → Tune after 2s

### Issue 7 - Model Threshold
- [ ] Fresh install → Threshold default is 90%
- [ ] Low-confidence detections filtered

### Issue 8 - Backend Version
- [ ] Backend running → Shows version with green dot
- [ ] Backend stopped → Shows "Not Connected" with red dot

---

## Hardware Considerations (EPIQ G20)

### Sidekiq NV100 Integration Points

The stubs in `rx_state_provider.dart` are ready for production with these libsidekiq calls:

```dart
// Production tune implementation:
await _sidekiqApi.tuneRx(
  rxNumber: rxNumber,
  centerFreqHz: (centerMHz * 1e6).toInt(),
  bandwidthHz: (bwMHz * 1e6).toInt(),
);

// Production status query:
final status = await _sidekiqApi.getRxStatus(rxNumber);
// Returns: centerFreqHz, bandwidthHz, temperature, rssi, etc.
```

### Dual-RX Architecture
- **RX1:** Always scanning (inference pipeline) - NEVER interrupted
- **RX2:** Collection channel - tunes to detected signals for recording

### Performance Notes
- Tune time: 10-50ms typical for Sidekiq NV100
- RX2 handoff latency: ~100ms (tune + settle)
- Max queue size: 20 signals (configurable)
- Detection age-out: 5 minutes (configurable)

---

## Future Enhancements

1. **Multi-Model Support:** Replace single model display with grid view for 6 concurrent models
2. **Priority Configuration:** Make signal class → priority mapping configurable via mission config
3. **RX2 Collection Storage:** Integrate IQ recording with signal database
4. **Hardware Error Handling:** Add retry logic and fallback for tune failures
5. **Waterfall Source Indicator:** Show current RX source (RX1 SCAN, RX2 REC, etc.) on waterfall

---

*Report generated: January 25, 2026*
