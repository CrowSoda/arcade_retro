# G20 Comprehensive Code Cleanup & Quality Assurance Plan

**Created:** January 25, 2026  
**Scope:** Full codebase review for SRP compliance, code quality, and logical correctness

---

## Executive Summary

After reviewing the entire g20_demo codebase (Flutter frontend + Python backend), I've identified opportunities for improvement across several categories:

| Category | Issues Found | Priority |
|----------|-------------|----------|
| Dead Code | 5 files | 🔴 HIGH |
| SRP Violations | 8 major areas | 🔴 HIGH |
| Logical Issues | 6 potential bugs | 🔴 HIGH |
| Provider Sprawl | 15+ scattered providers | 🟠 MEDIUM |
| Code Duplication | 4 areas | 🟠 MEDIUM |
| Debug Code | 62+ emoji instances | 🟡 LOW |
| Documentation | Missing API docs | 🟡 LOW |

---

## ✅ WHAT'S GOOD (Keep These Patterns)

### 1. **Architecture Foundation**
- Clean feature-based folder structure (`lib/features/`)
- Separation of providers, widgets, and models
- Riverpod for state management (good choice)
- gRPC + WebSocket hybrid backend design

### 2. **Good Patterns In Use**
- `ConsumerStatefulWidget` for stateful screens with Riverpod
- Proper use of `StateNotifier` for complex state
- `copyWith` pattern on state classes
- Proper cleanup in `dispose()` methods

### 3. **Backend Strengths**
- GPU-accelerated FFT processing (well optimized)
- Parent process watchdog for orphan prevention
- Signal handler coordination for graceful shutdown
- Row-strip streaming design (bandwidth efficient)

### 4. **Specific Good Code**
```dart
// Good: Detection pruning with absolute row tracking
void pruneByAbsoluteRow(int currentRow, int bufferHeight) {
  final cutoffRow = currentRow - bufferHeight;
  state = state.where((det) => det.absoluteRow >= cutoffRow).toList();
}
```

---

## 🔴 PHASE 1: CRITICAL CLEANUP (Do First)

### 1.1 Delete Dead Code Files

**Files to delete (confirmed unused):**
```bash
rm lib/core/utils/path_resolver.dart
rm lib/features/live_detection/widgets/track_overlay.dart
rm lib/features/live_detection/widgets/waterfall_frame_handler.dart
rm lib/features/live_detection/widgets/waterfall_display.dart
rm lib/features/live_detection/models/track.dart
```

**Fix unused import in `live_detection_screen.dart`:**
```dart
// REMOVE THIS LINE:
import 'widgets/waterfall_display.dart';  // ❌ UNUSED
```

### 1.2 Fix Logical Bugs

#### Bug 1: `_skipFirstFrame` never used in condition
**File:** `video_stream_provider.dart` (line ~350)
```dart
// Current code (BUG):
bool _skipFirstFrame = false;
bool _firstFrameReceived = false;

void _handleStrip(Uint8List data) {
  // ALWAYS skips first frame, ignores _skipFirstFrame setting!
  if (!_firstFrameReceived) {
    _firstFrameReceived = true;
    debugPrint('[VideoStream] Skipping first frame (always skip on connect)');
    return;
  }
}

// FIX: Respect the setting
void _handleStrip(Uint8List data) {
  if (_skipFirstFrame && !_firstFrameReceived) {
    _firstFrameReceived = true;
    debugPrint('[VideoStream] Skipping first frame per setting');
    return;
  }
  _firstFrameReceived = true;  // Mark as received either way
  // ... rest of processing
}
```

#### Bug 2: FPS counter reset logic issue
**File:** `video_stream_provider.dart` (line ~310)
```dart
// Current code (BUG): _lastFrameTime reset causes stale FPS
if (elapsed >= 1000) {
  _measuredFps = _fpsFrameCount * 1000.0 / elapsed;
  _fpsFrameCount = 0;
  _lastFrameTime = now;  // Only reset here, loses sub-second accuracy
}

// FIX: Better rolling average
final timeSinceLastCalc = now.difference(_lastFpsCalcTime!).inMilliseconds;
if (timeSinceLastCalc >= 1000) {
  _measuredFps = _fpsFrameCount * 1000.0 / timeSinceLastCalc;
  _fpsFrameCount = 0;
  _lastFpsCalcTime = now;
}
```

#### Bug 3: Unused `mapState` variable
**File:** `live_detection_screen.dart` (line ~60)
```dart
// Current code (unused variable):
final mapState = ref.read(mapStateProvider);  // ❌ Read but never used
ref.read(mapStateProvider.notifier).zoomToFitAllDetections(detections);

// FIX: Remove the unused read
ref.read(mapStateProvider.notifier).zoomToFitAllDetections(detections);
```

#### Bug 4: Potential division by zero
**File:** `unified_pipeline.py` (line ~650)
```python
# Current code:
throughput_chunks_per_sec=session.chunk_count / max(1, elapsed / 1000)

# This is fine, but in settings_screen.dart line ~180:
Text('${((660000 - fftSize) ~/ (fftSize ~/ 2) + 1)}')  # OK, but fragile

# Consider adding validation for edge cases
```

#### Bug 5: Race condition in detection callback
**File:** `live_detection_screen.dart` (line ~43)
```dart
// Current code: Callback set in initState, but video may already be streaming
WidgetsBinding.instance.addPostFrameCallback((_) {
  final videoNotifier = ref.read(videoStreamProvider.notifier);
  videoNotifier.setDetectionCallback((detections, pts) {
    // Could miss early detections
  });
  videoNotifier.connect('localhost', 8765);  // Connect AFTER setting callback
});

// FIX: Set callback before potential auto-reconnect scenarios
// Or use a provider listener pattern instead
```

#### Bug 6: Memory leak potential in toast overlay
**File:** `inputs_panel.dart` (line ~50)
```dart
// Current code: OverlayEntry may not be removed if widget tree changes
entry = OverlayEntry(
  builder: (context) => ...,
);
overlay.insert(entry);

// FIX: Add safety check
Future.delayed(Duration(seconds: 3), () {
  if (entry.mounted) entry.remove();
});
```

---

## 🔴 PHASE 2: SRP VIOLATIONS (Refactor)

### 2.1 `video_stream_provider.dart` - Too Many Responsibilities

**Current:** 580+ lines handling:
- WebSocket connection management
- Binary message parsing
- Pixel buffer management
- Detection conversion
- FPS calculation
- PSD data handling
- Colormap commands
- FFT size commands
- dB range commands

**Recommended Split:**

```
lib/features/live_detection/providers/
├── video_stream_provider.dart      # KEEP: Connection + state coordination
├── pixel_buffer_manager.dart       # NEW: Buffer allocation, scrolling, clearing
├── waterfall_commands.dart         # NEW: setFps, setFftSize, setColormap, etc.
├── stream_message_parser.dart      # NEW: Binary header parsing
└── video_detection_model.dart      # NEW: VideoDetection class (extract from provider)
```

### 2.2 `settings_screen.dart` - Widget + Provider Mix

**Current:** 850+ lines mixing:
- 12+ StateProvider definitions (should be in separate file)
- 8+ widget classes
- Setting persistence logic

**Recommended Split:**

```
lib/features/settings/
├── settings_screen.dart           # KEEP: Main screen only
├── providers/
│   └── settings_providers.dart    # NEW: All settings StateProviders
├── widgets/
│   ├── fft_size_selector.dart     # NEW: Extract _FftSizeSelector
│   ├── score_threshold_selector.dart
│   ├── colormap_selector.dart
│   └── db_range_selector.dart
└── settings_persistence.dart      # NEW: SharedPreferences logic
```

### 2.3 `live_detection_screen.dart` - Screen as Orchestrator

**Current:** 500+ lines handling:
- Screen layout
- 5+ private widget classes inline
- Detection lifecycle management
- FPS control wiring
- Display mode toggling

**Recommended Split:**

```
lib/features/live_detection/
├── live_detection_screen.dart     # KEEP: Layout only, thin orchestration
├── widgets/
│   ├── display_mode_header.dart   # NEW: Extract _DisplayModeHeader
│   ├── mode_toggle_button.dart    # NEW: Extract _ModeToggleButton
│   ├── collapse_handle.dart       # NEW: Extract _CollapseHandle
│   ├── waterfall_psd_view.dart    # NEW: Extract _WaterfallPsdView
│   ├── mission_picker_dialog.dart # NEW: Extract mission picker
│   └── mission_card.dart          # NEW: Extract _MissionCard
└── controllers/
    └── detection_lifecycle_controller.dart  # NEW: Pruning logic
```

### 2.4 `unified_pipeline.py` - Backend Monolith

**Current:** 1100+ lines handling:
- 5+ colormap LUT generators
- FFT processing
- Inference pipeline
- WebSocket streaming
- Detection serialization
- Source selection state

**Recommended Split:**

```
backend/
├── unified_pipeline.py            # KEEP: Main pipeline orchestration
├── colormaps.py                   # NEW: All LUT generation functions
├── source_selector.py             # NEW: StreamSourceSelector class
├── fft_processor.py               # KEEP: GPUSpectrogramProcessor (gpu_fft.py)
├── detection_serializer.py        # NEW: Detection to JSON conversion
└── video_encoder.py               # NEW: RGBA strip encoding logic
```

### 2.5 `inputs_panel.dart` - UI + Business Logic Mix

**Current:** 470+ lines with:
- Manual capture state handling
- Mission loading logic
- Auto-tune timer logic
- Frequency validation
- dB range sliders

**Recommended Split:**

```
lib/features/live_detection/widgets/
├── inputs_panel.dart              # KEEP: Layout only
├── frequency_input.dart           # NEW: Freq input + validation
├── bandwidth_dropdown.dart        # KEEP: Already separate
├── timeout_selector.dart          # NEW: Timeout buttons
└── db_range_sliders.dart          # NEW: Extract _DbRangeSliders

lib/features/live_detection/controllers/
└── auto_tune_controller.dart      # NEW: Debounce timer logic
```

### 2.6 `app_shell.dart` - Mixed Concerns

**Current:** 320+ lines with:
- Navigation rail
- Recording indicator (animated)
- Multi-RX status display
- Connection indicator

**Recommended Split:**

```
lib/features/shell/
├── app_shell.dart                 # KEEP: Navigation only
├── widgets/
│   ├── recording_indicator.dart   # NEW: Animated recording pulse
│   ├── rx_status_card.dart        # NEW: _SingleRxCard
│   └── connection_indicator.dart  # NEW: Green/yellow/red dot
```

### 2.7 `detection_provider.dart` - Model + Provider + Helpers

**Current:** 300+ lines with:
- Detection model class
- DetectionNotifier
- Mock data generation
- Video detection conversion helper

**Recommended Split:**

```
lib/features/live_detection/
├── models/
│   └── detection.dart             # NEW: Detection class only
├── providers/
│   └── detection_provider.dart    # KEEP: Notifier only
└── utils/
    └── detection_converter.dart   # NEW: convertVideoDetection helper
```

### 2.8 `server.py` - Massive Multi-Service File

**Current:** 900+ lines with:
- 3 different WebSocket handlers
- gRPC services
- Signal handling
- Parent watchdog
- Multiple data classes

**Recommended Split:**

```
backend/
├── server.py                      # KEEP: Entry point + serve_both
├── grpc_services/
│   ├── device_control.py          # NEW: DeviceControlServicer
│   └── inference_service.py       # NEW: InferenceServicer
├── websocket_handlers/
│   ├── video_handler.py           # NEW: video_pipeline_handler
│   ├── unified_handler.py         # NEW: unified_pipeline_handler
│   └── inference_handler.py       # NEW: ws_inference_handler
├── lifecycle/
│   └── shutdown_coordinator.py    # NEW: Signal handling, watchdog
└── models/
    └── session_models.py          # NEW: ChannelState, CaptureSession, etc.
```

---

## 🟠 PHASE 3: PROVIDER CONSOLIDATION

### 3.1 Current Provider Sprawl

Providers are scattered across 7+ files with inconsistent organization:

**In `settings_screen.dart`:**
- `autoTuneDelayProvider`
- `scoreThresholdProvider`
- `skipFirstWaterfallFrameProvider`
- `waterfallMinDbProvider`
- `waterfallMaxDbProvider`
- `backendVersionProvider`
- `waterfallFftSizeProvider`
- `waterfallTimeSpanProvider`
- `waterfallFpsProvider`
- `waterfallColormapProvider`

**In `inputs_panel.dart`:**
- `waterfallDynamicRangeProvider`

**In `live_detection_screen.dart`:**
- `rightPanelCollapsedProvider`

### 3.2 Recommended Provider Organization

```
lib/core/providers/
├── waterfall_settings_providers.dart  # FFT, FPS, colormap, dB range, time span
├── inference_settings_providers.dart  # Score threshold, model selection
├── ui_state_providers.dart            # Panel collapsed, display mode
└── connection_providers.dart          # Backend version, connection status

lib/features/live_detection/providers/
├── detection_provider.dart            # Detection state
├── video_stream_provider.dart         # Connection + streaming
├── waterfall_provider.dart            # Display state
└── scanner_provider.dart              # Mission scanning
```

---

## 🟠 PHASE 4: CODE DUPLICATION

### 4.1 Colormap LUT Generation (Backend)

**Current:** 5 nearly identical functions in `unified_pipeline.py`
```python
def _generate_viridis_lut(): ...  # 30 lines
def _generate_plasma_lut(): ...   # 30 lines
def _generate_inferno_lut(): ...  # 30 lines
def _generate_magma_lut(): ...    # 30 lines
def _generate_turbo_lut(): ...    # 30 lines
```

**Fix:** Single generic function
```python
def _generate_lut(control_points: list, positions: list) -> np.ndarray:
    """Generate 256-entry RGB LUT from control points."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(len(positions) - 1):
            if positions[j] <= i <= positions[j + 1]:
                t = (i - positions[j]) / (positions[j + 1] - positions[j])
                lut[i] = [int((control_points[j][c] * (1-t) + control_points[j+1][c] * t) * 255) 
                          for c in range(3)]
                break
    return lut

# Usage
VIRIDIS_LUT = _generate_lut(VIRIDIS_CONTROL_POINTS, CONTROL_POSITIONS)
PLASMA_LUT = _generate_lut(PLASMA_CONTROL_POINTS, CONTROL_POSITIONS)
```

### 4.2 Selection Option Widgets (Frontend)

**Current:** 5+ nearly identical option button widgets
- `_DelayOption`, `_TimeSpanOption`, `_FpsOption`, `_ThresholdOption`, `_FftSizeOption`

**Fix:** Generic `SegmentedOption` widget
```dart
class SegmentedOption<T> extends StatelessWidget {
  final String label;
  final String? sublabel;
  final T value;
  final bool selected;
  final VoidCallback onTap;
  final Color? activeColor;
  
  // ... single implementation
}
```

### 4.3 Toast/Dialog Patterns

**Current:** `showFadingToast()` and `showCaptureWarningDialog()` are single-use but not reusable

**Fix:** Create `lib/core/widgets/dialogs.dart` with:
- `G20Toast` - Fading overlay toast
- `G20ConfirmDialog` - Warning with confirm/cancel
- `G20ErrorDialog` - Error display

### 4.4 Mission Picker Dialog

**Current:** Duplicated in `live_detection_screen.dart` AND `inputs_panel.dart`

**Fix:** Single `lib/features/config/widgets/mission_picker_dialog.dart`

---

## 🟡 PHASE 5: DEBUG CODE CLEANUP

### 5.1 Emoji in Debug Strings

**62 instances** across 10 files. Examples:
```dart
debugPrint('📻 Tuned to ${centerMHz.toStringAsFixed(1)} MHz');
debugPrint('🗺️ Showing ${detections.length} detections on map');
debugPrint('🔗 Testing connection...');
debugPrint('💾 Settings saved');
```

**Fix:** Use consistent tagged format:
```dart
debugPrint('[Tuning] Tuned to ${centerMHz.toStringAsFixed(1)} MHz');
debugPrint('[Map] Showing ${detections.length} detections');
debugPrint('[Settings] Testing connection...');
debugPrint('[Settings] Settings saved');
```

**Files to update:**
1. `inputs_panel.dart`
2. `live_detection_screen.dart`
3. `settings_screen.dart`
4. `app_shell.dart`
5. `video_stream_provider.dart`
6. `detection_provider.dart`
7. `tuning_state_provider.dart`
8. `backend_launcher.dart`

### 5.2 Commented-Out Code

Search and remove:
```bash
grep -r "// TODO" lib/
grep -r "// TEMP" lib/
grep -r "// DEBUG" lib/
```

---

## 🟡 PHASE 6: DOCUMENTATION

### 6.1 Missing API Documentation

**Files needing dartdoc comments:**
- All providers need `///` doc comments explaining purpose
- All public classes need class-level documentation
- Complex methods need parameter documentation

### 6.2 Architecture Documentation

Create `docs/ARCHITECTURE.md`:
```markdown
# G20 Architecture

## Data Flow
IQ File → GPU FFT → Waterfall Rows → WebSocket → Flutter → RawImage

## State Management
Riverpod StateNotifiers for all mutable state
StateProviders for simple settings

## Backend Communication
- WebSocket for streaming (8765)
- gRPC for control commands (50051)
```

---

## Implementation Checklist

### Phase 1: Critical (1-2 days)
- [ ] Delete 5 dead code files
- [ ] Fix unused import in live_detection_screen.dart
- [ ] Fix `_skipFirstFrame` logic bug
- [ ] Fix FPS counter reset logic
- [ ] Remove unused `mapState` variable
- [ ] Add toast overlay safety check

### Phase 2: SRP Refactoring (3-5 days)
- [ ] Extract video_stream_provider components
- [ ] Extract settings_screen providers to separate file
- [ ] Extract settings_screen widgets
- [ ] Extract live_detection_screen widgets
- [ ] Split unified_pipeline.py colormap code
- [ ] Extract server.py handlers

### Phase 3: Provider Consolidation (1-2 days)
- [ ] Create waterfall_settings_providers.dart
- [ ] Move UI state providers to ui_state_providers.dart
- [ ] Update all imports

### Phase 4: Deduplication (1-2 days)
- [ ] Generic colormap LUT generator
- [ ] Generic SegmentedOption widget
- [ ] Consolidate dialogs
- [ ] Single mission picker implementation

### Phase 5: Debug Cleanup (0.5 days)
- [ ] Replace all emoji debug strings with tags
- [ ] Remove commented-out code
- [ ] Standardize log format

### Phase 6: Documentation (1 day)
- [ ] Add dartdoc to all providers
- [ ] Create ARCHITECTURE.md
- [ ] Document WebSocket protocol

---

## Estimated Total Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1 | 1-2 days | 🔴 DO NOW |
| Phase 2 | 3-5 days | 🔴 DO NEXT |
| Phase 3 | 1-2 days | 🟠 MEDIUM |
| Phase 4 | 1-2 days | 🟠 MEDIUM |
| Phase 5 | 0.5 days | 🟡 LOW |
| Phase 6 | 1 day | 🟡 LOW |
| **TOTAL** | **8-13 days** | |

---

## Quick Wins (Can Do Today)

1. ✅ Delete dead files (5 minutes)
2. ✅ Fix unused import (1 minute)
3. ✅ Fix `_skipFirstFrame` bug (10 minutes)
4. ✅ Move settings providers to new file (30 minutes)
5. ✅ Replace emojis with tags (1 hour with find/replace)

---

*Generated: January 25, 2026*
