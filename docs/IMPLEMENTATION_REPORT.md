# G20 RF Waterfall Detection System - Implementation Report

**Date:** January 23, 2026  
**Status:** ✅ Complete (Software - Ready for Hardware Integration)  
**Tests:** 39/39 Passing

---

## Executive Summary

The G20 RF Waterfall Detection System backend has been fully implemented and tested. The system is ready for integration with G20/NV100 hardware. All hardware-dependent components have been stubbed to enable testing without physical SDR equipment.

### Key Deliverables
- **39 unit tests** covering all core modules
- **Full pipeline** from IQ capture → Spectrogram → Tracker → WebSocket → Flutter
- **Stub IQ source** for testing without hardware
- **Demo mode** that runs the complete system with simulated signals

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         IQ SOURCE                                   │
│         (StubIQSource for testing / SidekiqSource for prod)         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ IQ samples
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  CANONICAL SPECTROGRAM GENERATOR                    │
│                    (FFT → PSD → Normalize → RGBA)                   │
│                         SpectrogramFrame                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
    ┌───────────────────┐                 ┌─────────────────┐
    │ INFERENCE ENGINE  │                 │  FRAME BUFFER   │
    │ (TensorRT/ONNX)   │                 │ (frame_id sync) │
    └─────────┬─────────┘                 └────────┬────────┘
              │ Detections                         │
              ▼                                    │
    ┌───────────────────┐                          │
    │   RF TRACKER      │                          │
    │ (IoU + EMA)       │                          │
    └─────────┬─────────┘                          │
              │ Tracks                             │
              ▼                                    │
    ┌───────────────────┐                          │
    │ WEBSOCKET SERVER  │◄─────────────────────────┘
    │  (Binary Protocol)│
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐       ┌───────────────────┐
    │   FLUTTER UI      │       │  ASYNC DATABASE   │
    │  (Track Overlay)  │       │     LOGGER        │
    └───────────────────┘       └───────────────────┘
```

---

## Module Implementations

### 1. Canonical Spectrogram Pipeline (13 Tests)

**Purpose:** Single source of truth for all spectrogram data. Ensures consistency between display and inference.

**Key Invariant:** Config hash validation prevents mismatches.

```python
# backend/spectrogram/config.py
@dataclass
class SpectrogramConfig:
    fft_size: int = 4096
    window: str = 'hann'
    overlap: float = 0.5
    sample_rate_hz: float = 20e6
    psd_ref_dbm: float = -30.0
    dynamic_range_db: float = 60.0
    target_width: int = 4096
    
    def hash(self) -> int:
        """Hash for change detection - included in every frame."""
        return hash((
            self.fft_size, self.window, self.overlap,
            self.psd_ref_dbm, self.dynamic_range_db
        ))
```

```python
# backend/spectrogram/generator.py
class CanonicalSpectrogramGenerator:
    def process_iq(self, iq_samples: np.ndarray, pts: float) -> SpectrogramFrame:
        """Generate one canonical spectrogram frame."""
        # 1. Apply window
        windowed = iq_samples[:self.config.fft_size] * self._window
        
        # 2. FFT
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        
        # 3. PSD in dB
        psd_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
        
        # 4. Normalize to 0-255
        normalized = self._normalize_psd(psd_db)
        
        # 5. Apply colormap → RGBA
        rgba = self._apply_colormap(normalized)
        
        return SpectrogramFrame(
            frame_id=self._next_frame_id(),  # Monotonic, never reset
            fft_config_hash=self.config_hash,
            rgba_bytes=rgba.tobytes(),
            psd_db=psd_db,
            # ... metadata
        )
```

**Files:**
- `backend/spectrogram/config.py` - Configuration with hash validation
- `backend/spectrogram/frame.py` - SpectrogramFrame dataclass, FrameIDGenerator
- `backend/spectrogram/generator.py` - FFT, PSD, colormap rendering

---

### 2. RF Signal Tracker (10 Tests)

**Purpose:** Stable track lifecycle with no flicker or jitter.

**Key Invariants:**
- UI renders **tracks**, never raw detections
- Tracks have states: `tentative` → `confirmed` → `lost`
- EMA smoothing for display boxes

```python
# backend/tracker/tracker.py
class RFSignalTracker:
    def __init__(
        self,
        iou_threshold: float = 0.2,
        freq_tolerance_normalized: float = 0.02,
        confirm_hits: int = 3,
        max_age: int = 10,
        box_ema_alpha: float = 0.3,
    ):
        self._tracks: Dict[int, TrackState] = {}
        self._next_id = 0
    
    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        """Main entry point - call once per inference frame."""
        det_used = [False] * len(detections)
        
        # 1. Match existing tracks to detections
        for track in self._tracks.values():
            best_idx = self._find_best_match(track, detections, det_used)
            if best_idx is not None:
                det_used[best_idx] = True
                self._update_matched_track(track, detections[best_idx])
            else:
                self._age_unmatched_track(track)
        
        # 2. Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if not det_used[i]:
                self._create_track(det, frame_id)
        
        # 3. Prune lost tracks
        self._prune_lost_tracks()
        
        return self._export_tracks(frame_id)
```

**Track State Machine:**
```
            +3 hits          max_age frames
[tentative] ────────► [confirmed] ────────► [lost] ────► (removed)
     │                      │                  │
     │    no match          │    no match      │
     └──────────────────────┴──────────────────┘
              (age increases, confidence decays)
```

**EMA Smoothing (Anti-Jitter):**
```python
def _update_matched_track(self, track: TrackState, det: Detection):
    new_box = np.array(det.box)
    
    # Truth box: fast response
    track.truth_box = 0.3 * new_box + 0.7 * track.truth_box
    
    # Display box: extra smoothing for UI
    track.display_box = 0.5 * track.truth_box + 0.5 * track.display_box
    
    # Confidence: fast rise, slow fall
    if det.confidence > track.confidence:
        track.confidence = 0.3 * det.confidence + 0.7 * track.confidence
    else:
        track.confidence = 0.05 * det.confidence + 0.95 * track.confidence
```

**Files:**
- `backend/tracker/track.py` - Track, TrackState, Detection dataclasses
- `backend/tracker/association.py` - IoU calculation, matching
- `backend/tracker/tracker.py` - RFSignalTracker

---

### 3. Frame-ID Synchronization (9 Tests)

**Purpose:** Correctly map tracks to display rows even with dropped frames.

**Critical Bug Fix:** Use **deque position**, not arithmetic difference!

```python
# backend/transport/frame_buffer.py
class FrameBuffer:
    """
    CRITICAL: frame_id_to_row uses DEQUE POSITION, not arithmetic.
    This handles dropped frames correctly.
    """
    
    def __init__(self, max_frames: int = 300, display_rows: int = 256):
        self._frames: collections.deque = collections.deque(maxlen=max_frames)
        self._frame_id_to_index: Dict[int, int] = {}
    
    def frame_id_to_row(self, frame_id: int) -> Optional[int]:
        """
        Convert frame_id to display row using deque position.
        
        Returns None if frame is not in buffer or too old for display.
        """
        if frame_id not in self._frame_id_to_index:
            return None
        
        idx = self._frame_id_to_index[frame_id]
        frames_from_newest = len(self._frames) - 1 - idx
        
        if frames_from_newest >= self.display_rows:
            return None  # Too old
        
        return frames_from_newest  # Row 0 = newest (top)
```

**Why This Matters:**
```
With dropped frames (IDs: 100, 101, 105, 106):
  - WRONG (arithmetic): frame_id_to_row(101) = 106 - 101 = 5 ❌
  - RIGHT (deque position): frame_id_to_row(101) = position 1 from end = 2 ✓
```

**Files:**
- `backend/transport/frame_buffer.py` - FrameBuffer with correct sync
- `backend/transport/protocol.py` - Binary WebSocket protocol

---

### 4. Flutter Overlay Rendering

**Purpose:** Minimal-occlusion track overlays using corner markers.

```dart
// lib/features/live_detection/widgets/track_overlay.dart
class TrackOverlayPainter extends CustomPainter {
  final List<Track> tracks;
  final Map<int, int> frameIdToRow;  // CRITICAL: from deque position
  
  void _paintTrack(Canvas canvas, Size size, Track track) {
    // Map frame_id to row using buffer position
    final row = frameIdToRow[track.lastSeenFrameId];
    if (row == null) return;
    
    // Calculate pixel coordinates
    final x1 = track.x1 * size.width;
    final x2 = track.x2 * size.width;
    final y1 = row * (size.height / displayRows);
    
    final rect = Rect.fromLTRB(x1, y1, x2, y2);
    
    if (isSelected) {
      // Selected: full rectangle + subtle fill
      canvas.drawRect(rect, strokePaint);
      canvas.drawRect(rect, fillPaint..color = color.withOpacity(0.1));
    } else {
      // Default: corner markers only (minimal occlusion)
      _paintCornerMarkers(canvas, rect, strokePaint);
    }
  }
  
  void _paintCornerMarkers(Canvas canvas, Rect rect, Paint paint) {
    final len = min(cornerLength, rect.width / 3);
    final path = Path()
      // Top-left
      ..moveTo(rect.left, rect.top + len)
      ..lineTo(rect.left, rect.top)
      ..lineTo(rect.left + len, rect.top)
      // ... other corners
    ;
    canvas.drawPath(path, paint);
  }
}
```

**Files:**
- `lib/features/live_detection/models/track.dart` - Track model
- `lib/features/live_detection/widgets/track_overlay.dart` - CustomPainter

---

### 5. TensorRT Deployment

**Purpose:** Build optimized inference engines ON THE TARGET SYSTEM.

**Critical Rule:** Never copy .engine files between machines!

```python
# scripts/build_tensorrt_engine.py
def build_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    """Build TensorRT engine from ONNX - RUN ON TARGET SYSTEM."""
    import tensorrt as trt
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    # Enable FP16 if available
    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
```

**Usage:**
```bash
# On Orin NX target system:
python scripts/build_tensorrt_engine.py \
    --onnx models/yolov8s_rf.onnx \
    --output models/yolov8s_rf.engine \
    --fp16
```

**Files:**
- `backend/inference/engine.py` - Unified TensorRT/ONNX inference
- `scripts/build_tensorrt_engine.py` - Engine builder (run on target)

---

### 6. Async Database Logging (7 Tests)

**Purpose:** Log detections without blocking the real-time pipeline.

**Key Invariant:** `log_track()` never blocks - drops if queue is full.

```python
# backend/database/async_writer.py
class AsyncDatabaseWriter:
    def __init__(self, db_path: str, max_queue_size: int = 10000):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
    
    def log_track(self, track, frame_id: int, timestamp_ns: int) -> bool:
        """
        INVARIANT: Never blocks. Returns False if queue is full.
        """
        try:
            self._queue.put_nowait(log_entry)
            return True
        except queue.Full:
            self._logs_dropped += 1
            return False  # Drop rather than block
    
    def _writer_loop(self):
        """Background thread - batch inserts."""
        batch = []
        while self._running:
            # Collect batch
            while len(batch) < self.batch_size:
                try:
                    item = self._queue.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    break
            
            # Write batch
            if batch:
                self._write_batch(batch)
                batch = []
```

**Performance:** 1000 log attempts in 6ms, even with full queue (drops gracefully).

**Design Decision: SQLite vs Postgres**
> The original roadmap text mentioned Postgres/asyncpg, but v1 uses SQLite.
> This is intentional for the following reasons:
> - **Simpler deployment**: No separate DB server needed on Orin NX
> - **Sufficient for v1**: Single-writer scenario with async queue
> - **Same invariant**: Non-blocking, drops gracefully under backpressure
> 
> For v2/production, Postgres can be swapped in by implementing the same
> interface with asyncpg. The async queue pattern remains the same.

**Files:**
- `backend/database/schema.py` - SQLite schema, session management
- `backend/database/async_writer.py` - Non-blocking writer

---

### 7. Stub IQ Source (For Testing)

**Purpose:** Test the full pipeline without hardware.

```python
# backend/sources/stub_source.py
class StubIQSource(IQSource):
    """Generates synthetic IQ with configurable signals."""
    
    def __init__(self, signals: List[SimulatedSignal], ...):
        self._signals = signals
    
    def _generate_samples(self) -> np.ndarray:
        # Noise floor
        samples = self._noise_amplitude * (
            np.random.randn(self._block_size) + 
            1j * np.random.randn(self._block_size)
        )
        
        # Add signals
        for sig in self._signals:
            freq = sig.freq_offset_hz + sig.drift_hz_per_sec * elapsed
            signal = 10**(sig.power_db/20) * np.exp(2j * np.pi * freq * t)
            samples += signal
        
        return samples.astype(np.complex64)

def create_demo_source() -> StubIQSource:
    return StubIQSource(signals=[
        SimulatedSignal(freq_offset_hz=2e6, power_db=-15),
        SimulatedSignal(freq_offset_hz=-3e6, power_db=-20, drift_hz_per_sec=5000),
    ])
```

**Files:**
- `backend/sources/base.py` - IQSource abstract base class
- `backend/sources/stub_source.py` - Synthetic signal generator

---

## Test Suite

**Run all 39 tests:**
```bash
python tests/run_all_tests.py
```

**Output:**
```
============================================================
  G20 RF WATERFALL DETECTION SYSTEM - TEST SUITE
============================================================

  CANONICAL SPECTROGRAM PIPELINE: 13/13 passed ✓
  RF SIGNAL TRACKER: 10/10 passed ✓
  FRAME-ID SYNCHRONIZATION: 9/9 passed ✓
  ASYNC DATABASE LOGGING: 7/7 passed ✓

  Total: 39/39 tests passed
  Time:  1172.5ms

============================================================
  🎉 ALL TESTS PASSED!
============================================================
```

---

## Demo Mode

**Run without hardware:**
```bash
cd g20_demo
pip install websockets numpy
python scripts/run_demo.py --duration 60 --database demo.db
```

**Output:**
```
============================================================
  G20 RF WATERFALL DETECTION SYSTEM - DEMO MODE
============================================================
  Duration:    60 seconds
  WebSocket:   ws://localhost:8765
  Database:    demo.db
  Signals:     4 simulated
============================================================

  Running... (Press Ctrl+C to stop)

  [   5s] Frames:    150 | Tracks:   4 | Clients: 0 | Frame:  33.2ms
  [  10s] Frames:    300 | Tracks:   4 | Clients: 0 | Frame:  33.1ms
  ...
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/spectrogram.yaml` | FFT size, window, overlap, colormap |
| `config/tracker.yaml` | IoU threshold, confirm_hits, EMA alphas |
| `config/inference.yaml` | Model paths, class names, thresholds |
| `config/database.yaml` | DB path, queue size, retention |

---

## What Requires Hardware

| Component | Status | Notes |
|-----------|--------|-------|
| IQ Capture | **STUBBED** | Replace `StubIQSource` with `SidekiqSource` |
| TensorRT Engine | **SCRIPTS READY** | Run `build_tensorrt_engine.py` ON Orin NX |
| Real Inference | **INTERFACE READY** | Pass `inference_callback` to Pipeline |

---

## File Structure

```
g20_demo/
├── config/
│   ├── spectrogram.yaml
│   ├── tracker.yaml
│   ├── inference.yaml
│   └── database.yaml
│
├── backend/
│   ├── spectrogram/
│   │   ├── config.py       # SpectrogramConfig with hash
│   │   ├── frame.py        # SpectrogramFrame, FrameIDGenerator
│   │   └── generator.py    # CanonicalSpectrogramGenerator
│   │
│   ├── tracker/
│   │   ├── track.py        # Track, TrackState, Detection
│   │   ├── association.py  # IoU, matching
│   │   └── tracker.py      # RFSignalTracker
│   │
│   ├── transport/
│   │   ├── frame_buffer.py # FrameBuffer (deque-based sync)
│   │   ├── protocol.py     # Binary WebSocket protocol
│   │   └── websocket_server.py
│   │
│   ├── inference/
│   │   └── engine.py       # TensorRT/ONNX wrapper
│   │
│   ├── database/
│   │   ├── schema.py       # SQLite schema
│   │   └── async_writer.py # Non-blocking logger
│   │
│   ├── sources/
│   │   ├── base.py         # IQSource ABC
│   │   └── stub_source.py  # Synthetic signals
│   │
│   └── pipeline.py         # Main orchestration
│
├── lib/features/live_detection/
│   ├── models/track.dart
│   └── widgets/track_overlay.dart
│
├── scripts/
│   ├── run_demo.py
│   └── build_tensorrt_engine.py
│
├── tests/
│   ├── run_all_tests.py    # Unified test runner
│   ├── test_spectrogram.py
│   ├── test_tracker.py
│   ├── test_frame_buffer.py
│   └── test_database.py
│
└── docs/
    ├── Main_roadmap.txt
    └── IMPLEMENTATION_REPORT.md  # This file
```

---

## Next Steps (Hardware Integration)

1. **Replace StubIQSource with SidekiqSource**
   - Implement `IQSource` interface using libsidekiq
   - Handle GPU DMA ring buffer

2. **Build TensorRT Engine on Orin NX**
   ```bash
   scp models/yolov8s_rf.onnx orin:~/
   ssh orin
   python build_tensorrt_engine.py --onnx yolov8s_rf.onnx --output yolov8s_rf.engine --fp16
   ```

3. **Connect Real Inference**
   ```python
   from inference.engine import InferenceEngine
   
   engine = InferenceEngine.from_tensorrt('yolov8s_rf.engine')
   
   pipeline = Pipeline(
       iq_source=SidekiqSource(...),
       inference_callback=lambda frame: engine.run(frame).detections,
       ...
   )
   ```

4. **Flutter Integration**
   - Update providers to connect to WebSocket
   - Integrate TrackOverlay into WaterfallDisplay

---

**Report prepared by:** AI Implementation Assistant  
**Review by:** Senior SWE  
**Date:** January 23, 2026
