# G20/NV100 Real-Time RF Waterfall Detection System
## Implementation Roadmap v1.0

---

## Executive Summary

Build a real-time RF signal detection and visualization system for the Epic G20 platform with NV100 SDR running on NVIDIA Orin NX. The system detects signals on a live waterfall display and overlays tracked detections without obscuring the underlying signal.

**Core Principle:** State of the art is defined by architectural invariants, not model choice. Get the invariants right first; optimize later.

**End-to-end latency target:** ≤100ms from IQ capture to screen.

---

## System Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NV100 SDR                                      │
│                         (libsidekiq, 20 MSps)                               │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ IQ samples (GPU DMA)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CANONICAL SPECTROGRAM GENERATOR (GPU)                    │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────────┐  │
│  │ Ring Buffer │ → │ FFT (cuFFT) │ → │ PSD (dB)    │ → │ Normalize/Scale│  │
│  │ (IQ)        │   │ 4096-point  │   │ 20log10     │   │ to 0-255       │  │
│  └─────────────┘   └─────────────┘   └─────────────┘   └───────┬────────┘  │
│                                                                 │           │
│                              SpectrogramFrame                   │           │
│                              {frame_id, gpu_buffer, metadata}   │           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┬─────────────────┐
                ▼                 ▼                 ▼                 ▼
        ┌───────────────┐ ┌─────────────┐ ┌───────────────┐ ┌───────────────┐
        │ Display Path  │ │ Inference   │ │ Recording     │ │ Database      │
        │ (30 fps)      │ │ (10-15 fps) │ │ (async)       │ │ (async batch) │
        └───────┬───────┘ └──────┬──────┘ └───────────────┘ └───────────────┘
                │                │
                │                ▼
                │         ┌─────────────┐
                │         │ YOLOv8s     │
                │         │ TensorRT    │
                │         └──────┬──────┘
                │                │ Detections
                │                ▼
                │         ┌─────────────┐
                │         │ Tracker     │
                │         │ (IoU + EMA) │
                │         └──────┬──────┘
                │                │ Tracks
                │                ▼
                ▼         ┌─────────────┐
        ┌───────────────┐ │ Track State │
        │ Flutter UI    │◄┤ Publisher   │
        │               │ └─────────────┘
        │ ┌───────────┐ │
        │ │ Waterfall │ │
        │ │ Widget    │ │
        │ ├───────────┤ │
        │ │ Track     │ │
        │ │ Overlay   │ │
        │ └───────────┘ │
        └───────────────┘
```

### Component Boundaries

| Component | Runs On | Language | Responsibility |
|-----------|---------|----------|----------------|
| IQ Capture | GPU DMA | C/libsidekiq | Ring buffer fill |
| Spectrogram Gen | GPU | Python/CuPy | FFT, PSD, normalize, frame packaging |
| Inference | GPU | Python/TensorRT | Detection on pooled spectrogram |
| Tracker | CPU | Python | Association, smoothing, lifecycle |
| Transport | CPU | Python | ZMQ internal, WebSocket external |
| Flutter UI | CPU/GPU | Dart | Waterfall render, track overlay |
| Database | CPU | Python/asyncpg | Async batch logging |

### GPU/CPU Boundary

```
GPU-RESIDENT:
├── IQ ring buffer
├── FFT computation (cuFFT)
├── PSD calculation
├── Normalization
├── Inference tensor (pooled from canonical)
└── YOLOv8s TensorRT inference

CPU-BOUND (v1, acceptable):
├── Frame metadata assembly
├── Detection post-processing
├── Tracker state machine
├── WebSocket serialization
├── Flutter texture upload (GPU→CPU copy, ~1MB/frame)
└── Database batching

v2 OPTIMIZATION (later):
└── External texture path eliminates Flutter GPU→CPU copy
```

---

## Current State Assessment

### Existing Implementation (g20_demo)

The current codebase has a working foundation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| IQ Pipeline | ✅ Working | `backend/unified_pipeline.py` | 4096 FFT, RGBA pre-render |
| Waterfall Display | ✅ Working | `lib/features/live_detection/widgets/waterfall_display.dart` | Binary WebSocket, 30fps |
| Detection Boxes | ⚠️ Basic | `waterfall_display.dart` | PTS-based scrolling, no tracker |
| WebSocket Transport | ✅ Working | `backend/server.py` | Binary protocol for waterfall |
| Inference | ⚠️ PyTorch only | `backend/unified_pipeline.py` | FasterRCNN, ~15fps |
| Tracker | ❌ Missing | - | Raw detections render directly |
| Frame-ID Sync | ❌ Missing | - | Uses PTS (time-based), not frame_id |
| Database Logging | ❌ Missing | - | No persistence |
| TensorRT | ❌ Missing | - | PyTorch only |

### Key Gaps to Address

1. **No Tracker** - Raw detections cause flicker/jitter
2. **Time-based sync** - PTS scrolling causes drift
3. **Fill-based boxes** - Obscures signal texture
4. **No frame_id identity** - Can't replay deterministically
5. **No database logging** - No persistence for analysis

---

## Architectural Invariants (Non-Negotiable)

These are hard constraints. Every implementation decision must preserve them.

| Invariant | Rule | Violation Symptom |
|-----------|------|-------------------|
| **One canonical spectrogram** | Single FFT size, window, overlap, PSD normalization. Inference sees spatial transform of display data, never re-FFT. | Box drift, "box doesn't match signal" complaints |
| **Frame-ID is identity** | Monotonic integer, never reset. All detections/tracks reference frame_id, not wall-clock time. | Non-deterministic replay, sync bugs |
| **Tracking-by-detection** | UI renders tracks, never raw detections. Tracks have lifecycle states. | Flicker, jitter, operator distrust |
| **Minimal-occlusion overlay** | Corner markers, 0% fill default. Attributes encoded via stroke/saturation/dash. | Obscured signals, operator complaints |
| **Deterministic latency** | Inference cadence decoupled from display cadence. Bounded jitter over max throughput. | Unpredictable UX, debugging nightmare |

---

## Data Contracts

### SpectrogramFrame

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SpectrogramFrame:
    # Identity
    frame_id: int                    # Monotonic, never reset, primary key
    
    # Timing (metadata only, not for sync)
    timestamp_ns: int                # Device monotonic clock
    pts: float                       # Presentation timestamp (legacy compat)
    
    # FFT configuration (hash for change detection)
    fft_config_hash: int             # Hash of (fft_size, window, overlap, normalization)
    
    # Frequency span
    freq_start_hz: float
    freq_end_hz: float
    sample_rate_hz: float
    
    # Data reference (not the data itself)
    gpu_buffer_id: int               # For zero-copy paths
    width: int                       # FFT bins (e.g., 4096)
    height: int                      # Rows in this frame (typically 1)
    
    # Optional CPU copy (v1 only)
    rgba_bytes: Optional[bytes] = None
    psd_db: Optional[np.ndarray] = None
```

### Detection (raw, internal only)

```python
@dataclass
class Detection:
    frame_id: int                    # Source frame reference
    class_id: int
    class_name: str
    confidence: float                # 0.0 - 1.0
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized 0-1
    
    # Derived RF parameters
    freq_center_hz: float
    freq_bandwidth_hz: float
```

### Track (rendered to UI)

```python
@dataclass
class Track:
    # Identity
    track_id: int                    # Unique across session
    
    # Classification
    class_id: int
    class_name: str
    
    # Position (in canonical spectrogram space, normalized 0-1)
    truth_box: tuple[float, float, float, float]   # Actual tracked position
    display_box: tuple[float, float, float, float] # Smoothed for rendering
    
    # Confidence
    confidence: float                # Accumulated, not per-frame
    
    # Lifecycle
    state: str                       # 'tentative' | 'confirmed' | 'lost'
    age_frames: int                  # Frames since last detection match
    hits: int                        # Total matched detections
    
    # Motion (optional, for drift handling)
    motion_mode: str                 # 'stationary' | 'drifting'
    drift_rate_hz_per_sec: float     # Estimated drift rate
    drift_confidence: float          # Confidence in drift estimate
    
    # RF parameters
    freq_center_hz: float
    freq_bandwidth_hz: float
    
    # Timestamps (metadata)
    first_seen_frame_id: int
    last_seen_frame_id: int
```

### TrackUpdate (sent to UI)

```python
@dataclass
class TrackUpdate:
    frame_id: int                    # Reference to source spectrogram frame
    timestamp_ns: int                # For logging
    tracks: list[Track]              # All active tracks
    
    # Statistics
    total_detections: int            # Raw detections this frame
    inference_latency_ms: float
    tracker_latency_ms: float
```

### Wire Protocol (Binary WebSocket)

```
Message Header (8 bytes):
├── type: uint8          # 0x01=Frame, 0x02=TrackUpdate, 0x03=Status
├── flags: uint8         # Reserved
├── length: uint32       # Payload length
└── frame_id: uint32     # Reference frame

TrackUpdate Payload:
├── track_count: uint16
├── inference_latency_ms: float32
├── tracker_latency_ms: float32
└── tracks[]: 
    ├── track_id: uint32
    ├── class_id: uint16
    ├── confidence: float32
    ├── state: uint8 (0=tentative, 1=confirmed, 2=lost)
    ├── display_box: float32[4] (x1, y1, x2, y2)
    ├── freq_center_hz: float64
    ├── freq_bandwidth_hz: float64
    └── motion_mode: uint8 (0=stationary, 1=drifting)
```

---

## Implementation Phases

### Phase 1: Canonical Spectrogram Pipeline (Week 1-2)
**Goal:** Single source of truth for all downstream consumers.

- [ ] Unify FFT configuration (`SpectrogramConfig`)
- [ ] Implement `CanonicalSpectrogramGenerator` with frame_id
- [ ] Update inference to derive from canonical frame (spatial transform only)
- [ ] Add config hash validation

**See:** [ROADMAP_V1_BACKEND.md](ROADMAP_V1_BACKEND.md) - Phase 1

---

### Phase 2: Tracker Implementation (Week 2-3)
**Goal:** Stable tracks with no flicker, optional drift handling.

- [ ] Implement `RFSignalTracker` with IoU + frequency association
- [ ] Add EMA smoothing for display boxes
- [ ] Implement confidence accumulation (asymmetric rise/fall)
- [ ] Add lifecycle states: tentative → confirmed → lost
- [ ] Optional: drift detection for drifting signals

**See:** [ROADMAP_V1_BACKEND.md](ROADMAP_V1_BACKEND.md) - Phase 2

---

### Phase 3: Frame-ID Synchronization (Week 3)
**Goal:** Replace time-based scrolling with frame-indexed mapping.

- [ ] Implement `FrameBuffer` manager (frame_id indexed)
- [ ] Update WebSocket protocol with frame_id
- [ ] Update Flutter waterfall to use frame_id → row mapping
- [ ] Verify box-to-signal alignment with scroll

**See:** [ROADMAP_V1_FLUTTER.md](ROADMAP_V1_FLUTTER.md) - Phase 3

---

### Phase 4: Overlay Rendering (Week 3-4)
**Goal:** Single-pass rendering with minimal occlusion.

- [ ] Implement `TrackOverlayPainter` with corner markers
- [ ] Add class-based coloring
- [ ] Add state-based stroke width
- [ ] Add opacity from confidence + age
- [ ] Selected track: full rectangle + subtle fill

**See:** [ROADMAP_V1_FLUTTER.md](ROADMAP_V1_FLUTTER.md) - Phase 4

---

### Phase 5: TensorRT Deployment (Week 4)
**Goal:** Optimized inference with bounded latency.

- [ ] Export YOLOv8s to ONNX
- [ ] Build TensorRT engine ON TARGET DEVICE
- [ ] Implement calibration data collection for INT8
- [ ] Implement `TensorRTInference` wrapper
- [ ] Validate accuracy vs PyTorch baseline

**See:** [ROADMAP_V1_BACKEND.md](ROADMAP_V1_BACKEND.md) - Phase 5

---

### Phase 6: Async Database Logging (Week 4)
**Goal:** Never block real-time pipeline.

- [ ] Implement `DetectionLogger` with async batch inserts
- [ ] Add queue with graceful overflow handling
- [ ] Create schema with frame_id correlation
- [ ] Test under load (no blocking)

**See:** [ROADMAP_V1_BACKEND.md](ROADMAP_V1_BACKEND.md) - Phase 6

---

## Latency Budget

**Target:** ≤100ms end-to-end (IQ capture to pixels on screen)

| Stage | Target | Notes |
|-------|--------|-------|
| IQ capture → GPU buffer | ≤5ms | DMA, ring buffer |
| FFT + PSD + normalize | ≤3ms | cuFFT 4096-point |
| GPU→CPU frame copy | ≤2ms | 1MB RGBA (v1 only) |
| Inference (YOLOv8s INT8) | ≤15ms | At 10-15 FPS cadence |
| Detection post-process | ≤1ms | NMS already done in model |
| Tracker update | ≤2ms | Simple IoU + EMA |
| WebSocket serialize + send | ≤5ms | Binary protocol, local |
| Flutter decode + render | ≤15ms | decodeImageFromPixels + paint |
| Display vsync | ≤16ms | 60 FPS display |
| **Total** | **≤64ms typical** | Budget has 36ms headroom |

### Measurement Points (Instrument These)

- `t0`: IQ buffer timestamp (device clock)
- `t1`: Frame packaging complete
- `t2`: Inference complete
- `t3`: Tracker update complete
- `t4`: WebSocket send complete
- `t5`: Flutter frame rendered (if measurable)

---

## File Structure

```
g20_waterfall/
├── config/
│   ├── spectrogram.yaml       # FFT config (single source of truth)
│   ├── tracker.yaml           # Tracker parameters
│   ├── inference.yaml         # Model paths, thresholds
│   └── database.yaml          # Connection string, batch params
│
├── backend/
│   ├── spectrogram/
│   │   ├── generator.py       # CanonicalSpectrogramGenerator
│   │   ├── frame.py           # SpectrogramFrame dataclass
│   │   └── config.py          # SpectrogramConfig
│   │
│   ├── inference/
│   │   ├── tensorrt_engine.py # TensorRTInference
│   │   ├── preprocessing.py   # Spatial transform from canonical
│   │   ├── postprocessing.py  # Detection parsing
│   │   └── calibration.py     # INT8 calibration tools
│   │
│   ├── tracker/
│   │   ├── tracker.py         # RFSignalTracker
│   │   ├── track.py           # Track, TrackState dataclasses
│   │   └── association.py     # IoU + frequency matching
│   │
│   ├── transport/
│   │   ├── frame_buffer.py    # FrameBuffer (frame_id indexed)
│   │   ├── websocket.py       # Binary WebSocket server
│   │   └── protocol.py        # Message serialization
│   │
│   ├── database/
│   │   ├── logger.py          # DetectionLogger (async batch)
│   │   └── schema.sql         # Table definitions
│   │
│   └── pipeline.py            # Main orchestration
│
├── flutter_app/
│   ├── lib/
│   │   ├── providers/
│   │   │   ├── waterfall_provider.dart
│   │   │   ├── track_provider.dart
│   │   │   └── websocket_provider.dart
│   │   │
│   │   ├── widgets/
│   │   │   ├── waterfall_display.dart
│   │   │   ├── track_overlay.dart     # TrackOverlayPainter
│   │   │   └── signal_info_panel.dart
│   │   │
│   │   ├── models/
│   │   │   ├── track.dart
│   │   │   └── spectrogram_frame.dart
│   │   │
│   │   └── main.dart
│   │
│   └── pubspec.yaml
│
├── scripts/
│   ├── build_tensorrt_engine.sh   # Run ON target device
│   ├── collect_calibration.py
│   └── validate_model.py
│
├── tests/
│   ├── test_tracker.py
│   ├── test_frame_sync.py
│   └── test_overlay_render.dart
│
└── docs/
    ├── ROADMAP_V1_OVERVIEW.md     # This file
    ├── ROADMAP_V1_BACKEND.md      # Backend implementation details
    ├── ROADMAP_V1_FLUTTER.md      # Flutter implementation details
    └── ROADMAP_V1_DEPLOYMENT.md   # Deployment & testing guide
```

---

## Acceptance Criteria Summary

### Functional

- [ ] Boxes never drift from visible signals (canonical space mapping works)
- [ ] Tracks are stable (no flicker between frames)
- [ ] Confidence displays smoothly (accumulation works)
- [ ] Overlay never obscures signal texture (corner markers, 0% fill)
- [ ] Deterministic replay possible from logged frames + track metadata
- [ ] Selected track shows full rectangle with subtle fill

### Performance

- [ ] End-to-end latency ≤100ms
- [ ] Display sustains 30fps without drops
- [ ] Inference completes within 20ms (FP16) or 15ms (INT8)
- [ ] 50+ simultaneous tracks render without frame drops
- [ ] Database logging never blocks pipeline

### Operational

- [ ] System runs 24+ hours without degradation
- [ ] Config changes don't require code changes
- [ ] Logs include frame_id for debugging
- [ ] Latency metrics are observable

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Flutter GPU→CPU copy becomes bottleneck | Display frame drops | Medium | Profile early; v2 external textures if needed |
| TensorRT engine not portable | Deployment failure | High | Always build engine ON target device; document exact JetPack version |
| INT8 calibration data not representative | Accuracy regression | Medium | Collect calibration data from actual deployment environment; validate mAP before shipping |
| Tracker parameters need tuning per deployment | Poor UX | Medium | Expose key params (iou_thresh, freq_tol, confirm_hits) as config; provide tuning guide |
| Database becomes bottleneck under load | Logging gaps | Low | Async queue with drop policy; monitor queue depth |
| Memory leak in long-running sessions | Crash after hours | Medium | Profile with valgrind/heaptrack; implement explicit buffer cycling |

---

## Getting Started (For Developer)

### Step 1: Set Up Development Environment

```bash
# Clone repo structure
mkdir -p g20_waterfall/{config,backend/{spectrogram,inference,tracker,transport,database},flutter_app,scripts,tests}

# Backend dependencies
cd g20_waterfall
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install numpy cupy-cuda12x asyncpg websockets pyyaml torch torchvision
```

### Step 2: Implement in Order

1. `config/spectrogram.yaml` + `backend/spectrogram/` — Get canonical spectrogram working first
2. `backend/tracker/` — Implement tracker with tests
3. `backend/transport/frame_buffer.py` — Frame-ID indexing
4. `flutter_app/lib/widgets/track_overlay.dart` — Single-pass rendering
5. `backend/inference/` — TensorRT integration
6. `backend/database/` — Async logging

### Step 3: Validate Each Phase

After each phase, verify acceptance criteria before moving on. Don't optimize prematurely.

---

## Related Documents

- [ROADMAP_V1_BACKEND.md](ROADMAP_V1_BACKEND.md) - Detailed backend implementation guide
- [ROADMAP_V1_FLUTTER.md](ROADMAP_V1_FLUTTER.md) - Flutter/UI implementation guide
- [ROADMAP_V1_DEPLOYMENT.md](ROADMAP_V1_DEPLOYMENT.md) - Deployment and testing guide
