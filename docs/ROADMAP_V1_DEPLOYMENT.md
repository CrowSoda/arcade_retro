# G20/NV100 Deployment & Testing Guide
## ROADMAP_V1_DEPLOYMENT.md

---

## Overview

This document contains deployment procedures, testing checklists, and operational guidance for the G20 RF Waterfall Detection System on NVIDIA Orin NX.

**Related Documents:**
- [ROADMAP_V1_OVERVIEW.md](ROADMAP_V1_OVERVIEW.md) - System architecture and overview
- [ROADMAP_V1_BACKEND.md](ROADMAP_V1_BACKEND.md) - Backend implementation
- [ROADMAP_V1_FLUTTER.md](ROADMAP_V1_FLUTTER.md) - Flutter/UI implementation

---

## Target Platform

| Component | Specification |
|-----------|---------------|
| Platform | Epic G20 |
| SDR | NV100 (libsidekiq, 20 MSps) |
| Compute | NVIDIA Orin NX |
| JetPack | 5.x (document exact version) |
| CUDA | 11.4+ |
| TensorRT | 8.5+ |

---

## Development Environment Setup

### Step 1: Create Directory Structure

```bash
# Clone repo structure
mkdir -p g20_waterfall/{config,backend/{spectrogram,inference,tracker,transport,database},flutter_app,scripts,tests}
```

### Step 2: Backend Dependencies

```bash
cd g20_waterfall
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install numpy cupy-cuda12x asyncpg websockets pyyaml torch torchvision
```

### Step 3: Flutter Setup

```bash
# Ensure Flutter is installed
flutter --version

# Navigate to flutter app
cd flutter_app

# Get dependencies
flutter pub get

# Verify build
flutter build linux  # or windows/macos
```

---

## TensorRT Engine Build

**CRITICAL:** TensorRT engines are NOT portable. They must be built ON the target device.

### Step 1: Export Model to ONNX (Development Machine)

```bash
pip install ultralytics
yolo export model=yolov8s.pt format=onnx imgsz=640 opset=17 simplify=True
```

### Step 2: Transfer to Target Device

```bash
scp yolov8s.onnx user@orin-nx:/path/to/models/
```

### Step 3: Build Engine on Target (Orin NX)

```bash
# SSH to Orin NX
ssh user@orin-nx

# FP16 Engine (recommended first)
/usr/src/tensorrt/bin/trtexec \
    --onnx=/path/to/models/yolov8s.onnx \
    --saveEngine=/path/to/models/yolov8s_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --verbose

# INT8 Engine (requires calibration data)
/usr/src/tensorrt/bin/trtexec \
    --onnx=/path/to/models/yolov8s.onnx \
    --saveEngine=/path/to/models/yolov8s_int8.engine \
    --int8 \
    --calib=/path/to/calibration_cache.bin \
    --workspace=4096 \
    --verbose
```

### Step 4: Verify Engine

```bash
# Test engine loads and runs
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=/path/to/models/yolov8s_fp16.engine \
    --iterations=100 \
    --warmUp=10
```

Expected output:
- FP16 latency: ≤20ms
- INT8 latency: ≤15ms
- No errors or warnings

---

## Calibration Data Collection

For INT8 quantization, collect representative data from the deployment environment:

```python
# Run on target device with live RF data
python scripts/collect_calibration.py \
    --output-dir /path/to/calibration_data \
    --count 500 \
    --source live
```

Requirements:
- Minimum 500 frames
- Representative of actual deployment RF environment
- Include various signal types and SNR levels

---

## Database Setup

### PostgreSQL Installation

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE g20_tracks;
CREATE USER g20_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE g20_tracks TO g20_user;
\q
```

### Schema Initialization

```bash
# Run schema creation
psql -U g20_user -d g20_tracks -f backend/database/schema.sql
```

### Connection Configuration

Update `config/database.yaml`:

```yaml
connection_string: postgresql://g20_user:secure_password@localhost/g20_tracks
batch_size: 50
flush_interval_ms: 500
max_queue_size: 1000
```

---

## Configuration Files

### config/spectrogram.yaml

```yaml
# Canonical spectrogram configuration - DO NOT MODIFY IN PRODUCTION
fft_size: 4096
window: hann
overlap: 0.5
psd_ref_dbm: -30.0
dynamic_range_db: 60.0
colormap: viridis
```

### config/tracker.yaml

```yaml
# Tracker parameters - may require tuning per deployment
association:
  iou_threshold: 0.2          # Lower = more lenient matching
  freq_tolerance_normalized: 0.02

lifecycle:
  confirm_hits: 3             # Frames to confirm track
  max_age: 10                 # Frames before track lost

smoothing:
  box_ema_alpha: 0.3          # Higher = more responsive
  display_ema_alpha: 0.5
  conf_rise_alpha: 0.3
  conf_fall_alpha: 0.05

drift_detection:
  enabled: false              # Enable for drifting signals
  frames_required: 5
  min_displacement: 0.005
```

### config/inference.yaml

```yaml
model_path: models/yolov8s_fp16.engine
input_size: [640, 640]
confidence_threshold: 0.25
nms_threshold: 0.45
```

---

## Starting the System

### 1. Start Backend

```bash
cd g20_waterfall
source venv/bin/activate

# Start the main pipeline
python backend/pipeline.py \
    --config config/ \
    --port 8765
```

### 2. Start Flutter App

```bash
cd flutter_app

# Development
flutter run -d linux

# Production build
flutter build linux --release
./build/linux/x64/release/bundle/g20_waterfall
```

### 3. Verify Connection

Check that:
- WebSocket connects (ws://localhost:8765)
- Waterfall frames streaming
- Track overlay rendering

---

## Testing Checklist

### Functional Tests

#### Phase 1: Canonical Spectrogram

- [ ] Single `SpectrogramConfig` instance used everywhere
- [ ] Config hash included in every frame
- [ ] Changing config triggers warning

#### Phase 2: Tracker

- [ ] No flicker: tracks stable frame-to-frame
- [ ] No jitter: display boxes smooth
- [ ] Confidence accumulation works
- [ ] Lifecycle: tentative → confirmed → lost
- [ ] (Optional) Drift detection works

#### Phase 3: Frame-ID Sync

- [ ] Track boxes scroll with waterfall exactly
- [ ] No drift between box and signal over time
- [ ] Boxes disappear cleanly when scrolling off

#### Phase 4: Overlay Rendering

- [ ] All tracks render in single paint call
- [ ] Corner markers only (no fill) by default
- [ ] Selected track shows full rectangle + subtle fill
- [ ] Opacity reflects confidence and age
- [ ] 50+ tracks render without frame drops

#### Phase 5: TensorRT

- [ ] Engine builds on Orin NX
- [ ] FP16 latency ≤20ms
- [ ] INT8 latency ≤15ms
- [ ] No accuracy regression (mAP within 1%)
- [ ] Engine loads in <5 seconds

#### Phase 6: Database

- [ ] Logging never blocks pipeline
- [ ] Queue overflow drops gracefully
- [ ] Batch inserts <10ms for 50 records
- [ ] All tracks have frame_id for replay

### Performance Tests

| Metric | Target | Test Method |
|--------|--------|-------------|
| End-to-end latency | ≤100ms | Timestamp injection |
| Display FPS | 30fps sustained | FPS counter |
| Inference latency | ≤15-20ms | TensorRT profiler |
| Track render (50+) | No frame drops | Stress test |
| Memory stability | No growth | 24hr run |

### Latency Budget

Fill in "Measured" column during deployment validation:

| Stage | Target | Measured |
|-------|--------|----------|
| IQ capture → GPU buffer | ≤5ms | ___ |
| FFT + PSD + normalize | ≤3ms | ___ |
| GPU→CPU frame copy | ≤2ms | ___ |
| Inference (YOLOv8s) | ≤15-20ms | ___ |
| Tracker update | ≤2ms | ___ |
| WebSocket serialize + send | ≤5ms | ___ |
| Flutter decode + render | ≤15ms | ___ |
| **End-to-end total** | **≤100ms** | ___ |

### Latency Measurement

Instrument these points:
- `t0`: IQ buffer timestamp (device clock)
- `t1`: Frame packaging complete
- `t2`: Inference complete
- `t3`: Tracker update complete
- `t4`: WebSocket send complete
- `t5`: Flutter frame rendered

```python
# Add to pipeline.py
import time

class LatencyTracker:
    def __init__(self):
        self.points = {}
    
    def mark(self, name: str):
        self.points[name] = time.perf_counter_ns()
    
    def report(self):
        if 't0' not in self.points:
            return
        
        t0 = self.points['t0']
        for name, ts in self.points.items():
            print(f"{name}: {(ts - t0) / 1e6:.2f}ms")
```

---

## Acceptance Criteria Summary

### Functional

- [ ] Boxes never drift from visible signals
- [ ] Tracks are stable (no flicker)
- [ ] Confidence displays smoothly
- [ ] Overlay never obscures signal texture
- [ ] Deterministic replay possible
- [ ] Selected track shows full rectangle with subtle fill

### Performance

- [ ] End-to-end latency ≤100ms
- [ ] Display sustains 30fps
- [ ] Inference ≤20ms (FP16) or ≤15ms (INT8)
- [ ] 50+ tracks render without drops
- [ ] Database never blocks pipeline

### Operational

- [ ] System runs 24+ hours without degradation
- [ ] Config changes don't require code changes
- [ ] Logs include frame_id for debugging
- [ ] Latency metrics are observable

---

## Troubleshooting

### Common Issues

#### TensorRT Engine Won't Load

**Symptom:** `Error deserializing CUDA engine`

**Cause:** Engine built on different GPU/JetPack version

**Fix:** Rebuild engine ON target device with matching CUDA/TensorRT versions

#### Detection Boxes Drift from Signals

**Symptom:** Boxes don't align with visible signals

**Cause:** Re-FFT in inference pipeline or PTS-based sync

**Fix:** 
1. Verify inference uses spatial transform only (no re-FFT)
2. Verify frame_id based sync (not PTS)

#### Track Flicker

**Symptom:** Tracks appear/disappear rapidly

**Cause:** No tracker, raw detections displayed

**Fix:** Implement tracker with lifecycle states

#### Database Blocking Pipeline

**Symptom:** Frame drops during logging

**Cause:** Synchronous database writes

**Fix:** Use async logger with queue

#### Memory Growth Over Time

**Symptom:** Crash after hours of running

**Cause:** Frame buffer not pruning, image objects not disposed

**Fix:** 
1. Verify FrameBuffer.max_frames limit
2. Dispose Flutter Image objects
3. Profile with valgrind/heaptrack

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Flutter GPU→CPU copy bottleneck | Profile early; plan v2 external textures |
| TensorRT engine not portable | Always build ON target; document JetPack version |
| INT8 calibration not representative | Collect from actual deployment environment |
| Tracker params need tuning | Expose in config; provide tuning guide |
| Database bottleneck | Async queue with drop policy |
| Memory leak | Profile with valgrind; explicit buffer cycling |

---

## Monitoring & Logging

### Metrics to Collect

```python
# metrics.py
from dataclasses import dataclass

@dataclass
class PipelineMetrics:
    frames_processed: int = 0
    inference_latency_avg_ms: float = 0
    tracker_latency_avg_ms: float = 0
    websocket_latency_avg_ms: float = 0
    tracks_active: int = 0
    database_queue_depth: int = 0
    dropped_frames: int = 0
```

### Logging Format

```python
import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] frame_id=%(frame_id)s %(message)s',
    level=logging.INFO
)

# Usage
logger.info("Inference complete", extra={'frame_id': frame.frame_id})
```

---

## Implementation Order

1. `config/spectrogram.yaml` + `backend/spectrogram/` — Canonical spectrogram first
2. `backend/tracker/` — Tracker with tests
3. `backend/transport/frame_buffer.py` — Frame-ID indexing
4. `flutter_app/lib/widgets/track_overlay.dart` — Single-pass rendering
5. `backend/inference/` — TensorRT integration
6. `backend/database/` — Async logging

**Principle:** Validate each phase before moving on. Don't optimize prematurely.

---

## Critical Invariant Tests

### Canonical Spectrogram Invariant Test

```python
# tests/test_canonical_invariant.py
def test_inference_uses_canonical_source():
    """
    CRITICAL: Verify inference input is derived from canonical spectrogram 
    via spatial transform, NOT via separate FFT computation.
    
    Violation of this invariant causes detection box drift.
    """
    frame = generate_test_frame()
    
    # Get inference input
    inference_input = inference_pipeline.prepare_input(frame)
    
    # Verify dimensions
    assert inference_input.shape == (640, 640)
    
    # The key test: inference input must correlate highly with 
    # spatially downsampled canonical spectrogram
    canonical_downsampled = cv2.resize(
        frame.psd_db, 
        (640, 640), 
        interpolation=cv2.INTER_AREA
    )
    
    correlation = np.corrcoef(
        inference_input.flatten(), 
        canonical_downsampled.flatten()
    )[0, 1]
    
    assert correlation > 0.99, \
        f"Inference input diverged from canonical spectrogram (r={correlation:.3f}). " \
        "Check for re-FFT in inference pipeline."
```

### Frame-ID Monotonicity Test

```python
# tests/test_frame_id.py
def test_frame_id_strictly_monotonic():
    """Frame IDs must be strictly increasing, never reset or skip unexpectedly."""
    generator = CanonicalSpectrogramGenerator(config)
    
    frame_ids = []
    for _ in range(1000):
        frame = generator.process_iq(get_test_samples(), time.time_ns())
        frame_ids.append(frame.frame_id)
    
    for i in range(1, len(frame_ids)):
        assert frame_ids[i] == frame_ids[i-1] + 1, \
            f"Frame ID not strictly monotonic: {frame_ids[i-1]} -> {frame_ids[i]}"
```

### Frame Buffer Dropped Frames Test

```python
# tests/test_frame_buffer.py
def test_frame_id_to_row_with_dropped_frames():
    """Row mapping must work correctly when frames are dropped."""
    buffer = FrameBuffer(max_frames=100)
    
    # Simulate frames with gaps (dropped frames)
    frame_ids = [1, 2, 3, 5, 6, 8, 9, 10]  # 4 and 7 dropped
    for fid in frame_ids:
        buffer.add_frame(SpectrogramFrame(frame_id=fid))
    
    # Row should be position in buffer, not ID difference
    assert buffer.frame_id_to_row(10, display_rows=8) == 0  # newest
    assert buffer.frame_id_to_row(9, display_rows=8) == 1
    assert buffer.frame_id_to_row(8, display_rows=8) == 2
    assert buffer.frame_id_to_row(6, display_rows=8) == 3   # NOT 4 (10-6)
    assert buffer.frame_id_to_row(5, display_rows=8) == 4   # NOT 5 (10-5)
    assert buffer.frame_id_to_row(3, display_rows=8) == 5
    
    # Frame 4 was dropped - should return None, not a wrong row
    assert buffer.frame_id_to_row(4, display_rows=8) is None
    
    # Frame 7 was dropped - should return None
    assert buffer.frame_id_to_row(7, display_rows=8) is None
```

---

## Config Mismatch Handling

Add to `backend/pipeline.py`:

```python
class ConfigMismatchError(Exception):
    """Raised when frame config doesn't match pipeline config."""
    pass

class Pipeline:
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        self.config_hash = config.hash()
    
    def process_frame(self, frame: SpectrogramFrame):
        if frame.fft_config_hash != self.config_hash:
            raise ConfigMismatchError(
                f"Frame config hash {frame.fft_config_hash} doesn't match "
                f"pipeline config hash {self.config_hash}. "
                "This violates the canonical spectrogram invariant. "
                "All components must use identical FFT configuration."
            )
        # ... rest of processing
```

---

## 24-Hour Stability Test Script

Create `scripts/stability_test.sh`:

```bash
#!/bin/bash
set -e

echo "=== 24-Hour Stability Test ==="
echo "Starting at $(date)"

# Clean state
sudo systemctl restart postgresql
pkill -f "python.*pipeline.py" || true
sleep 2

# Start pipeline
python backend/pipeline.py --config config/ &
PID=$!
sleep 5

# Verify started
if ! ps -p $PID > /dev/null; then
    echo "FAIL: Pipeline failed to start"
    exit 1
fi

# Record baseline
BASELINE_MEM=$(ps -p $PID -o rss= | tr -d ' ')
echo "Baseline memory: ${BASELINE_MEM}KB"

# Monitor for 24 hours
for hour in {1..24}; do
    sleep 3600
    
    if ! ps -p $PID > /dev/null; then
        echo "FAIL: Pipeline crashed at hour $hour"
        exit 1
    fi
    
    CURRENT_MEM=$(ps -p $PID -o rss= | tr -d ' ')
    GROWTH=$((CURRENT_MEM - BASELINE_MEM))
    GROWTH_PCT=$((GROWTH * 100 / BASELINE_MEM))
    
    echo "Hour $hour: ${CURRENT_MEM}KB (growth: ${GROWTH}KB / ${GROWTH_PCT}%)"
    
    # Fail if >20% growth
    if [ $GROWTH_PCT -gt 20 ]; then
        echo "FAIL: Memory growth exceeded 20% at hour $hour"
        kill $PID
        exit 1
    fi
done

echo "PASS: 24-hour stability test complete"
kill $PID
```

---

## INT8 Calibration Data Requirements

Collect calibration data that accurately represents the deployment RF environment:

| Requirement | Specification |
|-------------|---------------|
| **Minimum frames** | 500 |
| **SNR distribution** | 20% low (<10dB), 60% medium (10-20dB), 20% high (>20dB) |
| **Signal types** | All classes the model will detect in production |
| **Overlapping signals** | At least 10% of frames with 2+ overlapping signals |
| **Empty frames** | 10% frames with noise floor only (no signals) |
| **Time diversity** | Collect across different times of day/operational periods |

**Validation:** After INT8 engine build, verify mAP is within 1% of FP16 baseline on a held-out test set.

---

## Verification Checklist

After implementing all changes, verify:

- [ ] `test_frame_id_to_row_with_dropped_frames` passes
- [ ] `test_inference_uses_canonical_source` passes
- [ ] `test_frame_id_strictly_monotonic` passes
- [ ] `ConfigMismatchError` is raised when configs don't match
- [ ] Flutter `ui.Image` disposal is implemented
- [ ] Latency budget table is filled during deployment validation
- [ ] Calibration requirements table criteria are met
- [ ] `stability_test.sh` script exists and is executable

---

## Summary of Technical Review Changes

| Document | Change |
|----------|--------|
| ROADMAP_V1_BACKEND.md | Frame buffer code updated (references Flutter doc) |
| ROADMAP_V1_DEPLOYMENT.md | Added latency budget, invariant tests, stability script, calibration requirements |
| ROADMAP_V1_FLUTTER.md | Fixed frame_id→row to use deque position; added ui.Image disposal |
| tests/ | Added test_frame_buffer.py, test_canonical_invariant.py, test_frame_id.py |
| scripts/ | Added stability_test.sh |

**The architecture is approved. These are the final additions needed before implementation begins.**
