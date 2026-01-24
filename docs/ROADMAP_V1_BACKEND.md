# G20/NV100 Backend Implementation Guide
## ROADMAP_V1_BACKEND.md

---

## Overview

This document contains detailed implementation code and guidance for the Python backend components of the G20 RF Waterfall Detection System.

**Related Documents:**
- [ROADMAP_V1_OVERVIEW.md](ROADMAP_V1_OVERVIEW.md) - System architecture and overview
- [ROADMAP_V1_FLUTTER.md](ROADMAP_V1_FLUTTER.md) - Flutter/UI implementation
- [ROADMAP_V1_DEPLOYMENT.md](ROADMAP_V1_DEPLOYMENT.md) - Deployment guide

---

## Phase 1: Canonical Spectrogram Pipeline

**Goal:** Single source of truth for all downstream consumers.

### Task 1.1: Unify FFT Configuration

```python
# config.py
from dataclasses import dataclass

@dataclass
class SpectrogramConfig:
    fft_size: int = 4096
    window: str = 'hann'
    overlap: float = 0.5
    psd_ref_dbm: float = -30.0
    dynamic_range_db: float = 60.0
    colormap: str = 'viridis'
    
    def hash(self) -> int:
        return hash((self.fft_size, self.window, self.overlap, 
                     self.psd_ref_dbm, self.dynamic_range_db))
```

**Acceptance criteria:**
- [ ] Single `SpectrogramConfig` instance used by all components
- [ ] Config hash included in every `SpectrogramFrame`
- [ ] Changing config triggers clear warning/reset

### Task 1.2: Implement Frame Generator

```python
# spectrogram_generator.py
class CanonicalSpectrogramGenerator:
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        self.frame_id = 0
        self._init_gpu_resources()
    
    def process_iq(self, iq_samples: np.ndarray, timestamp_ns: int) -> SpectrogramFrame:
        """
        Generate one canonical spectrogram frame.
        All downstream views derive from this.
        """
        # FFT on GPU
        psd_db = self._compute_psd_gpu(iq_samples)
        
        # Normalize to 0-255
        normalized = self._normalize(psd_db)
        
        # Apply colormap
        rgba = self._apply_colormap(normalized)
        
        # Package frame
        frame = SpectrogramFrame(
            frame_id=self.frame_id,
            timestamp_ns=timestamp_ns,
            fft_config_hash=self.config.hash(),
            # ... other fields
            rgba_bytes=rgba.tobytes(),
            psd_db=psd_db,
        )
        
        self.frame_id += 1
        return frame
```

### Task 1.3: Inference Input Derivation

```python
# inference_pipeline.py
class InferencePipeline:
    def __init__(self, model_path: str, input_size: tuple = (640, 640)):
        self.input_size = input_size
        self.engine = self._load_tensorrt_engine(model_path)
    
    def prepare_input(self, frame: SpectrogramFrame) -> np.ndarray:
        """
        Derive inference input from canonical spectrogram.
        SPATIAL TRANSFORM ONLY - no re-FFT.
        """
        # Resize canonical spectrogram to model input size
        # Use area interpolation for downsampling (preserves energy)
        psd_resized = cv2.resize(
            frame.psd_db, 
            self.input_size, 
            interpolation=cv2.INTER_AREA
        )
        
        # Normalize for model
        tensor = self._normalize_for_model(psd_resized)
        return tensor
```

**Acceptance criteria:**
- [ ] Inference input is derived from canonical frame, never recomputed
- [ ] Box coordinates from inference map correctly back to canonical space
- [ ] Visual test: detection boxes align with visible signals

---

## Phase 2: Tracker Implementation

**Goal:** Stable tracks with no flicker, optional drift handling.

### Task 2.1: Core Tracker

```python
# tracker.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    
    # Boxes (canonical space, normalized 0-1)
    truth_box: np.ndarray           # Updated by detection matching
    display_box: np.ndarray         # Smoothed for rendering
    
    # Confidence
    confidence: float
    
    # Lifecycle
    state: str = 'tentative'        # tentative | confirmed | lost
    age: int = 0                    # Frames since last match
    hits: int = 1                   # Total matches
    
    # Drift detection (optional)
    motion_mode: str = 'stationary'
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    drift_evidence: int = 0         # Consecutive frames with consistent drift


class RFSignalTracker:
    """
    Production tracker for RF signals.
    Implements: IoU + frequency association, EMA smoothing, 
    confidence accumulation, optional drift detection.
    """
    
    def __init__(
        self,
        # Association
        iou_threshold: float = 0.2,
        freq_tolerance_normalized: float = 0.02,
        
        # Lifecycle
        confirm_hits: int = 3,
        max_age: int = 10,
        
        # Smoothing
        box_ema_alpha: float = 0.3,
        display_ema_alpha: float = 0.5,
        conf_rise_alpha: float = 0.3,
        conf_fall_alpha: float = 0.05,
        
        # Drift detection (optional)
        enable_drift_detection: bool = False,
        drift_frames_required: int = 5,
        drift_min_displacement: float = 0.005,  # Normalized units
    ):
        self.iou_thresh = iou_threshold
        self.freq_tol = freq_tolerance_normalized
        self.confirm_hits = confirm_hits
        self.max_age = max_age
        self.box_alpha = box_ema_alpha
        self.display_alpha = display_ema_alpha
        self.conf_rise = conf_rise_alpha
        self.conf_fall = conf_fall_alpha
        
        self.enable_drift = enable_drift_detection
        self.drift_frames = drift_frames_required
        self.drift_min_disp = drift_min_displacement
        
        self.tracks: dict[int, TrackState] = {}
        self.next_id = 0
    
    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """
        Main entry point. Call once per inference frame.
        Returns list of tracks for UI rendering.
        """
        det_used = [False] * len(detections)
        
        # 1. Match existing tracks to detections
        for track in self.tracks.values():
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
        self.tracks = {
            tid: t for tid, t in self.tracks.items() 
            if t.state != 'lost'
        }
        
        # 4. Convert to output format
        return self._export_tracks(frame_id)
    
    def _find_best_match(
        self, 
        track: TrackState, 
        detections: list[Detection],
        det_used: list[bool]
    ) -> Optional[int]:
        """Find best matching detection for a track."""
        best_idx = None
        best_score = 0.0
        
        for i, det in enumerate(detections):
            if det_used[i]:
                continue
            if det.class_id != track.class_id:
                continue
            
            iou = self._compute_iou(track.truth_box, np.array(det.box))
            freq_dist = abs(
                self._center_freq(track.truth_box) - 
                self._center_freq(np.array(det.box))
            )
            
            # Match if IoU is good OR frequency is close
            if iou >= self.iou_thresh or freq_dist <= self.freq_tol:
                score = iou + max(0, self.freq_tol - freq_dist)
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        return best_idx
    
    def _update_matched_track(self, track: TrackState, det: Detection):
        """Update track with matched detection."""
        new_box = np.array(det.box)
        
        # Drift detection (optional)
        if self.enable_drift:
            displacement = new_box - track.truth_box
            self._update_drift_state(track, displacement)
        
        # Update truth box
        if track.motion_mode == 'drifting' and self.enable_drift:
            # Predict then correct
            predicted = track.truth_box + track.velocity
            track.truth_box = self.box_alpha * new_box + (1 - self.box_alpha) * predicted
        else:
            # Pure EMA
            track.truth_box = self.box_alpha * new_box + (1 - self.box_alpha) * track.truth_box
        
        # Update display box (extra smoothing)
        track.display_box = (
            self.display_alpha * track.truth_box + 
            (1 - self.display_alpha) * track.display_box
        )
        
        # Update confidence (asymmetric: fast rise, slow fall)
        if det.confidence > track.confidence:
            alpha = self.conf_rise
        else:
            alpha = self.conf_fall
        track.confidence = alpha * det.confidence + (1 - alpha) * track.confidence
        
        # Update lifecycle
        track.age = 0
        track.hits += 1
        if track.hits >= self.confirm_hits and track.state == 'tentative':
            track.state = 'confirmed'
    
    def _update_drift_state(self, track: TrackState, displacement: np.ndarray):
        """Detect and track signal drift."""
        disp_magnitude = np.linalg.norm(displacement[:2])  # x, y only
        
        if disp_magnitude > self.drift_min_disp:
            # Check direction consistency
            if track.drift_evidence > 0:
                direction_consistent = np.dot(displacement, track.velocity) > 0
            else:
                direction_consistent = True
            
            if direction_consistent:
                track.drift_evidence += 1
                # Update velocity estimate with EMA
                track.velocity = 0.3 * displacement + 0.7 * track.velocity
            else:
                track.drift_evidence = max(0, track.drift_evidence - 2)
        else:
            track.drift_evidence = max(0, track.drift_evidence - 1)
            track.velocity *= 0.8  # Decay velocity
        
        # Switch modes
        if track.drift_evidence >= self.drift_frames:
            track.motion_mode = 'drifting'
        elif track.drift_evidence == 0:
            track.motion_mode = 'stationary'
            track.velocity = np.zeros(4)
    
    def _age_unmatched_track(self, track: TrackState):
        """Age a track that wasn't matched this frame."""
        track.age += 1
        track.confidence *= 0.95  # Slow decay
        track.drift_evidence = max(0, track.drift_evidence - 1)
        
        if track.age > self.max_age:
            track.state = 'lost'
    
    def _create_track(self, det: Detection, frame_id: int):
        """Create new track from unmatched detection."""
        box = np.array(det.box)
        track = TrackState(
            track_id=self.next_id,
            class_id=det.class_id,
            class_name=det.class_name,
            truth_box=box.copy(),
            display_box=box.copy(),
            confidence=det.confidence,
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _export_tracks(self, frame_id: int) -> list[Track]:
        """Convert internal state to output format."""
        result = []
        for t in self.tracks.values():
            if t.state == 'lost':
                continue
            
            # Convert box to Hz
            freq_center = self._box_to_freq_center(t.display_box)
            freq_bw = self._box_to_freq_bandwidth(t.display_box)
            
            result.append(Track(
                track_id=t.track_id,
                class_id=t.class_id,
                class_name=t.class_name,
                truth_box=tuple(t.truth_box),
                display_box=tuple(t.display_box),
                confidence=t.confidence,
                state=t.state,
                age_frames=t.age,
                hits=t.hits,
                motion_mode=t.motion_mode,
                drift_rate_hz_per_sec=self._velocity_to_hz_per_sec(t.velocity),
                drift_confidence=min(1.0, t.drift_evidence / self.drift_frames),
                freq_center_hz=freq_center,
                freq_bandwidth_hz=freq_bw,
                first_seen_frame_id=0,  # TODO: track this
                last_seen_frame_id=frame_id,
            ))
        
        return result
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    @staticmethod
    def _center_freq(box: np.ndarray) -> float:
        return (box[0] + box[2]) / 2
```

**Acceptance criteria:**
- [ ] No flicker: tracks are stable frame-to-frame
- [ ] No jitter: display boxes are smooth
- [ ] Confidence doesn't flash: accumulation works
- [ ] Lifecycle states progress correctly: tentative → confirmed → lost
- [ ] Drift detection activates only with consistent evidence (if enabled)

---

## Phase 5: TensorRT Deployment

**Goal:** Optimized inference with bounded latency.

### Task 5.1: Model Export

```bash
# On development machine
# Export YOLOv8s to ONNX
pip install ultralytics
yolo export model=yolov8s.pt format=onnx imgsz=640 opset=17 simplify=True
```

### Task 5.2: Engine Build (ON TARGET DEVICE)

```bash
# SSH to Orin NX
# Build TensorRT engine - MUST be done on target device

# FP16 (recommended first)
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov8s.onnx \
    --saveEngine=yolov8s_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --verbose

# INT8 (requires calibration)
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov8s.onnx \
    --saveEngine=yolov8s_int8.engine \
    --int8 \
    --calib=calibration_cache.bin \
    --workspace=4096 \
    --verbose
```

### Task 5.3: Calibration Data Collection

```python
# calibration.py
"""
Collect calibration images for INT8 quantization.
Run this on representative RF data from target environment.
"""
import numpy as np
from pathlib import Path

class CalibrationDataCollector:
    def __init__(self, output_dir: str, target_count: int = 500):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.target_count = target_count
        self.collected = 0
    
    def add_frame(self, frame: SpectrogramFrame):
        """Add a spectrogram frame to calibration set."""
        if self.collected >= self.target_count:
            return
        
        # Derive inference input (same as production)
        tensor = self._prepare_inference_input(frame)
        
        # Save as numpy
        np.save(
            self.output_dir / f"calib_{self.collected:04d}.npy",
            tensor
        )
        self.collected += 1
        
        if self.collected % 50 == 0:
            print(f"Calibration: {self.collected}/{self.target_count}")
```

### Task 5.4: Inference Wrapper

```python
# tensorrt_inference.py
import tensorrt as trt
import numpy as np
from cuda import cudart

class TensorRTInference:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """Pre-allocate input/output buffers."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            
            # Allocate device memory
            err, device_mem = cudart.cudaMalloc(size * np.dtype(dtype).itemsize)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name, 'shape': shape, 'dtype': dtype,
                    'device': device_mem, 'size': size
                })
            else:
                self.outputs.append({
                    'name': name, 'shape': shape, 'dtype': dtype,
                    'device': device_mem, 'size': size
                })
    
    def infer(self, input_tensor: np.ndarray) -> list[Detection]:
        """
        Run inference on prepared input tensor.
        Returns list of Detection objects.
        """
        # Copy input to device
        cudart.cudaMemcpy(
            self.inputs[0]['device'],
            input_tensor.ctypes.data,
            input_tensor.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        
        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], inp['device'])
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], out['device'])
        
        # Execute
        self.context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)
        
        # Copy outputs to host and parse
        return self._parse_outputs()
    
    def _parse_outputs(self) -> list[Detection]:
        """Parse YOLOv8 outputs into Detection objects."""
        # Implementation depends on model output format
        # YOLOv8 outputs: [batch, 84, 8400] for 80-class COCO
        # Custom model may differ
        pass
```

**Acceptance criteria:**
- [ ] Engine builds successfully on Orin NX
- [ ] FP16 inference latency ≤20ms
- [ ] INT8 inference latency ≤15ms (if calibrated)
- [ ] No accuracy regression vs PyTorch baseline (mAP within 1%)
- [ ] Engine loads in <5 seconds

---

## Phase 6: Async Database Logging

**Goal:** Never block real-time pipeline.

```python
# database.py
import asyncio
import asyncpg
from dataclasses import asdict
from typing import Optional

class DetectionLogger:
    """
    Async batch logger for detections.
    Never blocks the real-time pipeline.
    """
    
    def __init__(
        self,
        connection_string: str,
        batch_size: int = 50,
        flush_interval_ms: int = 500,
    ):
        self.conn_string = connection_string
        self.batch_size = batch_size
        self.flush_interval = flush_interval_ms / 1000.0
        
        self.queue: asyncio.Queue[Track] = asyncio.Queue(maxsize=1000)
        self.pool: Optional[asyncpg.Pool] = None
        self._running = False
    
    async def start(self):
        """Initialize connection pool and start flush task."""
        self.pool = await asyncpg.create_pool(
            self.conn_string,
            min_size=2,
            max_size=5,
        )
        await self._create_tables()
        
        self._running = True
        asyncio.create_task(self._flush_loop())
    
    async def _create_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS tracks (
                    id SERIAL PRIMARY KEY,
                    track_id INTEGER NOT NULL,
                    frame_id BIGINT NOT NULL,
                    timestamp_ns BIGINT NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name VARCHAR(64),
                    confidence REAL,
                    state VARCHAR(16),
                    box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
                    freq_center_hz DOUBLE PRECISION,
                    freq_bandwidth_hz DOUBLE PRECISION,
                    motion_mode VARCHAR(16),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_tracks_frame 
                ON tracks(frame_id);
                
                CREATE INDEX IF NOT EXISTS idx_tracks_time 
                ON tracks(timestamp_ns);
            ''')
    
    def log(self, track: Track, frame_id: int, timestamp_ns: int):
        """
        Non-blocking log. Drops if queue is full.
        """
        try:
            self.queue.put_nowait((track, frame_id, timestamp_ns))
        except asyncio.QueueFull:
            # Drop oldest to make room (back-pressure)
            try:
                self.queue.get_nowait()
                self.queue.put_nowait((track, frame_id, timestamp_ns))
            except:
                pass  # Best effort
    
    async def _flush_loop(self):
        """Background task to batch insert."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush queued tracks to database."""
        batch = []
        while len(batch) < self.batch_size:
            try:
                item = self.queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break
        
        if not batch:
            return
        
        async with self.pool.acquire() as conn:
            await conn.executemany('''
                INSERT INTO tracks 
                (track_id, frame_id, timestamp_ns, class_id, class_name,
                 confidence, state, box_x1, box_y1, box_x2, box_y2,
                 freq_center_hz, freq_bandwidth_hz, motion_mode)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ''', [
                (t.track_id, fid, ts, t.class_id, t.class_name,
                 t.confidence, t.state, 
                 t.display_box[0], t.display_box[1], t.display_box[2], t.display_box[3],
                 t.freq_center_hz, t.freq_bandwidth_hz, t.motion_mode)
                for t, fid, ts in batch
            ])
    
    async def stop(self):
        """Flush remaining and close."""
        self._running = False
        await self._flush_batch()  # Final flush
        await self.pool.close()
```

**Acceptance criteria:**
- [ ] Logging never blocks inference or display
- [ ] Queue overflow drops gracefully (no crash)
- [ ] Batch inserts complete in <10ms for 50 records
- [ ] All logged tracks have frame_id for replay correlation

---

## Configuration Files

### config/spectrogram.yaml

```yaml
# Canonical spectrogram configuration
fft_size: 4096
window: hann
overlap: 0.5
psd_ref_dbm: -30.0
dynamic_range_db: 60.0
colormap: viridis
```

### config/tracker.yaml

```yaml
# Tracker parameters
association:
  iou_threshold: 0.2
  freq_tolerance_normalized: 0.02

lifecycle:
  confirm_hits: 3
  max_age: 10

smoothing:
  box_ema_alpha: 0.3
  display_ema_alpha: 0.5
  conf_rise_alpha: 0.3
  conf_fall_alpha: 0.05

drift_detection:
  enabled: false
  frames_required: 5
  min_displacement: 0.005
```

### config/inference.yaml

```yaml
# Inference configuration
model_path: models/yolov8s_fp16.engine
input_size: [640, 640]
confidence_threshold: 0.25
nms_threshold: 0.45
```

### config/database.yaml

```yaml
# Database configuration
connection_string: postgresql://user:pass@localhost/g20_tracks
batch_size: 50
flush_interval_ms: 500
max_queue_size: 1000
```
