# RF Signal Detection Integration Plan

**For:** Senior Software Engineer  
**Date:** January 24, 2026  
**Status:** Ready for Implementation  
**Project:** G20/NV100 RF Waterfall Detection System

---

## Executive Summary

This document provides the complete specification for integrating trained RF signal detection models into the G20 demo system. It covers the exact training configuration used, spectrogram generation parameters, waterfall display design recommendations, and a prioritized task list.

**Key Decisions:**
- **Model:** Faster R-CNN with ResNet18 backbone, 2 classes, trained from scratch
- **Waterfall Display:** 4-second rolling window (20 rows at 200ms/chunk)
- **Target Platform:** NVIDIA Orin NX with TensorRT FP16

---

## Part 1: Model Training Specification

### 1.1 Model Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Architecture** | Faster R-CNN | Two-stage detector, high accuracy |
| **Backbone** | ResNet18 + FPN | Lighter than ResNet50, faster inference |
| **Trainable Layers** | 5 (all) | Full training from scratch |
| **Pretrained Weights** | No | Domain-specific training |
| **Num Classes** | 2 | Background + Signal |

### 1.2 Training Configuration

```python
# Exact parameters from TENSORCADE training
training_config = {
    "backbone": "resnet18",
    "pretrained": False,
    "trainable_layers": 5,
    "num_classes": 2,
    "k_folds": 3,                    # K-fold cross-validation
    "early_stop_patience": 10,       # Epochs without improvement
    "learning_rate": 0.001,
    "batch_size": 4,
    "epochs": 100,                   # Max epochs (early stop may trigger)
}
```

### 1.3 Model Files

| File | Purpose | Location |
|------|---------|----------|
| `modern_burst_gap_fold3.pth` | PyTorch checkpoint | `ARCADE/models/curated/` |
| `modern_burst_gap_fold3.onnx` | ONNX export (portable) | To be generated |
| `modern_burst_gap_fold3.engine` | TensorRT engine (Orin) | Built ON target |

### 1.4 Inference Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Precision** | FP16 | Half precision for speed |
| **Score Threshold** | 0.90 | High confidence only |
| **NMS IoU Threshold** | 0.45 | Standard NMS |
| **Input Size** | 1024×1024 | Matches training output |

---

## Part 2: Spectrogram Generation Pipeline

### 2.1 STFT Parameters

```python
# Canonical spectrogram configuration
spectrogram_config = {
    "nfft": 4096,               # FFT size (frequency resolution)
    "noverlap": 2048,           # 50% overlap
    "window": "hann",           # Hann window
    "chunk_ms": 200.0,          # Time per chunk
    "dynamic_range_db": 80.0,   # dB range for normalization
    "output_size": 1024,        # Output image dimensions
}
```

### 2.2 Processing Pipeline

```
IQ Samples (complex64)
       │
       ▼
┌─────────────────────────────┐
│  STFT (torch.stft or scipy) │
│  nfft=4096, hop=2048        │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  FFT Shift (center DC)      │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  PSD: 10*log10(|X|² + 1e-12)│
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Normalize: (psd - vmin) /  │
│  (vmax - vmin) → [0, 1]     │
│  vmin = vmax - 80 dB        │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Resize to 1024×1024        │
│  (cv2.INTER_AREA)           │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Apply colormap → RGBA      │
│  (for display)              │
└─────────────────────────────┘
       │
       ▼
  SpectrogramFrame
```

### 2.3 Code Reference (TENSORCADE)

```python
# From ARCADE/src/tensorcade/workers.py
def process_chunk(self, chunk_iq, ...):
    # 1. GPU STFT
    Zxx = torch.stft(
        chunk_gpu,
        n_fft=4096,
        hop_length=2048,
        window=torch.hann_window(4096, device=device),
        center=False,
        return_complex=True
    )
    
    # 2. FFT shift + PSD
    Zxx_shifted = torch.fft.fftshift(Zxx, dim=0)
    sxx_db = 10 * torch.log10(Zxx_shifted.abs().square() + 1e-12)
    
    # 3. Normalize
    vmax = sxx_db.max()
    vmin = vmax - 80.0  # dynamic_range
    sxx_norm = ((sxx_db - vmin) / (vmax - vmin + 1e-12)).clamp_(0, 1)
    
    # 4. Resize to 1024x1024
    resized = F.interpolate(sxx_norm.unsqueeze(0).unsqueeze(0), 
                           size=(1024, 1024), mode='bilinear')
```

---

## Part 3: Waterfall Display Design

### 3.1 Recommendation: 4-Second Rolling Window

**Why 4 seconds?**
- Matches human attention span for signal analysis
- Tracks stay visible long enough for operator to confirm
- Scale doesn't make signals look too small
- Manageable memory footprint

### 3.2 Configuration

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| **Chunk duration** | 200ms | From training config |
| **Display window** | 4 seconds | Design choice |
| **Display rows** | 20 | 4000ms ÷ 200ms |
| **Frame buffer** | 30 frames | 20 + 10 margin |
| **Scroll rate** | 5 rows/sec | 1000ms ÷ 200ms |

### 3.3 Time Axis Convention

```
┌────────────────────────────────┐
│  Row 0 (oldest) - t = -4.0s    │  ← Scrolls off top
│  Row 1           - t = -3.8s   │
│  ...                           │
│  Row 18          - t = -0.4s   │
│  Row 19 (newest) - t = -0.2s   │  ← New frames appear
└────────────────────────────────┘
        Frequency axis →
```

### 3.4 Frame Buffer Configuration

```yaml
# config/spectrogram.yaml - UPDATED
fft_size: 4096
window: 'hann'
overlap: 0.5
sample_rate_hz: 20.0e6
psd_ref_dbm: -30.0
dynamic_range_db: 80.0
target_width: 1024

# Waterfall display (NEW)
chunk_ms: 200.0
display_window_sec: 4.0
display_rows: 20
frame_buffer_size: 30
```

### 3.5 Flutter Widget Update

```dart
// Recommended settings for WaterfallFrameWidget
WaterfallFrameWidget(
  frameStream: provider.frameStream,
  displayRows: 20,          // 4 seconds at 200ms/chunk
  chunkDurationMs: 200.0,
  timeAxisLabel: '-4s to 0s',
)
```

---

## Part 4: Architecture Comparison

### 4.1 TENSORCADE vs G20 Demo

| Component | TENSORCADE | G20 Demo | Recommendation |
|-----------|------------|----------|----------------|
| **Detection Model** | Faster R-CNN | YOLOv8s | Keep YOLOv8s for speed |
| **Framework** | PyTorch | TensorRT | Keep TensorRT |
| **STFT** | torch.stft (GPU) | numpy.fft (CPU) | **Move to GPU** |
| **Batching** | 4 chunks | 1 chunk | **Add batching** |
| **Precision** | FP16 | FP16 | Keep |
| **GUI** | PyQt5 | Flutter | Keep Flutter |
| **Tracker** | None | IoU + EMA | Keep tracker |

### 4.2 Best Practices to Adopt from TENSORCADE

1. **GPU STFT** - Move FFT computation to GPU
   ```python
   # Current (CPU)
   spectrum = np.fft.fft(windowed)
   
   # Recommended (GPU)
   spectrum = torch.stft(chunk_gpu, n_fft=4096, ...)
   ```

2. **Batched Inference** - Process 4 frames at once
   ```python
   # Current (single frame)
   output = model(frame)
   
   # Recommended (batched)
   outputs = model(torch.stack([f1, f2, f3, f4]))
   ```

3. **Pre-allocated Buffers** - Avoid allocation in hot path
   ```python
   # In __init__
   self._gpu_buffer = torch.zeros(batch_size, 3, 1024, 1024, device='cuda')
   
   # In process loop
   self._gpu_buffer[:batch_size].copy_(batch_tensor)  # Reuse buffer
   ```

4. **Telemetry** - Add timing instrumentation
   ```python
   # Measure each step
   times['stft'] = (time.perf_counter() - t0) * 1000
   times['inference'] = ...
   push_to_grafana(times)
   ```

---

## Part 5: Tasks for SWE

### Priority 1: Model Integration (BLOCKING)

```markdown
- [ ] **Export model to ONNX**
      ```bash
      python -c "
      import torch
      from torchvision.models.detection import fasterrcnn_resnet50_fpn
      # Load your model
      model = ...
      model.load_state_dict(torch.load('modern_burst_gap_fold3.pth'))
      model.eval()
      
      dummy = torch.randn(1, 3, 1024, 1024)
      torch.onnx.export(model, dummy, 'modern_burst_gap_fold3.onnx',
                       input_names=['input'],
                       output_names=['boxes', 'labels', 'scores'],
                       opset_version=17)
      "
      ```

- [ ] **Build TensorRT engine ON ORIN NX**
      ```bash
      # SSH to target
      python g20_demo/scripts/build_tensorrt_engine.py \
          --onnx models/modern_burst_gap_fold3.onnx \
          --output models/modern_burst_gap_fold3.engine \
          --fp16
      ```

- [ ] **Update inference.yaml with model path**
      ```yaml
      model:
        onnx_path: "models/modern_burst_gap_fold3.onnx"
        tensorrt_path: "models/modern_burst_gap_fold3.engine"
      class_names:
        - "background"
        - "signal"
      ```

- [ ] **Validate accuracy against TENSORCADE baseline**
      - Run both systems on same IQ file
      - Compare detection boxes and scores
      - Accept if F1 difference < 2%
```

### Priority 2: Waterfall Display (4-second window)

```markdown
- [ ] **Update frame_buffer configuration**
      File: `g20_demo/backend/transport/frame_buffer.py`
      ```python
      # Change defaults
      def __init__(self, max_frames: int = 30, display_rows: int = 20):
      ```

- [ ] **Update spectrogram.yaml**
      ```yaml
      # Add new parameters
      chunk_ms: 200.0
      display_window_sec: 4.0
      display_rows: 20
      ```

- [ ] **Update Flutter waterfall widget**
      File: `g20_demo/lib/features/live_detection/widgets/`
      - Set displayRows = 20
      - Add time axis labels (-4s to 0s)
      - Verify scroll rate matches 200ms chunks

- [ ] **Test track overlay alignment**
      - Tracks should scroll with spectrogram
      - No drift over 30+ seconds
```

### Priority 3: Performance Optimization

```markdown
- [ ] **Implement GPU STFT (Optional but recommended)**
      File: `g20_demo/backend/spectrogram/generator.py`
      ```python
      # Replace numpy FFT with torch STFT
      import torch
      
      def process_iq_gpu(self, iq_samples, ...):
          chunk_gpu = torch.from_numpy(iq_samples).to('cuda')
          Zxx = torch.stft(chunk_gpu, n_fft=self.config.fft_size, ...)
          ...
      ```

- [ ] **Add batched inference (batch_size=4)**
      File: `g20_demo/backend/inference/engine.py`
      - Accumulate 4 frames before inference
      - Or use sliding window approach

- [ ] **Pre-allocate GPU buffers**
      - Avoid torch.zeros() in hot path
      - Reuse buffers across frames
```

### Priority 4: Validation & Testing

```markdown
- [ ] **Run full test suite on target**
      ```bash
      python g20_demo/tests/run_all_tests.py
      # Expected: 39/39 passing
      ```

- [ ] **Benchmark end-to-end latency**
      ```bash
      python g20_demo/scripts/run_demo.py --duration 60 --database benchmark.db
      # Target: < 100ms end-to-end
      # Check: avg_frame_time_ms in output
      ```

- [ ] **Test with real G20/NV100 hardware**
      - Replace StubIQSource with SidekiqSource
      - Verify IQ capture works
      - Run 24-hour stability test
```

---

## Part 6: File Checklist

### Files to Modify

| File | Change |
|------|--------|
| `config/spectrogram.yaml` | Add chunk_ms, display_rows |
| `config/inference.yaml` | Update model paths, num_classes=2 |
| `backend/transport/frame_buffer.py` | Change default display_rows=20 |
| `lib/.../waterfall_*.dart` | Set displayRows=20, add time labels |

### Files to Create

| File | Purpose |
|------|---------|
| `models/modern_burst_gap_fold3.onnx` | ONNX export |
| `models/modern_burst_gap_fold3.engine` | TensorRT (on target) |

### Files Already Complete

| File | Tests |
|------|-------|
| `backend/spectrogram/` | 13 tests ✓ |
| `backend/tracker/` | 10 tests ✓ |
| `backend/transport/` | 9 tests ✓ |
| `backend/database/` | 7 tests ✓ |
| `backend/inference/engine.py` | Ready for model |
| `lib/.../track_overlay.dart` | Ready |

---

## Part 7: Quick Reference

### Commands

```bash
# Run tests
python g20_demo/tests/run_all_tests.py

# Run demo (stub signals)
python g20_demo/scripts/run_demo.py --duration 60

# Export model to ONNX
python -c "import torch; ..."

# Build TensorRT engine (ON TARGET)
python g20_demo/scripts/build_tensorrt_engine.py --onnx ... --output ... --fp16
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| NFFT | 4096 |
| Overlap | 2048 (50%) |
| Dynamic Range | 80 dB |
| Chunk Duration | 200ms |
| Display Window | 4 seconds |
| Display Rows | 20 |
| Score Threshold | 0.90 |
| Num Classes | 2 |
| Precision | FP16 |

---

**Document prepared by:** AI Implementation Assistant  
**Review by:** Senior SWE  
**Date:** January 24, 2026
