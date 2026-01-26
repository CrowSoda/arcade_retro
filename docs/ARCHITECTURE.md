# G20 Architecture

## Overview

G20 is a real-time RF signal detection system with a Flutter desktop UI and Python GPU backend. The system processes IQ data through GPU-accelerated FFT pipelines, runs Faster R-CNN object detection, and streams waterfall displays to the UI.

## Hard Points

### 1. Dual FFT Configuration (Inference vs Display)

The system maintains **two completely separate FFT pipelines**:

**Inference FFT** (fixed, locked to model training):
```
FFT Size: 4096
Hop Length: 2048 (50% overlap)
Dynamic Range: 80 dB
Output: 1024×1024 spectrogram → Faster R-CNN
```

**Waterfall FFT** (user-configurable):
```
FFT Size: 8192 - 65536 (selectable)
Hop Length: FFT_SIZE / 2
Dynamic Range: 60 dB
Output: 2048-wide RGBA strips → display
```

If you change inference FFT parameters, the model outputs garbage. The model was trained on 80dB dynamic range spectrograms - feeding it 60dB data produces wrong detections.

### 2. Row-Strip Protocol

The waterfall uses row-strip streaming instead of video encoding:

```
Message format: [TYPE:1][HEADER:17][PIXELS][PSD]

Header (17 bytes):
├─ frame_id:     uint32 (4 bytes)
├─ total_rows:   uint32 (4 bytes)  ← monotonic counter
├─ rows_in_strip: uint16 (2 bytes) ← ~20 rows/frame
├─ strip_width:  uint16 (2 bytes)  ← 2048
├─ pts:          float32 (4 bytes)
└─ source_id:    uint8 (1 byte)    ← waterfall source

Pixel data:
└─ RGBA bytes: rows_in_strip × strip_width × 4

PSD data:
└─ Float32 dB values: strip_width × 4 bytes
```

Flutter maintains a local pixel buffer and shifts it upward on each strip, pasting new rows at the bottom. Detection boxes use `base_row + row_offset` for positioning.

### 3. Detection Box Synchronization

Inference runs on **6 accumulated frames** (not single frames). Detection coordinates are normalized 0-1 relative to the inference window:

```python
# Backend computes row offset from model x-coordinates
total_inference_rows = rows_per_frame * inference_chunk_count  # ~228 rows
row_offset = int(detection.x1 * total_inference_rows)
row_span = int((detection.x2 - detection.x1) * total_inference_rows)
```

Flutter converts to display coordinates:
```dart
// row_offset relative to base_row (when inference started)
display_y = buffer_height - (total_rows_received - (base_row + row_offset))
```

### 4. Backend Lifecycle Management

The Flutter app auto-launches the Python backend:

1. **Startup**: `BackendLauncher` spawns `python server.py --ws-port 0`
2. **Discovery**: Server prints `WS_PORT:XXXX`, Flutter parses stdout
3. **Watchdog**: Python monitors parent PID, exits if Flutter dies
4. **Shutdown**: Flutter sends taskkill, Python cleans up resources

PID file (`backend/.backend.pid`) tracks stale processes for cleanup.

### 5. GPU FFT Processing

`gpu_fft.py` uses cuFFT via PyTorch for waterfall FFT:

```python
# Batched FFT on GPU (5-10x faster than CPU)
fft_result = torch.fft.rfft(windowed_segments, dim=1)
magnitudes = torch.abs(fft_result)
db = 20 * torch.log10(magnitudes + 1e-10)

# Decimate to fixed 20 rows per frame (regardless of FFT size)
decimated = db.reshape(TARGET_ROWS, pool_size, fft_size).max(axis=1)
```

This decouples FFT resolution from display bandwidth - larger FFT gives finer frequency resolution but same frame rate.

### 6. Model Loading

Faster R-CNN with ResNet18 backbone:

```python
backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
model = FasterRCNN(backbone, num_classes=2)
state = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(state)
model.half()  # FP16 for speed
```

Backend auto-detects TensorRT (.trt), ONNX (.onnx), or PyTorch (.pth) models. TensorRT gives best performance on Jetson.

## Data Flow

```
IQ File (.sigmf-data)
    │
    ├─► Waterfall Pipeline (GPU FFT)
    │   └─► Colormap → RGBA strips → WebSocket → Flutter display
    │
    └─► Inference Pipeline (every 6 frames)
        ├─► Spectrogram (4096 FFT, 80dB) → 1024×1024
        ├─► Faster R-CNN → bounding boxes
        └─► JSON detections → WebSocket → Flutter overlay
```

## Communication Channels

| Channel | Protocol | Purpose |
|---------|----------|---------|
| WebSocket `/ws/video` | Binary + JSON | Row strips, detections, metadata |
| gRPC :50051 | Protobuf | Device control, inference control |

WebSocket handles the high-bandwidth streaming. gRPC handles control commands that need request/response semantics.

## WebSocket Commands

Commands sent from Flutter to backend:

```json
{"command": "set_time_span", "seconds": 5.0}
{"command": "set_fps", "fps": 30}
{"command": "set_fft_size", "size": 32768}
{"command": "set_colormap", "colormap": 0}
{"command": "set_score_threshold", "threshold": 0.5}
{"command": "set_db_range", "min_db": -100, "max_db": -20}
{"command": "stop"}
```

## File Formats

### IQ Data (.sigmf-data)

Raw complex64 samples (interleaved float32 I/Q):
```
[I0][Q0][I1][Q1][I2][Q2]...
└─ 8 bytes per sample (2 × float32)
```

Metadata in `.sigmf-meta` JSON file.

### Model Files (.pth)

PyTorch state dict with:
- ResNet18 backbone weights
- FPN neck weights  
- RPN (region proposal network) weights
- Detection head weights

2 classes: background (0), signal (1).

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Frame rate | 30 fps | Waterfall + PSD update |
| Inference latency | <10ms | Per 6-frame batch |
| GPU FFT | <5ms | Per frame, any FFT size |
| End-to-end latency | <60ms | IQ sample to display |

## Troubleshooting

**Detections don't match signals**: Check dynamic range. Model trained on 80dB, waterfall shows 60dB.

**Boxes offset vertically**: Row sync issue. Check `base_row` and `row_offset` in detection JSON.

**Backend won't start**: Check `backend/.backend.pid` for stale process. Kill and delete file.

**No GPU FFT**: Verify CUDA available with `python -c "import torch; print(torch.cuda.is_available())"`.
