# G20 Backend

Python GPU server for RF signal detection. Runs GPU-accelerated FFT and Faster R-CNN inference, streams waterfall data to Flutter via WebSocket.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Generate gRPC stubs
generate_stubs.bat
```

## Run

```bash
# Both WebSocket + gRPC (default)
python server.py

# WebSocket only (port auto-discovered)
python server.py --ws-only --ws-port 0

# gRPC only
python server.py --grpc-only --port 50051
```

When using `--ws-port 0`, the OS picks a free port and the server prints `WS_PORT:XXXX` for the Flutter app to read.

## Files

| File | Purpose |
|------|---------|
| `server.py` | WebSocket + gRPC server, request routing |
| `unified_pipeline.py` | FFT processing, inference, row-strip encoding |
| `gpu_fft.py` | CUDA FFT via PyTorch, batched processing |
| `inference.py` | TensorRT/PyTorch model loading and execution |
| `waterfall_buffer.py` | Pixel buffer management |
| `colormaps.py` | Viridis, plasma, inferno, magma, turbo LUTs |

## WebSocket Protocol

### Endpoints

- `/ws/video` - Row-strip waterfall streaming (primary)
- `/ws/unified` - Legacy full-frame streaming
- `/ws/inference` - Inference-only mode

### Message Types

| Type | Code | Direction | Content |
|------|------|-----------|---------|
| Strip | 0x01 | Server→Client | Binary header + RGBA pixels + PSD dB |
| Detection | 0x02 | Server→Client | JSON detection frame |
| Metadata | 0x03 | Server→Client | JSON stream config |

### Commands (Client→Server)

```json
{"command": "set_time_span", "seconds": 5.0}
{"command": "set_fps", "fps": 30}
{"command": "set_fft_size", "size": 32768}
{"command": "set_colormap", "colormap": 0}
{"command": "set_score_threshold", "threshold": 0.5}
{"command": "set_db_range", "min_db": -100, "max_db": -20}
{"command": "stop"}
```

## FFT Configuration

Two separate pipelines with independent parameters:

**Inference (fixed)**:
```python
fft_size = 4096
hop_length = 2048
dynamic_range = 80.0  # Must match model training
output_size = 1024×1024
```

**Waterfall (configurable)**:
```python
fft_size = 8192 | 16384 | 32768 | 65536
hop_length = fft_size // 2
dynamic_range = 60.0
output_width = 2048
rows_per_frame = 20  # Fixed via max-pooling
```

Changing waterfall FFT size triggers cuFFT warmup (100-500ms).

## Benchmark

```bash
# Single model
python inference.py --model ../models/detector.pth --benchmark

# Multi-model parallel (CUDA streams)
python inference.py --model ../models/detector.pth --benchmark-multi 6

# TensorRT export
python inference.py --model ../models/detector.pth --export-trt
```

## Performance

| Operation | RTX 4090 | Orin NX |
|-----------|----------|---------|
| Inference (FP16) | ~3ms | ~5ms |
| GPU FFT (64K) | ~4ms | ~8ms |
| Row encoding | ~1ms | ~2ms |
| Total frame | ~10ms | ~18ms |

## Shutdown Handling

- Parent watchdog monitors Flutter PID, exits if parent dies
- Signal handlers for SIGINT, SIGTERM, SIGBREAK (Windows)
- PID file (`backend/.backend.pid`) for stale process cleanup
- `taskkill /F /T` on Windows to kill process tree

## Data Directories

```
../models/    # .pth, .trt, .onnx model files
../data/      # .sigmf-data IQ files
../config/    # YAML configuration
```

Server auto-discovers first available model and IQ file on startup.
