# G20 RF Detection Platform

Real-time RF signal detection and visualization using GPU-accelerated spectrograms and Faster R-CNN inference.

## Quick Start

```bash
# 1. Install Flutter dependencies
flutter pub get

# 2. Install Python backend dependencies  
cd backend && pip install -r requirements.txt

# 3. Generate gRPC stubs
cd backend && generate_stubs.bat

# 4. Run
flutter run -d windows
```

The Flutter app auto-launches the Python backend on startup.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  FLUTTER (Desktop UI)                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │ Waterfall View  │  │  PSD Chart      │  │ Detection Table │       │
│  │ (RGBA pixels)   │  │  (dB values)    │  │ (row-synced)    │       │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘       │
│           │                    │                    │                │
│           └────────────────────┴────────────────────┘                │
│                                │                                     │
│                    ┌───────────▼───────────┐                         │
│                    │  VideoStreamProvider  │                         │
│                    │   (WebSocket client)  │                         │
│                    └───────────┬───────────┘                         │
└────────────────────────────────┼─────────────────────────────────────┘
                                 │ WebSocket (row strips)
┌────────────────────────────────┼─────────────────────────────────────┐
│  PYTHON BACKEND                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                    server.py (WebSocket + gRPC)             │     │
│  └──────────────────────────────┬──────────────────────────────┘     │
│                                 │                                    │
│  ┌──────────────────────────────▼──────────────────────────────┐     │
│  │                  unified_pipeline.py                         │     │
│  │  ┌────────────┐  ┌─────────────────┐  ┌──────────────────┐  │     │
│  │  │ IQ Source  │→ │ GPU FFT (cuFFT) │→ │ Faster R-CNN     │  │     │
│  │  │ (.sigmf)   │  │ (waterfall)     │  │ (TensorRT/PT)    │  │     │
│  │  └────────────┘  └─────────────────┘  └──────────────────┘  │     │
│  └─────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, and hard points |
| [Backend](backend/README.md) | Python server setup and benchmarks |
| [TENSORCADE Compat](g20_tensorcade_compat/README.md) | Model validation against TENSORCADE baseline |

## Key Concepts

### Dual FFT Processing

The system runs **two separate FFT pipelines** with different parameters:

| Pipeline | FFT Size | Dynamic Range | Purpose |
|----------|----------|---------------|---------|
| Inference | 4096 | 80 dB | Must match TENSORCADE model training |
| Waterfall | 8K-64K | 60 dB | High-resolution display |

The inference FFT parameters are locked to match the model. The waterfall FFT can be changed via Settings.

### Row-Strip Streaming

Instead of sending full video frames, the backend sends ~20 row strips per frame:

1. Backend computes GPU FFT → RGBA pixels
2. Sends strip (17-byte header + pixels + PSD dB values)
3. Flutter shifts its pixel buffer up and pastes new strip at bottom
4. Detection boxes positioned by absolute row index

Bandwidth: ~9 MB/s at 30fps (2048×20 strips).

### Waterfall Source Indicator

The waterfall shows which RX stream is feeding the display:

- **SCANNING** - RX1 hunting for signals
- **RX1 REC** - RX1 detected and is recording
- **RX2 REC** - RX2 collecting (handoff from RX1)
- **MANUAL** - Manual collection active

## Project Structure

```
g20_demo/
├── lib/                          # Flutter app
│   ├── core/
│   │   ├── services/
│   │   │   └── backend_launcher.dart   # Auto-starts Python backend
│   │   └── grpc/
│   │       └── connection_manager.dart # gRPC client
│   └── features/
│       ├── live_detection/
│       │   ├── providers/
│       │   │   └── video_stream_provider.dart  # WebSocket client
│       │   └── widgets/
│       │       └── video_waterfall_display.dart
│       └── settings/
│
├── backend/                      # Python backend
│   ├── server.py                 # WebSocket + gRPC server
│   ├── unified_pipeline.py       # FFT + inference pipeline
│   ├── gpu_fft.py               # CUDA FFT processing
│   └── inference.py             # TensorRT/PyTorch engine
│
├── protos/                       # gRPC protocol definitions
│   ├── control.proto            # SDR hardware control
│   └── inference.proto          # ML inference service
│
├── config/
│   └── spectrogram.yaml         # Canonical FFT parameters
│
├── models/                       # Trained .pth models
├── data/                         # IQ capture files (.sigmf)
└── docs/                         # Additional documentation
```

## Configuration

### Backend Connection

Flutter connects to backend via WebSocket for streaming data and gRPC for control commands:

- **WebSocket**: `ws://localhost:8765/ws/video` (row-strip streaming)
- **gRPC**: `localhost:50051` (device control, inference control)

Port is auto-discovered - the Python server prints `WS_PORT:XXXX` on startup.

### Spectrogram Parameters

Edit `config/spectrogram.yaml` for inference FFT settings:

```yaml
fft_size: 4096
overlap: 0.5
dynamic_range_db: 80.0  # Must match model training
```

Waterfall display FFT is controlled via Settings UI (8K to 64K).

## Build

```bash
# Development
flutter run -d windows

# Release
flutter build windows --release
# Output: build/windows/x64/runner/Release/g20_demo.exe
```

## Requirements

- Flutter SDK 3.24+
- Python 3.8+
- CUDA 11.8+ (for GPU FFT)
- PyTorch with CUDA support
- Windows: Visual Studio 2022 with C++ workload

## License

Proprietary - Internal Use Only
