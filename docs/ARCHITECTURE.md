# G20 System Architecture

**Last Updated:** January 25, 2026

---

## Overview

G20 is a real-time RF signal detection system with a Flutter frontend and Python GPU-accelerated backend. The system processes IQ (In-phase/Quadrature) data through FFT to generate waterfalls, runs ML inference for signal detection, and streams results to the UI.

---

## System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                          G20 SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────┐    WebSocket     ┌─────────────────────────────┐  │
│  │               │    (8765)        │                             │  │
│  │   Flutter     │◄────────────────►│   Python Backend            │  │
│  │   Frontend    │                  │                             │  │
│  │               │    gRPC          │   • unified_pipeline.py     │  │
│  │               │◄────────────────►│   • gpu_fft.py              │  │
│  │               │    (50051)       │   • inference.py            │  │
│  │               │                  │   • server.py               │  │
│  └───────────────┘                  └─────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. IQ Processing Pipeline

```
IQ File (.iq, .bin)
       │
       ▼
┌──────────────────────┐
│  GPU FFT Processor   │  gpu_fft.py - GPUSpectrogramProcessor
│  • cuFFT kernels     │  • 8K/16K/32K/64K point FFT
│  • Welch averaging   │  • Overlap-add processing
│  • dB normalization  │  • Auto noise floor tracking
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Colormap LUT        │  viridis/plasma/inferno/magma/turbo
│  • dB → RGB          │  256-entry lookup tables
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Row Strip Encoder   │  RGBA rows (2048px × ~38 rows per frame)
│  • Binary header     │  frame_id, total_rows, pts, source_id
│  • RGBA pixels       │  
│  • PSD float32       │  Power spectral density per bin
└──────────────────────┘
       │
       ▼  WebSocket (0x01 STRIP message)
┌──────────────────────┐
│  Flutter Client      │  video_stream_provider.dart
│  • Pixel buffer      │  Scroll + paste new rows
│  • RawImage render   │  Direct RGBA → texture
│  • PSD chart         │  Latest row dB values
└──────────────────────┘
```

### 2. Detection Pipeline

```
FFT Spectrogram
       │
       ▼
┌──────────────────────┐
│  TensorRT Inference  │  inference.py
│  • YOLOv8 model      │  • Signal classification
│  • FP16 optimized    │  • Bounding box detection
│  • Batch processing  │  • Confidence scores
└──────────────────────┘
       │
       ▼  WebSocket (0x02 DETECTION message)
┌──────────────────────┐
│  Detection Overlay   │  detection_provider.dart
│  • Row-based coords  │  absoluteRow + rowSpan
│  • Lifecycle mgmt    │  Prune when scrolled off
│  • Table display     │  detection_table.dart
└──────────────────────┘
```

---

## Communication Protocols

### WebSocket Protocol (Port 8765)

Binary message format with 1-byte type prefix:

| Type | Value | Description |
|------|-------|-------------|
| STRIP | 0x01 | RGBA row strip with header |
| DETECTION | 0x02 | Detection JSON payload |
| METADATA | 0x03 | Stream configuration |

#### Strip Message Header (17 bytes)
```
Offset  Size  Type     Description
0       4     uint32   frame_id
4       4     uint32   total_rows (monotonic counter)
8       2     uint16   rows_in_strip (typically 38)
10      2     uint16   strip_width (typically 2048)
12      4     float32  pts (presentation timestamp)
16      1     uint8    source_id (0=SCAN, 1=RX1_REC, 2=RX2_REC, 3=MANUAL)
17+     N×4   RGBA     pixel data (rows_in_strip × strip_width × 4)
17+N    W×4   float32  PSD dB values (strip_width floats)
```

#### Commands (JSON via WebSocket)
```json
{"command": "set_fps", "fps": 30}
{"command": "set_fft_size", "size": 65536}
{"command": "set_colormap", "colormap": 0}
{"command": "set_db_range", "min_db": -100, "max_db": -20}
{"command": "set_time_span", "seconds": 2.5}
{"command": "set_score_threshold", "threshold": 0.9}
```

### gRPC Protocol (Port 50051)

Used for device control commands (tuning, capture control):

- `DeviceControlService` - SDR tuning, bandwidth, gain
- `InferenceService` - Model loading, inference control

---

## State Management (Flutter)

### Riverpod Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROVIDER HIERARCHY                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Settings Providers (settings_providers.dart)                    │
│  ├── waterfallFpsProvider          StateProvider<int>            │
│  ├── waterfallFftSizeProvider      StateProvider<int>            │
│  ├── waterfallColormapProvider     StateProvider<int>            │
│  ├── scoreThresholdProvider        StateProvider<double>         │
│  └── ...                                                         │
│                                                                  │
│  Stream Providers (video_stream_provider.dart)                   │
│  ├── videoStreamProvider           StateNotifierProvider         │
│  │   ├── pixelBuffer: Uint8List    (2048 × 2850 × 4 bytes)      │
│  │   ├── detections: List          (current visible)             │
│  │   ├── totalRowsReceived: int    (monotonic counter)          │
│  │   └── waterfallSource: enum     (SCAN/RX1_REC/RX2_REC)       │
│  │                                                               │
│  Detection Providers (detection_provider.dart)                   │
│  ├── detectionProvider             StateNotifierProvider         │
│  │   └── List<Detection>           (all active detections)      │
│  │                                                               │
│  UI State Providers (various)                                    │
│  ├── rightPanelCollapsedProvider   StateProvider<bool>          │
│  ├── displayModeProvider           derived from mapStateProvider│
│  └── activeMissionProvider         StateProvider<Mission?>       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

### Flutter Frontend
```
lib/
├── main.dart                 # Entry point
├── app.dart                  # MaterialApp configuration
├── core/
│   ├── config/
│   │   └── theme.dart        # G20Colors, dark theme
│   ├── services/
│   │   └── backend_launcher.dart  # Auto-start backend
│   └── widgets/
│       └── dialogs.dart      # Shared toasts, dialogs
├── features/
│   ├── shell/
│   │   └── app_shell.dart    # Navigation rail, status bar
│   ├── live_detection/
│   │   ├── live_detection_screen.dart
│   │   ├── providers/
│   │   │   ├── video_stream_provider.dart
│   │   │   ├── detection_provider.dart
│   │   │   ├── waterfall_provider.dart
│   │   │   └── ...
│   │   └── widgets/
│   │       ├── video_waterfall_display.dart
│   │       ├── psd_chart.dart
│   │       ├── detection_table.dart
│   │       └── inputs_panel.dart
│   ├── config/
│   │   ├── config_screen.dart    # Mission editor
│   │   ├── providers/
│   │   └── widgets/
│   │       └── mission_picker_dialog.dart
│   └── settings/
│       ├── settings_screen.dart
│       └── providers/
│           └── settings_providers.dart
```

### Python Backend
```
backend/
├── server.py                 # Entry point, WebSocket + gRPC
├── unified_pipeline.py       # Main waterfall processing
├── gpu_fft.py               # GPUSpectrogramProcessor (cuFFT)
├── inference.py             # TensorRT YOLOv8 inference
├── waterfall_buffer.py      # Row buffer management
└── requirements.txt
```

---

## Key Design Decisions

### 1. Row-Strip Streaming
Instead of full-frame video, we stream narrow horizontal strips (38 rows). This:
- Reduces latency (~33ms per strip at 30fps)
- Enables smoother scrolling
- Allows row-indexed detection positioning

### 2. Client-Side Pixel Buffer
The Flutter client maintains its own RGBA pixel buffer and scrolls/pastes strips. This:
- Eliminates server-side buffer state
- Reduces bandwidth (no repeated pixels)
- Enables arbitrary buffer height

### 3. Absolute Row Indexing
Detections are positioned by `absoluteRow` (monotonic counter) rather than pixel Y. This:
- Survives buffer scrolling
- Enables efficient pruning
- Decouples detection timing from display

### 4. Dual Protocol (WebSocket + gRPC)
- **WebSocket** for high-bandwidth streaming (waterfall, detections)
- **gRPC** for low-latency control commands (tuning, model switching)

---

## Performance Considerations

### GPU FFT
- cuFFT kernel warmup: 100-500ms on first call per size
- FFT sizes: 8K (~2ms), 16K (~4ms), 32K (~6ms), 64K (~10ms)
- Batch processing of 660K samples per frame

### Pixel Buffer
- Buffer size: 2048 × 2850 × 4 = ~23 MB
- Scroll operation: memmove ~1ms per strip
- RawImage → texture: ~1ms

### Target Performance
- 30 FPS sustained waterfall
- <100ms detection latency
- <50 MB client memory

---

## Future Improvements

1. **WebGPU Rendering** - Move pixel buffer to GPU for zero-copy
2. **JPEG Compression** - Option for low-bandwidth mode
3. **Multi-RX Fusion** - Combine RX1/RX2 in single view
4. **Recording Playback** - Scrub through captured IQ files

---

*Architecture document for G20 RF Detection System*
