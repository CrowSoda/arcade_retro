# G20 Demo Backend

TensorRT-optimized inference server for the G20 RF detection platform.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install PyTorch with CUDA (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Generate gRPC stubs
generate_stubs.bat

# 4. Run server
python server.py --port 50051
```

## Structure

```
backend/
├── inference.py      # TensorRT/PyTorch engine with multi-model support
├── server.py         # gRPC server (DeviceControl + InferenceService)
├── requirements.txt
├── generate_stubs.bat
└── generated/        # Generated proto stubs (after running generate_stubs.bat)
```

## Features

- **Auto-fallback**: TensorRT → ONNX → PyTorch
- **Multi-model parallel**: CUDA streams for N models simultaneously
- **GPU-accelerated spectrogram**: cuFFT via PyTorch
- **gRPC streaming**: 30 FPS detection frames

## Benchmarking

```bash
# Single model benchmark
python inference.py --model ../models/detector.pth --benchmark

# Multi-model parallel benchmark
python inference.py --model ../models/detector.pth --benchmark-multi 6

# Export to TensorRT (requires TensorRT installed)
python inference.py --model ../models/detector.pth --export-trt
```

## Target Hardware

| Platform | Expected Performance |
|----------|---------------------|
| RTX 4090 (dev) | ~3-4ms/inference (PyTorch FP16) |
| Orin NX (G20) | ~2-3ms/inference (TensorRT FP16) |
