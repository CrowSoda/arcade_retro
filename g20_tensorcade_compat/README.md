# Using Your TENSORCADE Model with G20 Demo

## The Goal

You have:
- A trained Faster R-CNN model (`modern_burst_gap_fold3.pth`)
- IQ test data it works on
- A new G20 demo system

You want:
- **Verify the model works** in the new system
- **Generate baseline detections** from known-good model
- **Compare G20 output** against baseline to ensure parity

## Quick Start

### Step 1: Verify Model Loads

```bash
cd g20_demo/g20_tensorcade_compat

# Test with your model
python tensorcade_wrapper.py ../models/modern_burst_gap_fold3.pth

# Expected output:
# ✓ Model loaded successfully
# ✓ Inference works
#   - Spectrogram shape: (1024, 1024)
#   - Detections: N
```

### Step 2: Test with Real IQ Data

```bash
python tensorcade_wrapper.py \
    ../models/modern_burst_gap_fold3.pth \
    ../data/your_test.sigmf-data \
    20e6  # sample rate
```

### Step 3: Generate Baseline Detections

```bash
python test_detection_baseline.py generate \
    --model ../models/modern_burst_gap_fold3.pth \
    --iq-dir ../data/test_iq/ \
    --output baselines/ \
    --sample-rate 20e6

# Creates:
#   baselines/baseline_detections.json
#   baselines/spectrograms/*.npy
```

### Step 4: Compare G20 Against Baseline

After running G20 on the same IQ files:

```bash
python test_detection_baseline.py compare \
    --baseline baselines/baseline_detections.json \
    --g20-results g20_output/detections.json

# Expected: F1 > 0.95 to pass
```

## Integrating with G20 Pipeline

### Option A: Direct PyTorch (Simplest)

Just use the wrapper in your G20 pipeline:

```python
from g20_tensorcade_compat.tensorcade_wrapper import TensorcadeWrapper, TensorcadeConfig

# Load once at startup
config = TensorcadeConfig(
    score_threshold=0.5,  # Adjust as needed
)
detector = TensorcadeWrapper("models/modern_burst_gap_fold3.pth", config)

# In your pipeline loop
def inference_callback(frame):
    detections, _ = detector.detect(frame.iq_samples, frame.frame_id)
    return detections

pipeline = Pipeline(
    iq_source=your_source,
    inference_callback=inference_callback,
)
```

### Option B: TensorRT (Later Optimization)

TensorRT conversion for Faster R-CNN is tricky. For now, just use PyTorch.
The model runs at ~15 FPS on GPU, which is enough for testing.

If you need faster inference later, consider:
1. Training a YOLOv8 model on the same data
2. Using ONNX Runtime instead of TensorRT

## Key Configuration Match

These MUST match between TENSORCADE and G20:

| Parameter | TENSORCADE Value | G20 Config |
|-----------|------------------|------------|
| FFT size | 4096 | `fft_size: 4096` |
| Overlap | 2048 (50%) | `overlap: 0.5` |
| Dynamic range | **80 dB** | `dynamic_range_db: 80.0` |
| Output size | 1024×1024 | `target_width: 1024` |
| Window | Hann | `window: hann` |

**CRITICAL:** If G20 uses 60dB dynamic range but the model was trained on 80dB,
detections will be wrong. Make sure these match!

## File Structure

```
g20_demo/
├── g20_tensorcade_compat/
│   ├── tensorcade_wrapper.py      # Model wrapper (matches TENSORCADE preprocessing)
│   ├── test_detection_baseline.py # Test suite
│   └── README.md                  # This file
│
├── models/
│   └── modern_burst_gap_fold3.pth  # Your trained model
│
├── data/
│   └── test_iq/              # Your IQ test files
│       ├── test1.sigmf-data
│       └── test1.sigmf-meta
│
└── baselines/
    ├── baseline_detections.json
    └── spectrograms/
        └── *.npy
```

## Troubleshooting

### "Model architecture mismatch"

The wrapper assumes ResNet18 backbone. If your model used ResNet50:

```python
config = TensorcadeConfig(backbone="resnet50")
wrapper = TensorcadeWrapper(model_path, config)
```

### "No detections on known-good data"

Check dynamic range. If model was trained on 80dB spectrograms but you're
feeding it 60dB normalized data, it won't work.

### "Detections are offset"

Check the vertical flip. TENSORCADE flips the spectrogram before inference
(line 1435 in workers.py). The wrapper handles this, but if you're
preprocessing separately, make sure to include the flip.

## What This Proves

When the comparison test passes (F1 > 0.95):

1. **Model loads correctly** in G20 environment
2. **Preprocessing matches** TENSORCADE exactly
3. **Detections are equivalent** to known-good baseline
4. **G20 display will be accurate** (boxes match signals)

This is your ground truth for the G20 system.

## Commands Reference

```bash
# Verify model
python tensorcade_wrapper.py <model.pth>

# Generate baseline
python test_detection_baseline.py generate \
    --model <model.pth> \
    --iq-dir <iq_directory> \
    --output <output_dir>

# Compare results
python test_detection_baseline.py compare \
    --baseline <baseline.json> \
    --g20-results <g20_results.json>

# Self-consistency test
python test_detection_baseline.py test \
    --model <model.pth> \
    --iq-dir <iq_directory>
```
