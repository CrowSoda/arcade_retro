# G20 Training & Inference Architecture

## Overview

This document outlines the training and inference pipeline for the G20 RF detection platform.

---

## Auto-Tuning Training System

Based on available labeled data, automatically calculate optimal hyperparameters:

### Heuristics Table

| Data Size | Epochs | K-Folds | Batch Size | Learning Rate | Early Stop |
|-----------|--------|---------|------------|---------------|------------|
| < 50      | 100    | none    | 4          | 1e-4          | 5          |
| 50-100    | 75     | 3       | 4          | 1e-4          | 5          |
| 100-200   | 75     | 5       | 8          | 3e-4          | 7          |
| 200-500   | 50     | 5       | 8          | 3e-4          | 10         |
| 500-2000  | 30     | 7       | 16         | 3e-4          | 12         |
| 2000+     | 20     | 10      | 16         | 3e-4          | 15         |

### Python Implementation

```python
def auto_tune_training(num_labeled_samples: int, num_classes: int) -> dict:
    """
    Auto-calculate training parameters based on available data.
    """
    
    # EPOCHS: More data = fewer epochs needed
    if num_labeled_samples < 50:
        epochs = 100
    elif num_labeled_samples < 200:
        epochs = 75
    elif num_labeled_samples < 500:
        epochs = 50
    elif num_labeled_samples < 2000:
        epochs = 30
    else:
        epochs = 20
    
    # K-FOLDS: Small data = fewer folds, large data = more folds
    if num_labeled_samples < 30:
        k_folds = 0  # Simple train/val split
    elif num_labeled_samples < 100:
        k_folds = 3
    elif num_labeled_samples < 300:
        k_folds = 5
    elif num_labeled_samples < 1000:
        k_folds = 7
    else:
        k_folds = 10
    
    # BATCH SIZE: Based on GPU memory and data size
    if num_labeled_samples < 100:
        batch_size = 4
    elif num_labeled_samples < 500:
        batch_size = 8
    else:
        batch_size = 16
    
    # LEARNING RATE: Conservative for small datasets
    lr = 1e-4 if num_labeled_samples < 100 else 3e-4
    
    # EARLY STOP PATIENCE
    early_stop = max(5, min(15, num_labeled_samples // 20))
    
    return {
        "epochs": epochs,
        "k_folds": k_folds,
        "batch_size": batch_size,
        "learning_rate": lr,
        "early_stop_patience": early_stop,
        "estimated_time_min": (epochs * num_labeled_samples * 0.02) / 60
    }
```

---

## Multi-Model Inference Architecture

Run multiple detection models simultaneously for comprehensive signal coverage.

### Architecture Diagram

```
                    ┌─→ [TRT Engine A (Stream 1)] ─→ det_A ─┐
                    │     (creamy_chicken)                  │
IQ → STFT → Tile ──├─→ [TRT Engine B (Stream 2)] ─→ det_B ─├─→ NMS Merge → Final
                    │     (wifi_bluetooth)                  │    Detections
                    └─→ [TRT Engine C (Stream 3)] ─→ det_C ─┘
                          (lte_uplink)
```

### Key Design Points

1. **Shared Preprocessing**: Single STFT computation shared by all models
2. **Parallel CUDA Streams**: Each model runs in its own stream
3. **Latency = Max(models)**: Not sum, since they run in parallel
4. **NMS Merge**: Non-max suppression to dedupe overlapping detections

### C++ Implementation

```cpp
class MultiModelInferencer {
    std::vector<std::unique_ptr<TrtEngine>> engines_;
    std::vector<cudaStream_t> streams_;
    
public:
    void loadModels(const std::vector<std::string>& engine_paths) {
        for (const auto& path : engine_paths) {
            engines_.push_back(TrtEngine::load({path}));
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            streams_.push_back(stream);
        }
    }
    
    std::vector<Detection> inferAll(const InferenceTile& tile) {
        std::vector<Detection> all_detections;
        
        // Launch all inferences in parallel
        for (size_t i = 0; i < engines_.size(); i++) {
            engines_[i]->inferAsync(tile, streams_[i]);
        }
        
        // Synchronize and collect
        for (size_t i = 0; i < engines_.size(); i++) {
            cudaStreamSynchronize(streams_[i]);
            auto dets = engines_[i]->getResults();
            for (auto& d : dets) d.model_id = i;
            all_detections.insert(all_detections.end(), dets.begin(), dets.end());
        }
        
        // NMS merge overlapping detections
        return nonMaxSuppression(all_detections, /*iou_threshold=*/0.5);
    }
};
```

---

## Training Pipeline (Python)

### Why Python for Training?

| Aspect | Python | C++ (LibTorch) |
|--------|--------|----------------|
| Ecosystem | PyTorch, WandB, HuggingFace | Limited |
| Debugging | Easy | Painful |
| Numerical stability | Rock solid | Known issues |
| Development speed | Hours | Days |

### Training Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                     │
│    SigMF IQ + Labels → ChunkWorker → Spectrogram PNG    │
│                                    → BBox JSON          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. AUTO-TUNE                                            │
│    Count samples → Calculate epochs, k-folds, etc.      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. TRAINING                                             │
│    Faster R-CNN (ResNet18/50 + FPN)                     │
│    K-Fold Cross-Validation                              │
│    Early Stopping                                       │
│    Output: model.pth                                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. EXPORT                                               │
│    PyTorch → ONNX → TensorRT Engine (.trt)              │
└─────────────────────────────────────────────────────────┘
```

---

## Inference Pipeline (C++ TensorRT)

### Why C++ for Inference?

- **3-5x faster** than PyTorch
- **Lower memory** footprint
- **No Python GIL** issues
- **Production ready** for embedded (Jetson)

### Inference Flow

```
┌─────────────────────────────────────────────────────────┐
│ IQ STREAM (from SDR or file playback)                   │
│ → 200ms chunks                                          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ GPU PREPROCESSING (cuFFT)                               │
│ → STFT (nfft=4096, hop=2048)                           │
│ → Power spectrum (dB)                                   │
│ → Normalize (dynamic range)                             │
│ → Resize to 1024x1024                                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ TENSORRT INFERENCE                                      │
│ → Load .trt engine                                      │
│ → Batch processing (4-8 tiles)                          │
│ → Deadline: 400ms max                                   │
│ → Output: boxes + scores + class IDs                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ POST-PROCESSING                                         │
│ → Score threshold filter (>0.5)                         │
│ → NMS (IoU 0.5)                                         │
│ → Track assignment                                      │
│ → Emit detections to GUI                                │
└─────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### Faster R-CNN with ResNet18-FPN

```
Input: 1024x1024x3 spectrogram (grayscale duplicated)
       │
       ▼
┌─────────────────────────┐
│ ResNet18 Backbone       │
│ (pretrained or scratch) │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Feature Pyramid Network │
│ (multi-scale features)  │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Region Proposal Network │
│ (candidate boxes)       │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ ROI Heads               │
│ → Box regression        │
│ → Classification        │
└─────────────────────────┘
       │
       ▼
Output: [boxes, scores, labels]
```

---

## Flutter Training UI Requirements

### Config Form
- Dataset selection (multi-select RFCAP/SigMF files)
- Auto-calculated parameters (display recommended values)
- Override toggles for manual control
- Estimated training time display

### Progress Display
- Epoch progress bar
- Loss curve chart (real-time)
- Validation metrics (Precision, Recall, F1)
- GPU utilization meter

### Job Management
- Start/Pause/Cancel training
- Job history with results
- Model comparison table

### Model Export
- "Export to TensorRT" button
- Engine file download/save
- Validation test on export

---

## File Formats

### Training Data (from ChunkWorker)
```
spec_image/YYYYMMDD_HHMMSS/
├── chunk_filename_000.png      # 1024x1024 grayscale spectrogram
├── chunk_filename_000_bboxes.json
│   {
│     "image": "chunk_filename_000.png",
│     "width": 1024,
│     "height": 1024,
│     "bboxes": [
│       {"label": "creamy_chicken", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}
│     ]
│   }
├── chunk_filename_001.png
├── chunk_filename_001_bboxes.json
└── ...
```

### Model Checkpoint
```
models/
├── creamy_chicken_fold1.pth
├── creamy_chicken_fold2.pth
├── creamy_chicken_fold3.pth
└── creamy_chicken_best.trt     # TensorRT engine (after export)
```

---

## Next Steps

1. [ ] Implement auto-tune function in `tensorcade/config.py`
2. [ ] Add multi-model support to `InferenceWorker`
3. [ ] Build Flutter Training UI (Phase 6 from gui_roadmap.md)
4. [ ] Create TensorRT export script (`torch2trt.py`)
5. [ ] Integration test: Training → Export → Inference
