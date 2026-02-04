# Architecture Overview: Two-Stage Detection

## High-Level Flow

```
Spectrogram (1024×1024)
         │
         ▼
┌─────────────────────┐
│  Stage 1: Blob      │  Classical (no ML)
│  Detection          │  Otsu + Region Growing
│  (existing code)    │  ~5-10ms CPU
└─────────┬───────────┘
          │
          │ List[BoundingBox]
          ▼
┌─────────────────────┐
│  Crop Extraction    │  Add 15% padding
│  + Preprocessing    │  Resize to 64×64
│                     │  Normalize
└─────────┬───────────┘
          │
          │ List[Tensor(64×64)]
          ▼
┌─────────────────────┐
│  Stage 2: CNN       │  Learned
│  Classifier         │  Binary: signal/not-signal
│                     │  ~10-20ms GPU (batched)
└─────────┬───────────┘
          │
          │ List[(bbox, confidence)]
          ▼
┌─────────────────────┐
│  Post-processing    │  Threshold filter
│  (NMS optional)     │  ~1-2ms CPU
└─────────┬───────────┘
          │
          ▼
    Final Detections
```

---

## Why This Architecture Achieves Position Invariance

### Three Mechanisms

1. **Crop extraction removes absolute context**
   - Each crop is extracted and centered
   - Classifier never sees position in original spectrogram
   - No "distance from border" information preserved

2. **All crops share the same spatial reference frame**
   - Object always centered in 64×64 window
   - Model learns signal appearance, not location
   - Same feature extraction regardless of origin position

3. **Crops don't include original image borders**
   - Zero-padding happens on crop, not original
   - Border information is relative to crop, not spectrogram
   - Position encoding is crop-relative (benign)

---

## Component Responsibilities

### Stage 1: Blob Detection (Localization)

| Responsibility | Implementation |
|----------------|----------------|
| Find all potential signals | Otsu thresholding |
| Determine bounding boxes | Region growing / connected components |
| Handle overlapping signals | None (leave to classifier confidence) |
| Position bias | **None** - purely classical |

**Output:** List of bounding boxes with NO confidence scores

### Stage 2: Crop Classifier (Recognition)

| Responsibility | Implementation |
|----------------|----------------|
| Determine if crop is target signal | Binary CNN |
| Provide confidence score | Sigmoid output |
| Learn signal characteristics | Trained on labeled crops |
| Position bias | **None** - never sees position |

**Output:** Confidence score per bounding box

---

## Data Flow Detail

```python
# Pseudocode for full pipeline

def detect_signals(spectrogram: np.ndarray) -> List[Detection]:
    # Stage 1: Classical blob detection
    bboxes = blob_detector.detect(spectrogram)  # ~500 candidates

    # Crop extraction
    crops = []
    for bbox in bboxes:
        crop = extract_crop(spectrogram, bbox, padding=0.15)
        crop = resize_letterbox(crop, target_size=(64, 64))
        crop = normalize(crop)
        crops.append(crop)

    # Stage 2: Batch classification
    batch = torch.stack(crops)
    confidences = classifier(batch)  # Forward pass

    # Filter by threshold
    detections = []
    for bbox, conf in zip(bboxes, confidences):
        if conf >= threshold:
            detections.append(Detection(bbox, conf))

    return detections
```

---

## Memory and Latency Budget

| Component | Memory | Latency | Notes |
|-----------|--------|---------|-------|
| Blob detector | ~10MB | 5-10ms | CPU, no GPU needed |
| Crop preprocessing | ~50MB | 1-2ms | NumPy/OpenCV |
| CNN classifier | ~5MB | 10-20ms | GPU, batched |
| Post-processing | ~1MB | 1-2ms | CPU threshold + NMS |
| **Total** | **~66MB** | **~30ms** | vs 115MB/35ms current |

---

## Comparison to Current Hydra System

| Aspect | Hydra (Faster R-CNN) | Two-Stage |
|--------|---------------------|-----------|
| Proposal generation | Learned (RPN) | Classical (Otsu) |
| Position bias | Yes (zero-padding) | No (crops centered) |
| Multi-class | Separate heads | Single binary (repeat) |
| Training data | 50-100+ samples | 25 samples viable |
| Backbone | ResNet-18 FPN (11M params) | None needed |
| Head | ROI + classifier | 3-layer CNN (~100K params) |

---

## Scalability Path

### Phase 1: Single Signal (25 labels)
- Train Siamese network for similarity
- Binary: "is this my signal?"
- Achieves 75-85% accuracy

### Phase 2: Multiple Signals (100+ labels per signal)
- Separate classifier per signal type
- OR multi-class single classifier
- Achieves 90%+ accuracy

### Phase 3: Production (500+ labels)
- Full CNN with focal loss
- Hard negative mining
- Achieves 95%+ accuracy
