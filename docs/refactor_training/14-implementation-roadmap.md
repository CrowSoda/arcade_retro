# Implementation Roadmap

## Overview

6-week implementation plan transitioning from Faster R-CNN to crop-based classification.

---

## ✅ Pre-Work: Blob Validation (COMPLETE)

**Blob detection validated on Creamy_Shrimp dataset (48 samples).**

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| IoU Recall | ≥90% | 91.7% | ✅ PASS |
| Blob Count | 50-500 | 120 | ✅ PASS |
| Missed | <10% | 8.3% | ✅ PASS |

**Known limitation:** 4 samples (8.3%) cannot be detected due to signal fragmentation. This is a fundamental ceiling for contour-based detection.

---

## 🔒 Blob Detection Config (LOCKED)

**Do not change. Use this exact config:**

```python
min_area = 50
max_area = 5000
min_aspect_ratio = 1.5
max_aspect_ratio = 15.0
block_size = 51
C = -5
```

If you modify these numbers, re-run validation with `scripts/test_blob_recall.py`.

---

## ✅ Week 1: Foundation (COMPLETE)

**Completed:** 2026-02-01

### Goals ✅
- ✅ Set up crop classifier module structure
- ✅ Implement blob detector wrapper
- ✅ Build Siamese network architecture

### Python Backend Tasks ✅

| Task | File | Status |
|------|------|--------|
| Directory structure | `backend/crop_classifier/` | ✅ Created |
| Blob detector | `inference/blob_detector.py` | ✅ BlobDetector with locked config |
| Preprocessor | `inference/preprocessor.py` | ✅ 32×64 letterbox, per-crop normalization |
| Siamese encoder | `models/siamese.py` | ✅ SiameseEncoder, SiameseNetwork, SiameseClassifier |
| Losses | `models/losses.py` | ✅ ContrastiveLoss, FocalLoss, TripletLoss |
| Config | `config/crop_classifier.yaml` | ✅ Full config with locked blob params |

### Testing Results ✅

| Test | Result |
|------|--------|
| Blob detector | ✅ 38 tests passing |
| Preprocessor | ✅ Outputs shape (1, 32, 64) |
| Siamese encoder | ✅ Forward pass, L2-normalized embeddings |
| Integration | ✅ boxes → crops → embeddings pipeline |

**Note:** Blob detection with locked config requires real RF spectrograms. Tests use manually specified boxes for synthetic data.

### Files Created

```
backend/crop_classifier/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── siamese.py        # SiameseEncoder, SiameseNetwork, SiameseClassifier
│   └── losses.py         # ContrastiveLoss, FocalLoss, TripletLoss
├── inference/
│   ├── __init__.py
│   ├── blob_detector.py  # BlobDetector with locked config
│   └── preprocessor.py   # CropPreprocessor, 32×64 letterbox
├── training/
│   └── __init__.py       # Placeholder for Week 3
└── labeling/
    └── __init__.py       # Placeholder for Week 3

config/
└── crop_classifier.yaml  # Full configuration

backend/tests/
└── test_crop_classifier.py  # 38 tests
```

---

## Week 2: Integrate with Existing Training UI (REVISED)

**Status:** Plan changed - using existing bounding box UI instead of new swipe UI

### Original Plan (DISCARDED)
The original plan called for a Tinder-style swipe labeling screen. This was built but discarded because:
- User already has a working training UI with bounding boxes
- No need to create a separate labeling workflow

### New Approach
Integrate crop classifier with the **existing training screen**:

1. **Same UI** - User draws bounding boxes on spectrogram (no change)
2. **Backend change** - Training uses those boxes to train Siamese model (not Faster R-CNN)
3. **Inference change** - Blob detection + Siamese classification replaces Faster R-CNN

### What Already Exists
The existing training UI (`lib/features/training/`) already supports:
- Drawing bounding boxes on spectrograms
- Saving labeled samples
- Training presets (Fast, Balanced, Quality, Extreme)
- Labels list with frequency ranges

### Next Steps
Week 3 will:
- Add backend `train_crop_classifier` WebSocket command
- Extract crops from user's bounding boxes
- Train Siamese model using those crops
- No Flutter UI changes needed

---

## Week 3: Initial Training Pipeline

### Goals
- Train first Siamese model with 25 labels
- Implement active learning selection
- Deploy pseudo-labeling

### Tasks

**Training:**
- [ ] Implement `training/trainer.py` - Siamese training loop
- [ ] Implement pair generation from labeled data
- [ ] Add contrastive loss training
- [ ] Implement gallery building for inference

**Active Learning:**
- [ ] Implement `training/active_learning.py` - Hybrid query strategy
- [ ] Add uncertainty scoring + diversity clustering
- [ ] Implement `labeling/session.py` - Session state management

**Pseudo-Labeling:**
- [ ] Implement `training/pseudo_labeling.py`
- [ ] Add threshold decay logic
- [ ] Integrate with training loop

### Deliverables
- Trained Siamese model from 25 labels
- Active learning loop selecting informative samples

---

## Week 4: Scale to Direct CNN

### Goals
- Transition to direct classifier at 100+ labels
- Implement full augmentation pipeline

### Tasks

**Models:**
- [ ] Implement `models/classifier.py` - Direct CNN
- [ ] Add focal loss to `models/losses.py`
- [ ] Implement model switching (Siamese → CNN at 100+ labels)

**Augmentation:**
- [ ] Implement `training/augmentation.py`
- [ ] Frequency shift, time shift, power scaling, noise injection, SpecAugment

### Critical Reminder
- **Siamese first, CNN later** — Siamese for 25-100 labels, switch to direct CNN at 100+

---

## Week 5: Inference Pipeline

### Goals
- Build complete two-stage detector
- Optimize for <50ms latency

### Tasks

**Inference:**
- [ ] Implement `inference/detector.py` - TwoStageDetector
- [ ] Implement `inference/batch_processor.py` - Async batching
- [ ] Add NMS post-processing
- [ ] Add timing instrumentation

**Integration:**
- [ ] Add WebSocket commands for crop classifier
- [ ] Add feature flag: Hydra vs. crop classifier
- [ ] Update `unified_pipeline.py`

---

## Week 6: Production Hardening

### Goals
- Full integration testing
- Documentation
- Monitoring

### Tasks
- [ ] End-to-end tests with real RFCAP files
- [ ] A/B comparison: Hydra vs. crop classifier
- [ ] Update g20.md with crop classifier section
- [ ] Document the 8.3% ceiling (4 fragmented signals cannot be detected)

---

## Success Metrics

| Checkpoint | Criteria |
|------------|----------|
| Week 2 | Siamese compiles, Flutter displays crops |
| Week 3 | 75%+ accuracy with 25 labels |
| Week 5 | <50ms latency, 85%+ accuracy with 100+ labels |
| Week 6 | All tests passing, docs complete |

---

## Dependencies

### Python
```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.0
opencv-python>=4.5
cleanlab>=2.0
```

### Flutter
No new Flutter dependencies needed - using existing training UI.
