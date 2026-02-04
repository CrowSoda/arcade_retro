# Migration Guide: Hydra → Crop Classifier

## Overview

This guide covers transitioning from the current Hydra (Faster R-CNN) system to the new two-stage crop classifier while maintaining backward compatibility.

---

## Migration Strategy: Parallel Operation

Run both systems simultaneously during transition:

```
┌─────────────────────────────────────────┐
│           Unified Pipeline              │
│                                         │
│  ┌─────────────┐   ┌─────────────────┐  │
│  │   Hydra     │   │ Crop Classifier │  │
│  │  (legacy)   │   │    (new)        │  │
│  └──────┬──────┘   └────────┬────────┘  │
│         │                   │           │
│         └─────────┬─────────┘           │
│                   │                     │
│           Feature Flag                  │
│                   │                     │
│                   ▼                     │
│              Detections                 │
└─────────────────────────────────────────┘
```

---

## Phase 1: Add Crop Classifier (Week 1-3)

### Step 1: Create Module Structure

```bash
mkdir -p backend/crop_classifier/{models,training,inference,labeling}
touch backend/crop_classifier/__init__.py
touch backend/crop_classifier/models/__init__.py
touch backend/crop_classifier/training/__init__.py
touch backend/crop_classifier/inference/__init__.py
touch backend/crop_classifier/labeling/__init__.py
```

### Step 2: Add Feature Flag

In `config/inference.yaml`:

```yaml
detection:
  # Options: 'hydra', 'crop_classifier', 'both'
  backend: hydra  # Start with Hydra only

  crop_classifier:
    enabled: false
    score_threshold: 0.5
    model_path: models/crop_classifier/classifier.pth
```

### Step 3: Update Unified Pipeline

```python
# In unified_pipeline.py

class UnifiedPipeline:
    def __init__(self, config_path: str = "config/inference.yaml"):
        self.config = load_config(config_path)

        self.hydra_detector = None
        self.crop_detector = None

        backend = self.config['detection']['backend']

        if backend in ('hydra', 'both'):
            from hydra import HydraDetector
            self.hydra_detector = HydraDetector(...)

        if backend in ('crop_classifier', 'both'):
            from crop_classifier.inference import TwoStageDetector
            self.crop_detector = TwoStageDetector(...)

    def detect(self, spectrogram):
        backend = self.config['detection']['backend']

        if backend == 'hydra':
            return self.hydra_detector.detect(spectrogram)

        elif backend == 'crop_classifier':
            return self.crop_detector.detect(spectrogram)

        elif backend == 'both':
            # Run both and log comparison
            hydra_results = self.hydra_detector.detect(spectrogram)
            crop_results = self.crop_detector.detect(spectrogram)

            self._log_comparison(hydra_results, crop_results)

            return hydra_results  # Return Hydra until validated
```

---

## Phase 2: Train Initial Model (Week 3-4)

### Step 1: Collect Initial Labels

Using existing training workflow:
1. Run blob detection on captures
2. Present crops to user via new labeling UI
3. Collect 25 initial labels

### Step 2: Train Siamese Model

```python
# Script: scripts/train_crop_classifier.py

from crop_classifier.models import SiameseNetwork
from crop_classifier.training import train_siamese
from crop_classifier.labeling import LabelStorage

# Load labels
storage = LabelStorage("training_data/crop_labels.db")
crops, labels = storage.get_all_labeled()

# Train
model = SiameseNetwork()
history = train_siamese(model, crops, labels)

# Save
torch.save(model.state_dict(), "models/crop_classifier/siamese.pth")

# Build gallery
gallery = build_signal_gallery(model, crops[labels == 1])
torch.save(gallery, "models/crop_classifier/gallery.pt")
```

### Step 3: Validate Model

```python
# Compare accuracy on held-out set
hydra_acc = evaluate_hydra(val_crops, val_labels)
crop_acc = evaluate_crop_classifier(val_crops, val_labels)

print(f"Hydra accuracy: {hydra_acc:.2%}")
print(f"Crop classifier accuracy: {crop_acc:.2%}")
```

---

## Phase 3: A/B Testing (Week 5)

### Enable Parallel Mode

```yaml
# config/inference.yaml
detection:
  backend: both  # Run both systems
```

### Log Comparison Metrics

```python
def _log_comparison(self, hydra: list, crop: list):
    """Log detection comparison for analysis."""
    comparison = {
        'timestamp': time.time(),
        'hydra_count': len(hydra),
        'crop_count': len(crop),
        'hydra_avg_conf': np.mean([d.confidence for d in hydra]) if hydra else 0,
        'crop_avg_conf': np.mean([d.confidence for d in crop]) if crop else 0,
        'overlap_count': self._count_overlapping(hydra, crop),
    }

    with open('logs/detection_comparison.jsonl', 'a') as f:
        f.write(json.dumps(comparison) + '\n')
```

### Analyze Results

```python
# Script: scripts/analyze_ab_test.py

import pandas as pd

df = pd.read_json('logs/detection_comparison.jsonl', lines=True)

print("Detection counts:")
print(f"  Hydra avg: {df['hydra_count'].mean():.1f}")
print(f"  Crop avg: {df['crop_count'].mean():.1f}")

print("\nConfidence:")
print(f"  Hydra avg: {df['hydra_avg_conf'].mean():.2%}")
print(f"  Crop avg: {df['crop_avg_conf'].mean():.2%}")

print("\nOverlap:")
print(f"  Avg overlapping: {df['overlap_count'].mean():.1f}")
```

---

## Phase 4: Cutover (Week 6)

### Prerequisites

Before switching:
- [ ] Crop classifier accuracy ≥ Hydra accuracy
- [ ] Latency within acceptable range (<50ms)
- [ ] All edge cases tested
- [ ] Rollback plan documented

### Step 1: Update Config

```yaml
# config/inference.yaml
detection:
  backend: crop_classifier  # Switch to new system
```

### Step 2: Monitor Closely

```python
# First 24 hours: extra logging
if self.config.get('migration_monitoring'):
    for detection in detections:
        if detection.confidence < 0.6:
            await self.flag_for_review(detection)
```

### Step 3: Rollback Plan

If issues arise:

```yaml
# Immediate rollback
detection:
  backend: hydra
```

---

## Data Migration

### Training Data

No migration needed - crop classifier uses different format:

| Hydra | Crop Classifier |
|-------|-----------------|
| Full spectrograms | Cropped regions |
| Bounding boxes | Binary labels |
| `training_data/signals/` | `training_data/crop_labels/` |

### Model Files

Keep Hydra models during transition:

```
models/
├── backbone/          # Keep - used by Hydra
├── heads/             # Keep - Hydra heads
├── crop_classifier/   # NEW - crop classifier
│   ├── siamese.pth
│   ├── classifier.pth
│   └── gallery.pt
└── registry.json      # Keep - Hydra registry
```

---

## API Changes

### WebSocket Commands

New commands added (existing unchanged):

| Command | Endpoint | Notes |
|---------|----------|-------|
| `get_crops` | `/crop_classifier` | Get blob crops for labeling |
| `submit_labels` | `/crop_classifier` | Record user labels |
| `train_classifier` | `/crop_classifier` | Train/retrain model |
| `start_session` | `/crop_classifier` | Start active learning |

Existing commands work unchanged:
- `train_signal` → Still works (Hydra)
- `load_heads` → Still works (Hydra)
- Detection via video stream → Uses backend from config

### Flutter Changes

New screens added:
- `SwipeLabelingScreen` - Mobile labeling
- `GridLabelingScreen` - Desktop labeling

Existing screens unchanged:
- Training screen still works with Hydra
- Live detection uses whichever backend is configured

---

## Rollback Procedures

### Immediate Rollback (< 5 minutes)

Change config and restart:

```bash
# Edit config/inference.yaml
# backend: crop_classifier → backend: hydra

# Restart backend
pkill -f "python server.py"
python server.py
```

### Full Rollback (remove crop classifier)

```bash
# 1. Switch config
# backend: hydra

# 2. Remove crop classifier code (optional)
rm -rf backend/crop_classifier/

# 3. Remove models (optional)
rm -rf models/crop_classifier/

# 4. Remove config section (optional)
# Edit config/inference.yaml to remove crop_classifier section
```

---

## Timeline Summary

| Week | Milestone | Hydra Status | Crop Classifier Status |
|------|-----------|--------------|------------------------|
| 1-2 | Foundation | Primary | In development |
| 3 | Initial training | Primary | Testing |
| 4 | Scale to CNN | Primary | Parallel testing |
| 5 | A/B testing | Parallel | Parallel |
| 6 | Cutover | Backup | Primary |
| 7+ | Cleanup | Deprecated | Primary |

---

## Deprecation Schedule

| Date | Action |
|------|--------|
| Week 6 | Crop classifier becomes primary |
| Week 8 | Hydra deprecated (still available) |
| Week 12 | Hydra code removal (if no issues) |
| Week 16 | Hydra model files cleanup |
