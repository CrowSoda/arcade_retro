# Code Structure

## Directory Layout

```
g20_demo/
├── backend/
│   ├── crop_classifier/           # NEW: Crop-based classification
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── siamese.py         # Siamese network for few-shot
│   │   │   ├── classifier.py      # Direct CNN classifier
│   │   │   └── losses.py          # Focal loss, contrastive loss
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py         # Training loops
│   │   │   ├── augmentation.py    # Spectrogram augmentations
│   │   │   ├── active_learning.py # Query strategies
│   │   │   └── pseudo_labeling.py # Self-training logic
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── detector.py        # Two-stage detector
│   │   │   ├── blob_detector.py   # Classical blob detection
│   │   │   ├── preprocessor.py    # Crop extraction/normalization
│   │   │   └── batch_processor.py # Async GPU batching
│   │   └── labeling/
│   │       ├── __init__.py
│   │       ├── session.py         # Active learning session
│   │       └── storage.py         # Label persistence
│   │
│   ├── hydra/                     # EXISTING (keep for now)
│   │   ├── config.py
│   │   ├── detector.py
│   │   └── version_manager.py
│   │
│   ├── training/                  # EXISTING (keep for now)
│   │   ├── sample_manager.py
│   │   ├── service.py
│   │   └── splits.py
│   │
│   ├── server.py                  # WebSocket server
│   ├── unified_pipeline.py        # FFT pipeline
│   └── api/
│       └── crop_routes.py         # NEW: Crop classifier endpoints
│
├── lib/
│   └── features/
│       └── labeling/              # NEW: Flutter labeling UI
│           ├── screens/
│           │   ├── swipe_labeling_screen.dart
│           │   └── grid_labeling_screen.dart
│           ├── widgets/
│           │   ├── crop_card.dart
│           │   └── confidence_badge.dart
│           ├── providers/
│           │   └── labeling_provider.dart
│           └── models/
│               └── crop_data.dart
│
├── models/
│   ├── backbone/                  # EXISTING: Shared backbone
│   ├── heads/                     # EXISTING: Hydra heads
│   └── crop_classifier/           # NEW: Crop classifier weights
│       ├── siamese.pth
│       ├── classifier.pth
│       └── gallery.pt             # Siamese gallery embeddings
│
└── config/
    ├── crop_classifier.yaml       # NEW: Classifier config
    └── spectrogram.yaml           # EXISTING
```

---

## Module Responsibilities

### `crop_classifier/models/`

| File | Purpose |
|------|---------|
| `siamese.py` | Siamese network architecture for few-shot learning |
| `classifier.py` | Direct CNN classifier for 100+ labels |
| `losses.py` | Focal loss, contrastive loss implementations |

### `crop_classifier/training/`

| File | Purpose |
|------|---------|
| `trainer.py` | Training loops for Siamese and CNN |
| `augmentation.py` | Spectrogram-specific augmentations |
| `active_learning.py` | Hybrid uncertainty + diversity query |
| `pseudo_labeling.py` | Confident predictions → training data |

### `crop_classifier/inference/`

| File | Purpose |
|------|---------|
| `detector.py` | Two-stage detection pipeline |
| `blob_detector.py` | Wrapper for Otsu + region growing |
| `preprocessor.py` | Crop extraction, letterbox, normalize |
| `batch_processor.py` | Async batch collection for GPU |

### `crop_classifier/labeling/`

| File | Purpose |
|------|---------|
| `session.py` | Active learning session state |
| `storage.py` | Persist labels (SQLite or JSON) |

---

## Configuration

### `config/crop_classifier.yaml`

```yaml
# Crop preprocessing
# IMPORTANT: Run test_blob_recall.py to analyze signal shapes
# and determine appropriate target_size
preprocessing:
  target_size: [96, 32]  # Or [64, 64] if signals are square-ish
  padding_pct: 0.15
  normalization: per_crop

# Blob detection
# Tuned based on recall test results
blob_detection:
  sensitivity: high  # low/medium/high
  min_area: 30       # Lower if missing small signals
  max_area: 50000
  use_otsu: true
  otsu_multiplier: 0.8  # Lower = more sensitive

# Model
model:
  type: siamese  # or 'classifier'
  embedding_dim: 64
  dropout: 0.5

# Training
training:
  phase1:  # Siamese
    epochs: 50
    learning_rate: 0.001
    batch_size: 16
    margin: 1.0
  phase2:  # CNN
    epochs: 75
    learning_rate: 0.001
    batch_size: 8
    early_stop_patience: 15
    weight_decay: 0.0001

# Augmentation
augmentation:
  p_freq_shift: 0.5
  p_time_shift: 0.5
  p_power_scale: 0.5
  p_noise: 0.7
  p_freq_mask: 0.3
  p_time_mask: 0.3

# Active learning
active_learning:
  batch_size: 5
  diversity_beta: 3
  pseudo_label_threshold: 0.9
  pseudo_label_min_threshold: 0.7

# Labeling UI thresholds
# User only reviews crops between auto_reject and auto_accept
labeling:
  auto_accept_threshold: 0.8   # Above this = auto-accept
  auto_reject_threshold: 0.2   # Below this = auto-reject
  show_summary: true           # Show "Auto-accepted: X, Review: Y" bar

# Inference
inference:
  score_threshold: 0.5
  nms_threshold: 0.5
  max_detections: 100
```

---

## WebSocket API Extensions

### New Commands

Add to `server.py`:

```python
# Crop classifier commands
@ws_handler("crop_classifier")
async def handle_crop_classifier(websocket, command: str, data: dict):
    if command == "get_crops":
        # Get blob detection crops for labeling
        crops = await crop_service.get_crops_for_labeling(
            spectrogram=data['spectrogram'],
            max_crops=data.get('max_crops', 100),
        )
        return {"crops": crops}

    elif command == "submit_labels":
        # Record user labels
        await crop_service.record_labels(
            labels=data['labels'],  # {crop_id: bool}
            session_id=data['session_id'],
        )
        return {"success": True}

    elif command == "train_classifier":
        # Train/retrain classifier
        result = await crop_service.train(
            signal_name=data['signal_name'],
            config=data.get('config', {}),
        )
        return {"result": result}

    elif command == "get_predictions":
        # Get model predictions for crops
        predictions = await crop_service.predict(
            crop_ids=data['crop_ids'],
        )
        return {"predictions": predictions}

    elif command == "start_session":
        # Start active learning session
        session = await crop_service.start_session(
            signal_name=data['signal_name'],
            spectrogram=data['spectrogram'],
        )
        return {"session_id": session.id, "first_batch": session.first_batch}

    elif command == "next_batch":
        # Get next batch of uncertain samples
        batch = await crop_service.get_next_batch(
            session_id=data['session_id'],
        )
        return {"crops": batch}
```

---

## Migration from Hydra

Keep both systems during transition:

```python
# In unified_pipeline.py

class UnifiedPipeline:
    def __init__(self, use_crop_classifier: bool = False):
        self.use_crop_classifier = use_crop_classifier

        if use_crop_classifier:
            from crop_classifier.inference import TwoStageDetector
            self.detector = TwoStageDetector(...)
        else:
            from hydra import HydraDetector
            self.detector = HydraDetector(...)

    def detect(self, spectrogram):
        if self.use_crop_classifier:
            return self.detector.detect(spectrogram)
        else:
            return self.detector.detect(spectrogram)
```

Feature flag in config:

```yaml
# config/inference.yaml
detection:
  backend: crop_classifier  # or 'hydra'
```
