# New Directory Structure

## Complete File Layout

```
g20_demo/
├── models/
│   ├── backbone/
│   │   ├── v1.pth                      # Extracted ResNet18-FPN (~55MB)
│   │   ├── active.pth                  # Symlink → v1.pth
│   │   └── metadata.json               # Backbone version info
│   │
│   ├── heads/
│   │   ├── creamy_chicken/
│   │   │   ├── v1.pth                  # Head weights (~10MB)
│   │   │   ├── v2.pth                  # Newer version
│   │   │   ├── active.pth              # Symlink → best version
│   │   │   └── metadata.json           # Version history
│   │   │
│   │   ├── lte_uplink/
│   │   │   ├── v1.pth
│   │   │   ├── active.pth
│   │   │   └── metadata.json
│   │   │
│   │   └── {signal_name}/
│   │       └── ...
│   │
│   ├── registry.json                   # Index of all signals
│   │
│   └── legacy/                         # Old full models (migration)
│       └── creamy_chicken_fold3.pth    # Original model backup
│
├── training_data/
│   ├── signals/
│   │   ├── creamy_chicken/
│   │   │   ├── samples/
│   │   │   │   ├── 0001.npz            # Compressed spectrogram
│   │   │   │   ├── 0001.json           # Bounding boxes + metadata
│   │   │   │   ├── 0002.npz
│   │   │   │   ├── 0002.json
│   │   │   │   └── ...
│   │   │   │
│   │   │   ├── splits/
│   │   │   │   ├── v1.json             # Train/val split version 1
│   │   │   │   ├── v2.json             # Extended split
│   │   │   │   └── active.json         # Symlink → current split
│   │   │   │
│   │   │   └── manifest.json           # Sample count, dates
│   │   │
│   │   └── {signal_name}/
│   │       └── ...
│   │
│   └── index.json                      # All signals, sample counts
│
├── backend/
│   ├── hydra/                          # NEW: Hydra-specific code
│   │   ├── __init__.py
│   │   ├── detector.py                 # HydraDetector class
│   │   ├── trainer.py                  # Head training logic
│   │   ├── backbone_extractor.py       # Extract from full model
│   │   ├── version_manager.py          # Version control
│   │   └── config.py                   # Constants
│   │
│   ├── training/                       # NEW: Training service
│   │   ├── __init__.py
│   │   ├── service.py                  # TrainingService class
│   │   ├── dataset.py                  # SpectrogramDataset
│   │   └── splits.py                   # Train/val split manager
│   │
│   └── [existing files unchanged]
│
└── lib/features/training/
    ├── training_screen.dart            # REPLACE: New UI
    ├── providers/
    │   ├── training_provider.dart      # NEW
    │   ├── signal_versions_provider.dart # NEW
    │   └── training_data_provider.dart # NEW
    └── widgets/
        ├── version_history.dart        # NEW
        ├── signal_selector.dart        # NEW
        └── training_progress.dart      # NEW (modify existing)
```

---

## Metadata File Schemas

### `models/backbone/metadata.json`

```json
{
    "version": 1,
    "created_at": "2026-01-25T10:00:00Z",
    "source": "extracted_from_creamy_chicken_fold3.pth",
    "architecture": "resnet18_fpn",
    "input_size": [3, 1024, 1024],
    "feature_channels": 256,
    "extraction_script": "backend/hydra/backbone_extractor.py",
    "original_model_hash": "abc123def456...",
    "notes": "Initial backbone - replace with generalized version when multi-signal data available"
}
```

### `models/heads/{signal_name}/metadata.json`

```json
{
    "signal_name": "creamy_chicken",
    "backbone_version": 1,
    "active_version": 2,
    "version_retention": 5,
    "versions": [
        {
            "version": 1,
            "created_at": "2026-01-20T10:00:00Z",
            "sample_count": 127,
            "split_version": 1,
            "epochs_trained": 23,
            "early_stopped": true,
            "best_epoch": 18,
            "metrics": {
                "f1_score": 0.91,
                "precision": 0.89,
                "recall": 0.93,
                "val_loss": 0.0234,
                "train_loss": 0.0189
            },
            "training_time_sec": 180,
            "learning_rate": 0.001,
            "batch_size": 4,
            "promoted_at": "2026-01-20T10:05:00Z",
            "is_active": false,
            "notes": "Initial training with field samples"
        },
        {
            "version": 2,
            "created_at": "2026-01-23T14:00:00Z",
            "sample_count": 200,
            "split_version": 2,
            "epochs_trained": 15,
            "early_stopped": true,
            "best_epoch": 12,
            "metrics": {
                "f1_score": 0.93,
                "precision": 0.92,
                "recall": 0.94,
                "val_loss": 0.0198,
                "train_loss": 0.0156
            },
            "training_time_sec": 95,
            "learning_rate": 0.001,
            "batch_size": 4,
            "parent_version": 1,
            "auto_promoted": true,
            "promotion_reason": "F1 improved 2.2% (> 2% threshold)",
            "is_active": true,
            "notes": "Added 73 samples including low SNR edge cases from urban environment"
        }
    ]
}
```

### `models/registry.json`

```json
{
    "backbone_version": 1,
    "backbone_path": "backbone/active.pth",
    "last_updated": "2026-01-25T10:00:00Z",
    "signals": {
        "creamy_chicken": {
            "active_head_version": 2,
            "head_path": "heads/creamy_chicken/active.pth",
            "sample_count": 200,
            "f1_score": 0.93,
            "last_trained": "2026-01-23T14:00:00Z",
            "is_loaded": false
        },
        "lte_uplink": {
            "active_head_version": 1,
            "head_path": "heads/lte_uplink/active.pth",
            "sample_count": 89,
            "f1_score": 0.87,
            "last_trained": "2026-01-21T09:00:00Z",
            "is_loaded": false
        },
        "wifi_24": {
            "active_head_version": 2,
            "head_path": "heads/wifi_24/active.pth",
            "sample_count": 156,
            "f1_score": 0.82,
            "last_trained": "2026-01-24T16:00:00Z",
            "is_loaded": true
        }
    }
}
```

### `training_data/signals/{signal_name}/manifest.json`

```json
{
    "signal_name": "creamy_chicken",
    "total_samples": 215,
    "labeled_samples": 200,
    "pending_samples": 15,
    "created_at": "2026-01-20T09:00:00Z",
    "last_updated": "2026-01-25T10:15:00Z",
    "sources": [
        {
            "capture_file": "MAN_024307ZJAN26_2430.rfcap",
            "sample_count": 127,
            "added_at": "2026-01-20T09:00:00Z"
        },
        {
            "capture_file": "MAN_143020ZJAN23_2428.rfcap",
            "sample_count": 73,
            "added_at": "2026-01-23T14:00:00Z"
        },
        {
            "capture_file": "MAN_101500ZJAN25_2431.rfcap",
            "sample_count": 15,
            "added_at": "2026-01-25T10:15:00Z",
            "is_pending": true
        }
    ]
}
```

### `training_data/signals/{signal_name}/splits/v1.json`

```json
{
    "version": 1,
    "created_at": "2026-01-20T10:00:00Z",
    "total_samples": 127,
    "train_samples": ["0001", "0002", "0003", "0005", "0006", "0008", "..."],
    "val_samples": ["0004", "0007", "0012", "0019", "..."],
    "train_count": 102,
    "val_count": 25,
    "val_ratio": 0.197,
    "random_seed": 42,
    "stratified": false,
    "notes": "Initial 80/20 split"
}
```

### `training_data/signals/{signal_name}/samples/0001.json`

```json
{
    "sample_id": "0001",
    "source_capture": "MAN_024307ZJAN26_2430.rfcap",
    "time_offset_sec": 0.05,
    "duration_sec": 0.2,
    "center_freq_mhz": 2430.0,
    "bandwidth_mhz": 5.0,
    "sample_rate_mhz": 20.0,
    "created_at": "2026-01-20T09:30:00Z",
    "created_by": "user_label",
    "spectrogram_params": {
        "fft_size": 4096,
        "hop_length": 2048,
        "dynamic_range_db": 80.0,
        "output_size": [1024, 1024]
    },
    "boxes": [
        {
            "x_min": 412,
            "y_min": 156,
            "x_max": 623,
            "y_max": 489,
            "label": "signal",
            "confidence": null,
            "area_pixels": 70269
        }
    ],
    "augmented_from": null,
    "quality_score": null
}
```

### `training_data/signals/{signal_name}/samples/0001.npz`

Compressed numpy archive containing:
```python
{
    "spectrogram": np.uint8,  # Shape: (1024, 1024), normalized 0-255
    "metadata": {
        "fft_size": 4096,
        "hop_length": 2048,
        "dynamic_range_db": 80.0,
        "vmin_db": -80.0,
        "vmax_db": 0.0,
    }
}
```

---

## Directory Creation Script

```python
# backend/hydra/setup_directories.py

import os
from pathlib import Path

def create_hydra_directories(base_dir: str = "."):
    """Create the Hydra directory structure."""
    
    dirs = [
        "models/backbone",
        "models/heads",
        "models/legacy",
        "training_data/signals",
    ]
    
    for d in dirs:
        path = Path(base_dir) / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")
    
    # Create empty registry.json
    registry_path = Path(base_dir) / "models" / "registry.json"
    if not registry_path.exists():
        registry_path.write_text('{"backbone_version": null, "signals": {}}')
        print(f"Created: {registry_path}")
    
    # Create empty training index
    index_path = Path(base_dir) / "training_data" / "index.json"
    if not index_path.exists():
        index_path.write_text('{"signals": {}}')
        print(f"Created: {index_path}")

if __name__ == "__main__":
    create_hydra_directories()
```
