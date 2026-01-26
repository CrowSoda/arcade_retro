# G20 Hydra Architecture - Implementation Plan

> **Status:** ✅ ALL PHASES COMPLETE  
> **Created:** 2026-01-26  
> **Updated:** 2026-01-26  
> **Completed:** Single session implementation

## ✅ Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Preparation | ✅ Complete | Directories created |
| Phase 1: Backbone Extraction | ✅ Complete | `backbone_extractor.py` implemented, extraction validated |
| Phase 2: Hydra Detector | ✅ Complete | `detector.py` implemented |
| Phase 3: Version Manager | ✅ Complete | `version_manager.py` with auto-promotion |
| Phase 4: Training Service | ✅ Complete | `service.py`, `dataset.py`, `splits.py`, `sample_manager.py` |
| Phase 5: WebSocket API | ✅ Complete | `/training` endpoint added to `server.py` |
| Phase 6: Flutter Frontend | ✅ Complete | Providers and widgets created |

### Extraction Results (2026-01-26)
```
Backbone: 116 tensors, 13,793,216 params, 55.2 MB
Head: 14 tensors, 14,499,865 params, 58.0 MB
Validation: PASSED
```

## Quick Navigation

| Document | Description |
|----------|-------------|
| [00_overview.md](00_overview.md) | Executive summary, goals, timeline |
| [01_current_codebase.md](01_current_codebase.md) | Analysis of existing code |
| [02_directory_structure.md](02_directory_structure.md) | New file layout & metadata schemas |
| [03_backend_phase1.md](03_backend_phase1.md) | Backbone extraction implementation |
| [04_backend_phase2.md](04_backend_phase2.md) | Hydra detector implementation |
| [05_backend_phase3.md](05_backend_phase3.md) | Version manager implementation |
| [06_training_and_api.md](06_training_and_api.md) | Training service & WebSocket API |
| [06a_training_dataset.md](06a_training_dataset.md) | **dataset.py & splits.py implementation** |
| [06b_training_data_flow.md](06b_training_data_flow.md) | **Flutter → Backend data flow** |
| [07_frontend.md](07_frontend.md) | Flutter providers & UI |
| [08_migration_and_testing.md](08_migration_and_testing.md) | Migration steps & testing |

---

## What is Hydra?

The Hydra architecture separates the FasterRCNN model into:

1. **Shared Backbone** (~55MB): ResNet18-FPN feature extractor, frozen after initial extraction
2. **Per-Signal Heads** (~10MB each): RPN + ROI detection heads, one per signal type

This enables:
- **12+ signals** running simultaneously within 200ms latency
- **~85% memory reduction** compared to separate full models
- **Incremental training** without full model retraining
- **Version control** with auto-promotion and rollback

---

## Key Numbers

| Metric | Current (Single Model) | After Hydra |
|--------|------------------------|-------------|
| Memory per signal | ~100MB | ~10MB |
| 6 signals loaded | 600MB | 115MB |
| 12 signals loaded | 1.2GB | 175MB |
| Inference (6 signals) | ~150ms | ~35ms |
| Inference (12 signals) | ~300ms | ~45ms |

---

## Implementation Order

```
Week 1:
├── Day 1-2: Backbone Extraction (Phase 1)
│   └── Extract backbone + head from creamy_chicken_fold3.pth
│   └── Validate outputs match original
│
├── Day 3-5: Hydra Detector (Phase 2)
│   └── Implement HydraDetector class
│   └── Integrate with unified_pipeline.py
│   └── Test: Flutter should work unchanged

Week 2:
├── Day 6-7: Version Manager (Phase 3)
│   └── Version creation, promotion, rollback
│
├── Day 8-10: Training Service (Phase 4)
│   └── Frozen-backbone training
│   └── Dataset and split management

Week 3:
├── Day 11-12: WebSocket API (Phase 5)
│   └── Training commands
│   └── Version commands
│
├── Day 13-15: Flutter Frontend (Phase 6)
│   └── New providers
│   └── Training screen redesign
│   └── Version history widget
```

---

## Critical Constraints

1. **Inference FFT is LOCKED**: 4096 FFT, 2048 hop, 80dB range (matches tensorcade training)
2. **Backbone is frozen**: Only heads train after initial extraction
3. **Auto-promote threshold**: 2% F1 improvement required (below 0.95), no regression above 0.95
4. **Version retention**: Keep last 5 versions per signal
5. **Minimum samples**: 5 labeled samples required for training

---

## Files to Create

### Backend (Python)
```
backend/
├── hydra/
│   ├── __init__.py
│   ├── config.py              # Constants
│   ├── backbone_extractor.py  # One-time migration tool
│   ├── detector.py            # HydraDetector class
│   └── version_manager.py     # Version control
│
└── training/
    ├── __init__.py
    ├── service.py             # TrainingService class
    ├── dataset.py             # SpectrogramDataset
    └── splits.py              # SplitManager
```

### Frontend (Dart)
```
lib/features/training/
├── providers/
│   ├── training_provider.dart
│   └── signal_versions_provider.dart
│
└── widgets/
    ├── version_history.dart
    └── training_progress_overlay.dart
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Backbone doesn't generalize | Test with 2-3 signal types before full rollout |
| Training slower than expected | Early stopping (patience=5), batch size tuning |
| Flutter integration breaks | Feature flag (`USE_HYDRA_DETECTOR`) for instant rollback |
| Version corruption | Atomic file writes, always keep previous version |

---

## Rollback Commands

```bash
# Quick disable (no data loss)
# Edit unified_pipeline.py: USE_HYDRA_DETECTOR = False

# Per-signal rollback
python -m backend.hydra.version_manager --signal creamy_chicken --rollback

# Full nuclear rollback
rm -rf models/backbone models/heads models/registry.json
cp models/legacy/creamy_chicken_fold3.pth models/
```

---

## Next Steps

1. **Review this plan** - Identify any missing details or concerns
2. **Approve/modify** - Make adjustments as needed
3. **Begin Phase 1** - Backbone extraction is the first concrete step
