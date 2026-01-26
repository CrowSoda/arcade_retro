# G20 Hydra Architecture - Implementation Plan Overview

## Executive Summary

This plan details implementing a **shared-backbone multi-head detection system** for G20. The architecture enables parallel detection of multiple signal types with drastically reduced memory footprint and inference time.

## Current System State

### What Exists Today

| Component | Status | File Location |
|-----------|--------|---------------|
| FasterRCNN Model | ✅ Single model | `models/creamy_chicken_fold3.pth` |
| Inference Engine | ✅ TRT/ONNX/PyTorch fallback | `backend/inference.py` |
| Unified Pipeline | ✅ Single-model detection | `backend/unified_pipeline.py` |
| WebSocket Server | ✅ Row-strip streaming | `backend/server.py` |
| GPU FFT | ✅ Batched spectrogram | `backend/gpu_fft.py` |
| Signal Database | ✅ JSON persistence | `lib/core/database/signal_database.dart` |
| Training Screen | ⚠️ Simulated training | `lib/features/training/training_screen.dart` |
| Training Spectrogram | ✅ Label box drawing | `lib/features/training/widgets/training_spectrogram.dart` |
| Mission Config | ✅ YAML-based | `lib/features/config/providers/mission_provider.dart` |

### Key Technical Facts

1. **Model Architecture**: ResNet18-FPN backbone + Faster R-CNN detection heads
2. **Inference FFT**: 4096 FFT, 2048 hop, 80dB range (FIXED - matches training)
3. **Waterfall FFT**: 8K-64K configurable, 60dB range (display only)
4. **Spectrogram Size**: 1024×1024 grayscale expanded to 3-channel
5. **Detection Classes**: 2 (background + creamy_chicken)
6. **Backend Communication**: WebSocket on dynamic port, gRPC on 50051

## Target Goals

| Metric | Current | After Hydra |
|--------|---------|-------------|
| Memory per signal | ~100MB | ~10MB |
| 6 signals loaded | 600MB | 105MB |
| 12 signals loaded | 1.2GB | 165MB |
| Inference (6 signals) | ~150ms | ~35ms |
| Inference (12 signals) | ~300ms | ~45ms |

## Implementation Phases

### Phase 1: Backbone Extraction (Backend)
- Extract backbone weights from existing model
- Create head-only weight files
- Validate output parity

### Phase 2: Hydra Detector (Backend)
- Implement shared-backbone inference
- Load/unload heads dynamically
- Parallel head execution

### Phase 3: Version Manager (Backend)
- Version control for heads
- Auto-promotion logic
- Rollback capability

### Phase 4: Training Service (Backend)
- Frozen-backbone training
- Dataset management
- Split management

### Phase 5: Flutter Providers (Frontend)
- Training state management
- Version history provider
- Training data provider

### Phase 6: Training UI (Frontend)
- Redesigned training screen
- Version history display
- Training progress overlay

## File Organization

All plan documents are in `docs/hydra_plan/`:

- `00_overview.md` - This file
- `01_current_codebase.md` - Detailed analysis of existing code
- `02_directory_structure.md` - New file layout with metadata schemas
- `03_backend_phase1.md` - Backbone extraction implementation
- `04_backend_phase2.md` - Hydra detector implementation
- `05_backend_phase3.md` - Version manager implementation
- `06_backend_phase4.md` - Training service implementation
- `07_frontend_phase1.md` - Flutter providers
- `08_frontend_phase2.md` - Training screen UI redesign
- `09_websocket_api.md` - WebSocket API additions
- `10_migration.md` - Step-by-step migration guide
- `11_testing.md` - Test requirements and benchmarks

## Estimated Timeline

| Phase | Duration | Dependency |
|-------|----------|------------|
| Phase 1: Backbone Extraction | 2 days | None |
| Phase 2: Hydra Detector | 3 days | Phase 1 |
| Phase 3: Version Manager | 2 days | Phase 2 |
| Phase 4: Training Service | 3 days | Phase 2 |
| Phase 5: Flutter Providers | 2 days | Phase 4 |
| Phase 6: Training UI | 3 days | Phase 5 |
| **Total** | **~15 days** | |

## Critical Constraints

1. **Inference FFT is LOCKED**: Must match tensorcade training exactly (4096 FFT, 80dB)
2. **Backbone is frozen**: Only heads train after initial extraction
3. **Version retention**: Keep last 5 versions minimum
4. **Auto-promote threshold**: 2% F1 improvement required (or no regression above 0.95)
5. **Backward compatibility**: Existing creamy_chicken detection must work unchanged during migration
