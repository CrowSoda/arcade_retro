# Two-Stage RF Signal Detection Refactor Plan

**Last Updated:** 2026-02-01
**Status:** Planning
**Goal:** Replace position-biased Faster R-CNN with position-invariant crop classification

---

## Document Index

| # | File | Description |
|---|------|-------------|
| **00a** | [**prework-blob-validation.md**](00a-prework-blob-validation.md) | **⚠️ DO FIRST: Validate blob detection recall** |
| 01 | [problem-statement.md](01-problem-statement.md) | Why Faster R-CNN fails for spectrogram detection |
| 02 | [architecture-overview.md](02-architecture-overview.md) | Two-stage solution architecture |
| 03 | [stage1-blob-detection.md](03-stage1-blob-detection.md) | Classical blob detection (existing) |
| 04 | [stage2-crop-classifier.md](04-stage2-crop-classifier.md) | Learned crop classification models |
| 05 | [crop-preprocessing.md](05-crop-preprocessing.md) | Crop extraction and normalization |
| 06 | [training-siamese.md](06-training-siamese.md) | Phase 1: Siamese network (25-100 labels) |
| 07 | [training-cnn.md](07-training-cnn.md) | Phase 2: Direct CNN (100+ labels) |
| 08 | [data-augmentation.md](08-data-augmentation.md) | Spectrogram-specific augmentations |
| 09 | [active-learning.md](09-active-learning.md) | Query strategies and pseudo-labeling |
| 10 | [flutter-labeling-ui.md](10-flutter-labeling-ui.md) | Swipe and grid labeling interfaces |
| 11 | [inference-pipeline.md](11-inference-pipeline.md) | End-to-end detection flow |
| 12 | [code-structure.md](12-code-structure.md) | File organization and modules |
| 13 | [accuracy-expectations.md](13-accuracy-expectations.md) | Performance targets and failure modes |
| 14 | [implementation-roadmap.md](14-implementation-roadmap.md) | Week-by-week development plan |
| 15 | [migration-guide.md](15-migration-guide.md) | Transitioning from current Hydra system |

---

## Quick Summary

**The Problem:** Current Faster R-CNN learns signal position, not signal characteristics. CNNs encode absolute position through zero-padding borders.

**The Solution:** Two-stage approach:
1. **Stage 1:** Classical blob detection (Otsu + region growing) - no position bias
2. **Stage 2:** Crop classifier (CNN) - sees only cropped signal, never position

**Target:** 25 manual label decisions → 200+ training samples via confirmation UI

---

## ⚠️ Pre-Work Required

**Before starting Week 1, complete the blob detection recall test:**

```bash
python scripts/test_blob_recall.py --spectrogram ... --labels ... --sensitivity medium
```

If recall < 90%, tune blob detector before proceeding. See [00a-prework-blob-validation.md](00a-prework-blob-validation.md).

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Classifier architecture | 3-layer CNN, not ResNet | Small crops, few samples |
| Loss function | Focal Loss | Handles class imbalance |
| Initial approach | Siamese network | Works with <50 labels |
| Scale approach | Direct classifier | When labels reach 100+ |
| Crop size | **Configurable** (analyze signal shapes) | Minimize padding waste |
| Labeling flow | **Uncertain-only review** | User only sees 0.2-0.8 confidence |
| Augmentation | Power scaling, noise, freq shift | Domain-appropriate |

---

## Current System vs. Proposed

| Aspect | Current (Hydra) | Proposed (Crop Classifier) |
|--------|-----------------|---------------------------|
| Architecture | Faster R-CNN + heads | Blob detection + CNN |
| Position handling | Learned (biased) | Removed (invariant) |
| Min labels | ~50-100 for convergence | 25 for useful model |
| Inference time | ~35ms (6 heads) | ~30ms (blob + batch CNN) |
| Memory | ~115MB (6 heads) | ~50MB (single CNN) |
| Training | End-to-end detection | Two-stage: proposal + classify |
