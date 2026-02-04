# Blob Recall Validation Report

**Date:** 2026-02-01
**Dataset:** Creamy_Shrimp (48 labeled samples)
**Reviewer:** Bob (Senior Dev)

---

## Executive Summary

⚠️ **WARNING** - Improved blob detection achieves **87.5% IoU recall** with **62 blobs/frame**.

- Target recall: ≥90% ❌ (achieved 87.5%)
- Target blob count: 50-500 ✅ (achieved 62)
- 6 samples still miss (12.5%)

---

## Test Results Summary

| Method | Parameters | Recall | Blobs/Frame | Verdict |
|--------|------------|--------|-------------|---------|
| Otsu | mult=1.0 | 18.8% | 2,605 | ❌ FAIL |
| Otsu | mult=0.6 | 0.0% | 86 | ❌ FAIL |
| Adaptive (no filter) | block=51, C=5, IoU=0.3 | 70.8% | 38,526 | ❌ FAIL |
| **Adaptive + Morphology + Aspect Filter** | **min_ar=1.5, max_ar=10** | **87.5%** | **62** | **⚠️ WARNING** |

---

## IMPROVED RESULTS (Post-Processing)

With morphology + aspect ratio filtering:

```python
# Post-processing steps:
1. Adaptive threshold (block=51, C=-5)
2. Morphological opening (3x3 ellipse) - removes noise
3. Morphological closing (7x1 rect) - horizontal connect
4. Aspect ratio filter: 1.5 < AR < 10.0 (signals are WIDE)
5. Area filter: 100 < area < 5000
```

**Results:**
- **IoU Recall: 87.5%** (42/48 matched at IoU ≥ 0.3)
- **Blobs/frame: 62** (down from 38,526!)
- **Missed: 6 samples**

---

## Missed Samples Analysis

| Sample | Best IoU | GT Aspect Ratio | GT Area | Blobs Found |
|--------|----------|-----------------|---------|-------------|
| 20b03a39ec31f640 | 0.000 | 3.92 | 663 | 65 |
| 34ea96532f14c6fc | 0.000 | 2.73 | 615 | 79 |
| 41e91bc47c50a256 | 0.000 | 2.55 | 308 | 29 |
| 5d52cad53959e565 | 0.000 | 8.65 | 2499 | 63 |
| 6af44029b939ef34 | 0.283 | 1.86 | 902 | 90 |
| 9feccae95e6f8474 | 0.000 | 2.56 | 656 | 53 |

**Observation:** All 6 have blobs found but none align with GT boxes. Signal aspect ratios (1.86-8.65) are within filter range.

**Root cause:** These signals are likely weak (low contrast) or fragmented into multiple small blobs.

---

## Why Otsu Failed

**Root Cause:** RF spectrograms have varying noise floors across frequencies.

Sample `1b92032e916f9a11`:
```
Overall mean: 90.5
Otsu threshold: 90.0  ← Almost identical to mean!
Signal mean: 121.3 (only 15 units above background)
```

Otsu creates ONE GIANT blob covering half the image. Signal IoU with giant blob: 0.002

---

## Why IoU-Based Matching Failed

**Root Cause:** Adaptive thresholding finds signals INSIDE larger blobs.

Sample `17f38f7022990730`:
```
Ground truth: 37×12 px = 444 px²
Detected blob: 834×812 px = 677,208 px²
Signal IS INSIDE the blob
IoU = 444 / (677,208 - 444 + 444) = 0.001
```

The signal is contained but IoU is penalized by the blob being larger.

---

## Why Containment Works

**Metric:** Is the signal's CENTER PIXEL inside any detected blob?

```python
cx = (gt['x_min'] + gt['x_max']) // 2
cy = (gt['y_min'] + gt['y_max']) // 2
detected = binary[cy, cx] == 255
```

**Result: 47/48 = 97.9%**

For the crop classifier workflow, this is sufficient because:
1. Signal is detected (inside a blob)
2. Crop extracted from blob centroid will contain the signal
3. Classifier filters false positives

---

## Analysis of Single Missed Sample

Sample `35c28825f481eb5b`:
```
Ground truth: (490,613) to (533,624) = 43×11 pixels
Signal mean: 92.1
Local area mean: 83.3
Signal-to-background delta: 9 intensity units (extremely weak)

Center pixel (511,618) value: 90
Local threshold (C=5): ~88
Result: 90 < 88 + 5 = 93, so NOT detected

With C=1: DETECTED (threshold = 84)
```

**This is an edge case** - signal SNR is below expected production quality.

---

## Signal Shape Analysis

```
Total signals: 48
Average size: 45×15 pixels (W×H)
Width range: 28 - 147 pixels
Height range: 11 - 22 pixels
Aspect ratios: 1.55 to 8.65

RECOMMENDED CROP SIZE: 32×64 (H×W)
  - 80th percentile width: ~50px
  - 80th percentile height: ~17px
  - Add 30% padding
  - Round to multiples of 16
```

---

## Recommended Configuration

```yaml
blob_detection:
  method: adaptive_threshold
  block_size: 51
  C: 5
  min_area: 30

preprocessing:
  crop_size: [32, 64]  # H×W
  padding_pct: 0.15

# For extremely weak signals (C=1):
# blob_detection:
#   C: 1  # Higher false positive rate
```

---

## Evidence Files

| File | Purpose |
|------|---------|
| `scripts/test_blob_recall.py` | Test script |
| `recall_results.json` | Otsu medium |
| `recall_results_max.json` | Otsu max |
| `recall_results_adaptive.json` | Adaptive medium |
| `recall_results_high.json` | Adaptive high |
| `recall_results_iou02.json` | IoU=0.2 |

---

## Conclusion

**Blob detection status: ⚠️ WARNING (87.5% IoU recall)**

| Criteria | Threshold | Actual | Status |
|----------|-----------|--------|--------|
| IoU Recall | ≥90% | 87.5% | ⚠️ WARNING |
| Blob Count | 50-500 | 62 | ✅ PASS |
| Missed Samples | <10% | 12.5% | ⚠️ WARNING |

**Options:**
1. **Proceed with 87.5%** - Accept 12.5% of crops may be incorrectly located
2. **Lower aspect ratio filter** - Try min_ar=1.0 to catch lower AR signals
3. **Different approach** - Peak detection in frequency bins

**Recommendation:** Investigate the 6 missed samples further OR accept 87.5% and rely on classifier to handle some misaligned crops.

---

## Detailed Test Output

### Adaptive High Sensitivity (block=51, C=5)

```
Testing 48 sample(s) with sensitivity=high
IoU threshold: 0.3

  17f38f7022990730: recall=0.0% (0/1)
  20b03a39ec31f640: recall=0.0% (0/1)
  30c49ff6a4580a7e: recall=0.0% (0/1)
  34ea96532f14c6fc: recall=0.0% (0/1)
  35c28825f481eb5b: recall=0.0% (0/1)
  41e91bc47c50a256: recall=0.0% (0/1)
  5f427d2fee6c2d2b: recall=0.0% (0/1)
  75b469837696c0d5: recall=0.0% (0/1)
  8401c45a15fef61a: recall=0.0% (0/1)
  c05e1ef8bd2fbe70: recall=0.0% (0/1)
  cddcd453ff8fea6a: recall=0.0% (0/1)
  ce515e3b17c6bde5: recall=0.0% (0/1)
  da1979bf1a623a94: recall=0.0% (0/1)
  ef930dac9bc48cb8: recall=0.0% (0/1)

============================================================
SIGNAL SHAPE ANALYSIS
============================================================
  Total signals: 48
  Avg size: 45×15 pixels
  Width range: 28 - 147 pixels
  Height range: 11 - 22 pixels
  Aspect ratios: 1.55 to 8.65
  Suggested crop size: (32, 64) (H×W)

============================================================
RECALL TEST RESULTS
============================================================
  Samples tested: 48
  Ground truth signals: 48
  Detected blobs: 38526
  Matched: 34
  Missed: 14
  False positives: 38492

  RECALL: 70.8%
```

### Containment Test (Custom Metric)

```
Containment recall (center pixel): 47/48 = 97.9%
Missed: ['35c28825f481eb5b']
```

---

**END OF REPORT**
