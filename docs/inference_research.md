# Inference Reasoning: Two-Stage RF Signal Detection

## Why We're Changing Our Approach

**TL;DR:** Our current Faster R-CNN learns WHERE signals appear, not WHAT signals look like. This document explains the research behind our fix.

---

## Part 1: The Position Bias Problem

### What's Happening

Our Faster R-CNN model memorizes signal positions instead of learning signal characteristics. When the same signal appears at a different frequency, the model fails to recognize it.

**Observed behavior:**
- Model achieves high accuracy on training data
- Model fails when signals appear at new positions
- Requires 50-100+ labeled samples per signal type
- Cannot generalize across frequency bands

### Why This Happens: Zero-Padding Encodes Position

Research from Islam et al. (ICLR 2020) revealed a surprising finding: **CNNs encode absolute position information through zero-padding at image boundaries.**

> "Position information is implicitly learned from the commonly used padding operation (zero-padding). Zero-padding is widely used for keeping the same dimensionality when applying convolution. However, its hidden effect in representation learning has been long omitted."
>
> — Islam et al., "How Much Position Information Do Convolutional Neural Networks Encode?", ICLR 2020

**The mechanism:**

1. Zero-padding creates a consistent "border signal" at image edges
2. As features propagate through layers, neurons can triangulate their absolute position relative to these borders
3. Deeper networks encode MORE position information, not less
4. The network exploits this "free" position information as a shortcut for classification

**Key experimental finding:**

When researchers removed padding from VGG networks, position encoding capability dropped by approximately half. When they added more padding to shallow networks, position encoding increased proportionally.

### Why This Matters for RF Spectrograms

In RF spectrograms:
- Y-axis = frequency (fixed mapping to pixel rows)
- X-axis = time

When a signal at 2.4 GHz always appears at pixel row 500 in training data, the CNN learns: **"classify pixels near row 500 as Signal_A"** rather than **"classify this spectral pattern as Signal_A"**.

This is a **shortcut** — it works perfectly on training data but fails catastrophically when:
- The same signal appears at a different frequency
- The SDR is retuned to a different center frequency
- The frequency axis is zoomed or rescaled

**Reference:**
- Islam, M.A., Jia, S., Bruce, N.D. (2020). "How Much Position Information Do Convolutional Neural Networks Encode?" ICLR 2020. https://openreview.net/forum?id=rJeB36NKvB
- Islam, M.A. et al. (2021). "Position, Padding and Predictions: A Deeper Look at Position Information in CNNs." arXiv:2101.12322

---

## Part 2: The Solution — Two-Stage Detection

### Core Insight

If the CNN never sees the signal's position in the original image, it cannot learn position-based shortcuts. It must learn the signal's actual visual characteristics.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: BLOB DETECTION                      │
│                    (Classical, No Learning)                     │
├─────────────────────────────────────────────────────────────────┤
│  Input: Full 1024×1024 spectrogram                              │
│                                                                 │
│  Process:                                                       │
│    1. Adaptive thresholding (local contrast detection)          │
│    2. Morphological operations (noise removal, gap filling)     │
│    3. Contour extraction with area/aspect filtering             │
│                                                                 │
│  Output: ~120 candidate bounding boxes (91.7% recall)           │
│                                                                 │
│  Position information: PRESERVED (we know where each blob is)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CROP EXTRACTION                              │
├─────────────────────────────────────────────────────────────────┤
│  For each candidate bounding box:                               │
│    1. Add 15% padding around bbox                               │
│    2. Extract crop from spectrogram                             │
│    3. Letterbox resize to 32×64 (preserving aspect ratio)       │
│    4. Normalize (mean=0, std=1)                                 │
│                                                                 │
│  Output: Normalized crop tensor, shape (1, 32, 64)              │
│                                                                 │
│  Position information: DESTROYED (all crops are centered)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: CROP CLASSIFIER                     │
│                    (Learned CNN)                                │
├─────────────────────────────────────────────────────────────────┤
│  Input: 32×64 normalized crop                                   │
│                                                                 │
│  Architecture: 3-layer CNN with Global Average Pooling          │
│    - Conv(1→32) → BN → ReLU → MaxPool                           │
│    - Conv(32→64) → BN → ReLU → MaxPool                          │
│    - Conv(64→128) → BN → ReLU → GlobalAvgPool                   │
│    - FC(128→num_classes)                                        │
│                                                                 │
│  Output: Class probabilities or embedding vector                │
│                                                                 │
│  What it learns: SIGNAL APPEARANCE (shape, texture, bandwidth)  │
│  What it cannot learn: SIGNAL POSITION (information destroyed)  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works

**The classifier sees ONLY the crop.** Every crop is:
- Centered in the same reference frame
- Normalized to the same size
- Stripped of any context about its original position

The classifier has no information about where the signal was in the original spectrogram. It MUST learn to recognize signals by their intrinsic visual properties:
- Bandwidth (how wide the signal is)
- Shape (flat-top, gaussian, notched, etc.)
- Texture (modulation patterns, noise characteristics)
- Temporal structure (continuous, pulsed, hopping)

**Same signal, different position → same crop appearance → same classification.**

---

## Part 3: Few-Shot Learning with Siamese Networks

### The Challenge

We have very few labeled samples per signal type (target: 25-100 labels total). Traditional CNN classifiers need thousands of samples per class to learn robust decision boundaries.

### Why Siamese Networks

Koch et al. (ICML 2015) introduced Siamese networks for one-shot image recognition. The key insight: **instead of learning to classify, learn to compare.**

> "We explore a method for learning siamese neural networks which employ a unique structure to naturally rank similarity between inputs. Once a network has been tuned, we can then capitalize on powerful discriminative features to generalize the predictive power of the network not just to new data, but to entirely new classes from unknown distributions."
>
> — Koch, Zemel, Salakhutdinov, "Siamese Neural Networks for One-shot Image Recognition", ICML 2015

**Traditional classifier:**
- Input: Image
- Output: Class probability distribution
- Requires: Many samples per class to learn decision boundaries

**Siamese network:**
- Input: Pair of images
- Output: Similarity score (same class or different class?)
- Requires: Enough pairs to learn similarity metric

### Architecture

```
        ┌───────────┐          ┌───────────┐
        │  Crop A   │          │  Crop B   │
        └─────┬─────┘          └─────┬─────┘
              │                      │
              ▼                      ▼
        ┌───────────┐          ┌───────────┐
        │  Encoder  │          │  Encoder  │  ← Shared weights
        │  (CNN)    │          │  (CNN)    │
        └─────┬─────┘          └─────┬─────┘
              │                      │
              ▼                      ▼
        ┌───────────┐          ┌───────────┐
        │ Embedding │          │ Embedding │
        │  (64-dim) │          │  (64-dim) │
        └─────┬─────┘          └─────┬─────┘
              │                      │
              └──────────┬───────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  Distance   │
                  │  (L2 norm)  │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ Similarity  │
                  │   Score     │
                  └─────────────┘
```

### Training with Contrastive Loss

Contrastive loss trains the network to:
- **Pull together** embeddings of same-class pairs
- **Push apart** embeddings of different-class pairs

```
L(A, B, y) = y × D(A,B)² + (1-y) × max(0, margin - D(A,B))²

Where:
  A, B = embedding vectors
  y = 1 if same class, 0 if different class
  D(A,B) = Euclidean distance
  margin = minimum distance for different classes (typically 1.0)
```

**With 25 labeled samples, we can generate:**
- ~300 same-class pairs (depends on class distribution)
- ~1000+ different-class pairs
- Total: ~1300 training pairs — enough to learn a similarity metric

### Inference

At inference time:
1. Extract crop from candidate blob
2. Compute embedding via encoder
3. Compare to gallery of known signal embeddings
4. Return class with highest similarity (nearest neighbor)

**Reference:**
- Koch, G., Zemel, R., Salakhutdinov, R. (2015). "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- Snell, J., Swersky, K., Zemel, R. (2017). "Prototypical Networks for Few-shot Learning." NeurIPS.

---

## Part 4: Scaling to Direct Classification

### When to Switch

Siamese networks excel with very few samples but plateau as data grows. Direct classifiers have higher asymptotic accuracy but need more data.

| Approach | Optimal Range | Accuracy |
|----------|---------------|----------|
| Siamese | 25-100 labels | 75-85% |
| Direct CNN | 100-500 labels | 85-95% |

**Transition trigger:** Siamese accuracy plateaus AND 100+ labels available.

### Why Not ResNet?

Our crops are small (32×64 pixels) with simple structure. Deep architectures like ResNet:
- Have too many parameters for small crops
- Overfit quickly on limited data
- Provide no benefit for simple textures

Our 3-layer CNN has ~100K parameters — appropriate for the task complexity.

### Focal Loss for Class Imbalance

Blob detection produces ~120 candidates per frame, but only ~1-5 are actual signals. This creates severe class imbalance (1:100 signal:noise ratio).

Lin et al. (ICCV 2017) introduced Focal Loss to address exactly this problem in object detection:

> "We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples."
>
> — Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

**Focal Loss formula:**

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

Where:
  p_t = predicted probability for true class
  α = class weight (0.25 for positive class)
  γ = focusing parameter (2.0)
```

**Effect:**
- Easy negatives (noise blobs correctly classified with high confidence) contribute nearly zero loss
- Hard examples (signals or confusing negatives) dominate the loss
- Network focuses learning on the difficult cases

**Reference:**
- Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). "Focal Loss for Dense Object Detection." ICCV 2017. https://arxiv.org/abs/1708.02002

---

## Part 5: Complete Pipeline

### Training Phase

```
1. LABEL COLLECTION (Active Learning)
   ├── Run blob detection on new spectrogram
   ├── Extract crops from candidate blobs
   ├── Classifier predicts confidence scores
   ├── User reviews ONLY uncertain crops (confidence 0.2-0.8)
   │   ├── Auto-accept: confidence > 0.8 → positive training sample
   │   ├── Auto-reject: confidence < 0.2 → negative training sample
   │   └── Review: 0.2 ≤ confidence ≤ 0.8 → user labels
   └── Add labeled crops to training set

2. MODEL TRAINING
   ├── Phase 1 (25-100 labels): Train Siamese with contrastive loss
   └── Phase 2 (100+ labels): Train direct CNN with focal loss

3. PSEUDO-LABELING (Data Augmentation)
   ├── High-confidence predictions (>0.9) become training samples
   └── Gradually lower threshold as model improves (0.9 → 0.7)
```

### Inference Phase

```
1. INPUT: 1024×1024 spectrogram

2. BLOB DETECTION (~5-10ms)
   ├── Adaptive threshold (block=51, C=-5)
   ├── Morphological open (3×3 ellipse)
   ├── Morphological close (7×1 rect)
   ├── Contour extraction
   └── Filter by area (50-5000) and aspect ratio (1.5-15.0)

3. CROP EXTRACTION (~1-2ms per crop)
   ├── Add 15% padding to bbox
   ├── Extract from spectrogram
   ├── Letterbox resize to 32×64
   └── Normalize (mean=0, std=1)

4. CLASSIFICATION (~10-20ms batch)
   ├── Batch all crops through CNN
   ├── Get class probabilities
   └── Apply confidence threshold

5. POST-PROCESSING (~1-2ms)
   ├── NMS to remove duplicates
   └── Return final detections with confidence

TOTAL: <50ms end-to-end
```

---

## Part 6: Why This Works Better

### Position Invariance Achieved

| Scenario | Old (Faster R-CNN) | New (Two-Stage) |
|----------|-------------------|-----------------|
| Same signal, same position | ✅ Works | ✅ Works |
| Same signal, different position | ❌ Fails | ✅ Works |
| New signal type, few samples | ❌ Needs 50-100 samples | ✅ Works with 5-10 samples |
| Retune SDR center frequency | ❌ Model breaks | ✅ Works unchanged |

### Sample Efficiency

The combination of:
1. **Position-invariant crops** — no position shortcuts to exploit
2. **Siamese learning** — compare, don't classify
3. **Active learning** — label only uncertain samples
4. **Pseudo-labeling** — leverage confident predictions

...enables effective learning from 25 labeled samples, vs 50-100+ with the old approach.

### Theoretical Grounding

This approach follows established principles from the ML literature:

1. **Removing spurious correlations** (position bias) — Islam et al. 2020
2. **Metric learning for few-shot problems** — Koch et al. 2015, Snell et al. 2017
3. **Focal loss for class imbalance** — Lin et al. 2017
4. **Active learning for efficient labeling** — Settles 2009

---

## Part 7: Known Limitations

### Stage 1 Ceiling

Blob detection achieves 91.7% recall. The 8.3% of missed signals are due to:
- Signal fragmentation (internal gaps cause multiple small blobs)
- Very weak signals (below adaptive threshold)

**These signals are invisible to the entire system.** No amount of classifier training can recover them.

**Mitigation:** If production recall needs to exceed 91.7%, revisit Stage 1 (lower thresholds, different proposal method).

### Novel Signal Types

The classifier learns a fixed set of signal classes. Truly novel modulation types will be classified as one of the known classes or flagged as "unknown" if entropy is high.

**Mitigation:** Monitor prediction entropy. High-entropy predictions may indicate novel signals requiring new labels.

### Compute Trade-off

120 crops × CNN forward pass is more compute than a single Faster R-CNN pass. However:
- Crops are small (32×64) — fast to process
- Batching is efficient on GPU
- Total latency still <50ms

---

## References

1. Islam, M.A., Jia, S., Bruce, N.D. (2020). "How Much Position Information Do Convolutional Neural Networks Encode?" ICLR 2020. https://openreview.net/forum?id=rJeB36NKvB

2. Islam, M.A. et al. (2021). "Position, Padding and Predictions: A Deeper Look at Position Information in CNNs." International Journal of Computer Vision. https://arxiv.org/abs/2101.12322

3. Koch, G., Zemel, R., Salakhutdinov, R. (2015). "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

4. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). "Focal Loss for Dense Object Detection." ICCV 2017. https://arxiv.org/abs/1708.02002

5. Snell, J., Swersky, K., Zemel, R. (2017). "Prototypical Networks for Few-shot Learning." NeurIPS 2017.

6. Khosla, P. et al. (2020). "Supervised Contrastive Learning." NeurIPS 2020.

7. Yang, Z., Wang, J., Zhu, Y. (2022). "Few-Shot Classification with Contrastive Learning." ECCV 2022.

---

## Summary

| Problem | Solution | Research Basis |
|---------|----------|----------------|
| CNN learns position, not signal | Crop-based classification destroys position info | Islam et al. 2020 |
| Few labeled samples | Siamese networks learn similarity, not classification | Koch et al. 2015 |
| Severe class imbalance | Focal loss down-weights easy negatives | Lin et al. 2017 |
| Labeling bottleneck | Active learning + pseudo-labeling | Settles 2009 |

**The two-stage approach forces the model to learn what signals look like, not where they appear.**
