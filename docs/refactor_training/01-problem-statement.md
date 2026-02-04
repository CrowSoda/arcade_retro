# Problem Statement: Position Bias in Faster R-CNN

## The Core Issue

**Current Faster R-CNN learns signal position, not signal characteristics.**

When your model learns "signals at frequency X typically appear at pixel row Y," it fails to generalize to signals at different frequencies or in different captures.

---

## Why CNNs Encode Position

### Research Background

The ICLR 2020 paper **"How Much Position Information Do Convolutional Neural Networks Encode?"** demonstrates:

1. **Zero-padding is the source:** Standard CNNs encode absolute position through zero-padding operations at image borders
2. **Higher layers accumulate more position info:** Position encoding propagates through the network
3. **Object detectors inherit this bias:** Both region proposals and bounding box regression in Faster R-CNN are affected

### How It Manifests in Spectrograms

```
Spectrogram (1024×1024)
┌─────────────────────────┐
│  Zero-padding border    │ ← CNN learns "distance from edge"
│  ┌─────────────────┐    │
│  │                 │    │
│  │   Signal at     │    │
│  │   row 200       │    │ ← Model memorizes Y=200 ≈ "this frequency"
│  │                 │    │
│  └─────────────────┘    │
│                         │
└─────────────────────────┘
```

When the same signal appears at a different frequency (different Y position), the model struggles because it learned **position-dependent features**, not signal characteristics.

---

## Evidence in Current System

### Symptom 1: Detection accuracy varies by frequency

- Signals trained at one frequency detect well there
- Same signal type at different frequency: poor recall
- Model has learned "signal X lives at pixel Y"

### Symptom 2: Training requires many samples

- Current system needs 50-100+ samples for convergence
- Much of this is learning position invariance by brute force
- Inefficient use of precious labeled data

### Symptom 3: Anchor mismatch compounds the problem

- Dynamic anchors help with aspect ratio
- But anchors still operate in absolute coordinate space
- RPN proposals inherit position bias from backbone features

---

## Why Faster R-CNN Specifically Fails

```
                    Position Bias Flow

Input Image (1024×1024)
        │
        ▼
┌───────────────────┐
│   Conv Layer 1    │ ← Zero-pad: learn distance from border
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│   Conv Layer 2    │ ← Position info accumulates
└───────┬───────────┘
        │
        ▼ (repeat for ResNet-18: 18 layers)
        │
┌───────────────────┐
│   FPN Features    │ ← Position thoroughly encoded
└───────┬───────────┘
        │
    ┌───┴───┐
    │       │
    ▼       ▼
┌───────┐ ┌───────────┐
│  RPN  │ │ ROI Heads │
│ (loc) │ │ (class)   │ ← Both inherit position bias
└───────┘ └───────────┘
```

Every component in Faster R-CNN processes features that contain absolute position information. There's no way to "opt out" of this bias within the architecture.

---

## The Label Budget Constraint

**User can only label ~25 signals manually.**

With 80+ signals per second in captures, manual labeling is the bottleneck. We need an approach that:

1. Works with minimal labels (25-50)
2. Doesn't waste labels teaching position invariance
3. Maximizes information from each human decision

Current Faster R-CNN fails all three requirements.

---

## Solution Requirements

Any replacement architecture must:

| Requirement | Why |
|-------------|-----|
| **No position in features** | Classifier never sees where crop came from |
| **Works with 25 labels** | Hard constraint from user workflow |
| **Fast inference** | <50ms for real-time operation |
| **Classical proposals** | Non-learned proposal generation avoids position bias |
| **Scalable training** | Can improve as more labels become available |

The two-stage approach (blob detection → crop classification) meets all requirements.
