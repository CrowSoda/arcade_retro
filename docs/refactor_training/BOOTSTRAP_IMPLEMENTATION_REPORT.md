# Bootstrap Implementation Report

## Summary

Implemented seed-based signal expansion system. User draws ~20 boxes as "seeds", system finds similar signals using template matching (NCC), user confirms via swipe UI, then trains Siamese model.

---

## What Was Implemented

### 1. Backend: SignalBootstrapper Class
**File:** `backend/crop_classifier/labeling/bootstrapper.py`

```python
class SignalBootstrapper:
    def find_similar(spectrogram, seed_boxes, top_k=50) -> BootstrapResult
    def confirm(confirmed_indices, rejected_indices) -> Stats
    def get_training_data() -> (crops, labels)
```

**Key function:** `score_candidates_by_seeds()` - Uses Normalized Cross-Correlation (NCC) to score similarity between detected blobs and user's seed templates.

### 2. Backend: WebSocket Commands
**File:** `backend/crop_classifier/api/handlers.py`

| Command | Input | Output |
|---------|-------|--------|
| `bootstrap` | spectrogram (base64), seed_boxes, top_k | candidates (ranked by similarity) |
| `confirm` | confirmed (indices), rejected (indices) | stats (ready_to_train flag) |

### 3. Flutter Provider Methods
**File:** `lib/features/training/providers/crop_classifier_provider.dart`

```dart
Future<BootstrapResult> bootstrap({
  required String spectrogramBase64,
  required List<Map<String, int>> seedBoxes,
  int topK = 50,
})

Future<BootstrapStats> confirmLabels({
  required List<int> confirmed,
  required List<int> rejected,
})
```

**Data classes added:**
- `BootstrapCandidate` - Single candidate with image, box, score
- `BootstrapStats` - Counts (positives, negatives, ready_to_train)
- `BootstrapResult` - Full result with conversion to CropReviewData

---

## Assumptions Made

### A1: Spectrogram Format
- Backend expects spectrogram as base64 PNG (grayscale uint8)
- Resolution matches what's displayed in the UI
- Coordinate system: origin at top-left, Y increases downward

### A2: Seed Box Format
```json
{
  "x_min": 100,  // pixels from left
  "y_min": 50,   // pixels from top
  "x_max": 200,
  "y_max": 80
}
```
- Coordinates are in spectrogram pixel space (not normalized 0-1)
- Boxes must be fully within spectrogram bounds

### A3: Template Matching Parameters
- **Target size:** 64×32 pixels (width×height) - all crops resized to this
- **Similarity method:** Normalized Cross-Correlation (NCC)
- **Score range:** -1.0 to 1.0 (0.3 default threshold for filtering)

### A4: Seeds Are Automatically Positive
When `confirm()` is called, ALL seeds are added as positives automatically. This means user's drawn boxes are always treated as ground truth.

### A5: Blob Detection Parameters (from blob_detector.py)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| min_area | 100 | Reject tiny noise blobs |
| max_area | 10000 | Reject huge regions |
| min_aspect_ratio | 1.5 | RF signals are wider than tall |
| max_aspect_ratio | 15.0 | Not too elongated |
| block_size | 51 | Adaptive threshold neighborhood |
| C | -15 | Threshold strictness |

### A6: IoU Overlap Filter
Candidates overlapping seeds (IoU > 0.3) are excluded to avoid duplicates.

---

## What's NOT Implemented (UI Wiring)

### Missing Piece: Get Spectrogram as PNG

The `TrainingSpectrogram` widget renders the spectrogram, but there's no method to capture it as base64 PNG. Options:

**Option A: Screenshot the widget**
```dart
// Add RepaintBoundary key to TrainingSpectrogram
final boundary = _spectrogramKey.currentContext!.findRenderObject() as RenderRepaintBoundary;
final image = await boundary.toImage(pixelRatio: 1.0);
final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
final base64 = base64Encode(byteData!.buffer.asUint8List());
```

**Option B: Backend generates from RFCAP**
Modify `bootstrap` to accept RFCAP path + time range instead of spectrogram image. Backend reads the same window that Flutter displays.

### Missing Piece: Convert LabelBox to Seed Box

Current `LabelBox` uses normalized coordinates (0-1). Need to convert to pixel coordinates:

```dart
List<Map<String, int>> _labelBoxesToSeedBoxes() {
  if (_loadedHeader == null) return [];

  // Get spectrogram dimensions from widget
  final specWidth = spectrogramWidth;  // Need access to this
  final specHeight = spectrogramHeight;

  return _labelBoxes.map((box) => {
    return {
      'x_min': (box.x1 * specWidth).round(),
      'y_min': (box.y1 * specHeight).round(),
      'x_max': (box.x2 * specWidth).round(),
      'y_max': (box.y2 * specHeight).round(),
    };
  }).toList();
}
```

### Modified _startTraining Flow

```dart
void _startTraining() async {
  // 1. Get spectrogram as PNG (needs implementation)
  final spectrogramPng = await _captureSpectrogramAsPng();

  // 2. Convert user's boxes to seed format
  final seedBoxes = _labelBoxesToSeedBoxes();

  if (seedBoxes.length < 5) {
    // Show error: need at least 5 seed boxes
    return;
  }

  // 3. Call bootstrap
  final cropNotifier = ref.read(cropClassifierProvider.notifier);
  final bootstrapResult = await cropNotifier.bootstrap(
    spectrogramBase64: spectrogramPng,
    seedBoxes: seedBoxes,
    topK: 50,
  );

  // 4. Show swipe UI with candidates (already sorted by similarity!)
  final reviewResult = await showCropReviewDialog(
    context: context,
    crops: bootstrapResult.toCropReviewDataList(),
  );

  // 5. Record confirmations
  final stats = await cropNotifier.confirmLabels(
    confirmed: reviewResult.acceptedIndices,
    rejected: reviewResult.rejectedIndices,
  );

  // 6. Train if ready
  if (stats.readyToTrain) {
    // Trigger training...
  }
}
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `backend/crop_classifier/labeling/bootstrapper.py` | **NEW** - SignalBootstrapper class |
| `backend/crop_classifier/api/handlers.py` | Added `bootstrap`, `confirm` commands |
| `lib/.../crop_classifier_provider.dart` | Added `bootstrap()`, `confirmLabels()`, data classes |
| `lib/.../training_screen.dart` | **NOT CHANGED** - UI wiring needed |

---

## Testing the Backend

```bash
# Start backend
cd g20_demo/backend && python server.py

# Test bootstrap command (from Python)
import asyncio
import websockets
import json
import base64
from PIL import Image

async def test_bootstrap():
    uri = "ws://localhost:8765/crop"
    async with websockets.connect(uri) as ws:
        # Create test spectrogram
        img = Image.open("test_spectrogram.png")
        # ... encode to base64 ...

        await ws.send(json.dumps({
            "command": "bootstrap",
            "spectrogram": base64_spec,
            "seed_boxes": [
                {"x_min": 100, "y_min": 50, "x_max": 200, "y_max": 80},
                # ... more seeds
            ],
            "top_k": 50,
        }))

        response = await ws.recv()
        print(json.loads(response))
```

---

## Recommendations

1. **Easiest path:** Modify backend to accept RFCAP path + time window instead of spectrogram PNG. This avoids the screenshot complexity.

2. **For immediate testing:** Use the existing `crop_detect_file` with `reference_boxes` parameter. It already supports template-based filtering.

3. **Long term:** Add `_captureSpectrogramAsPng()` method to TrainingSpectrogram widget using RepaintBoundary.

---

## Flow Diagram

```
User draws 20 boxes
        ↓
_labelBoxes populated
        ↓
Click "Train Model"
        ↓
Capture spectrogram PNG  ←── NEEDS IMPLEMENTATION
        ↓
Convert boxes to pixels  ←── NEEDS IMPLEMENTATION
        ↓
Call bootstrap(png, boxes)
        ↓
Backend: find_similar()
  - Blob detection (120 candidates)
  - NCC scoring against seeds
  - Rank and return top 50
        ↓
Show swipe UI (pre-sorted!)
        ↓
Call confirm(indices)
        ↓
Backend: seeds + confirmed = positives
         rejected = negatives
        ↓
stats.ready_to_train?
        ↓
Call crop_train
        ↓
Siamese model trained!
```
