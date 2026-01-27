# G20 Signal Detection Platform - Technical Investigation

Comprehensive fact-finding for embedding-based multi-signal detection design.

---

## SECTION 1: CURRENT MODEL & INFERENCE ARCHITECTURE

### 1.1 Model Loading (backend/inference.py)

**Load priority:** TensorRT → ONNX → PyTorch

```python
# From inference.py - InferenceEngine._load_pytorch()
backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
model = torchvision.models.detection.FasterRCNN(backbone, num_classes=self.num_classes)
state = torch.load(self.model_path, map_location=self.device, weights_only=False)
model.load_state_dict(state)
model.to(self.device)
model.eval()

if self.precision == "fp16" and self.device.type == "cuda":
    model.half()
```

### 1.2 Model Output Format

```python
# Output from Faster R-CNN after inference
results = [{
    "boxes": np.array([[x1, y1, x2, y2], ...]),  # Pixel coordinates (0-1024)
    "scores": np.array([0.95, 0.87, ...]),       # Confidence scores
    "labels": np.array([1, 1, ...])              # Class IDs
}]
```

### 1.3 Inference Function (unified_pipeline.py)

```python
def process_chunk(self, iq_data: np.ndarray, pts: float, score_threshold: float = 0.5):
    """
    Input: IQ samples (np.complex64)
    Output: Dict with 'detections', 'pts', 'inference_ms', 'stream_idx'
    """
    spec = self.compute_spectrogram(iq_data)  # 4096 FFT, 80dB, 1024x1024
    with torch.inference_mode():
        outputs = self.model(spec.half())

    # Parse to Detection objects
    detections = []
    for i in range(len(boxes)):
        det = Detection(
            x1=float(box[0]) / 1024,  # Normalized 0-1
            y1=1.0 - (float(box[3]) / 1024),  # Flip Y axis
            x2=float(box[2]) / 1024,
            y2=1.0 - (float(box[1]) / 1024),
            confidence=float(scores[i]),
            class_id=int(labels[i]),
            class_name=self.class_names[int(labels[i])],
        )
        detections.append(det)
```

### 1.4 Model File Structure

```
g20_demo/models/
└── creamy_chicken_fold3.pth    # PyTorch state_dict

# Content of .pth file:
type(checkpoint) = <class 'collections.OrderedDict'>
# Keys: backbone.body.*, backbone.fpn.*, rpn.*, roi_heads.*
```

### 1.5 Spectrogram Preprocessing (CRITICAL - MUST MATCH TRAINING)

```python
# From unified_pipeline.py - TripleBufferedPipeline.compute_spectrogram()
# INFERENCE PARAMS - MUST MATCH TENSORCADE EXACTLY!
inference_fft_size = 4096
inference_noverlap = 2048  # 50% overlap
inference_hop_length = 2048
inference_dynamic_range = 80.0  # CRITICAL!

Zxx = torch.stft(chunk, n_fft=4096, hop_length=2048, window=hann_window, return_complex=True)
Zxx = torch.fft.fftshift(Zxx, dim=0)
sxx_db = 10 * torch.log10(power + 1e-12)

# Normalize to 0-1
vmax = sxx_db.max()
vmin = vmax - 80.0  # 80dB dynamic range
sxx_norm = ((sxx_db - vmin) / (vmax - vmin)).clamp_(0, 1)

# Resize to 1024x1024, expand to 3 channels
resized = F.interpolate(sxx_norm, size=(1024, 1024), mode='bilinear')
return resized.expand(-1, 3, -1, -1)  # BLACK AND WHITE expanded to RGB
```

---

## SECTION 2: TRAINING PIPELINE

### 2.1 Training Screen (lib/features/training/training_screen.dart)

**Current state:** Training is **simulated** in Flutter - no actual PyTorch training happens in the app yet.

```dart
void _simulateTraining() async {
  for (int i = 0; i <= 100 && _isTraining; i++) {
    await Future.delayed(const Duration(milliseconds: 100));
    setState(() => _trainingProgress = i / 100);
  }
  // Simulate realistic F1 scores based on number of labels
  _saveTrainingResults();
}
```

### 2.2 Training Data Format (LabelBox)

```dart
// From training_spectrogram.dart
class LabelBox {
  int id;
  String className;
  double? timeStartSec;
  double? timeEndSec;
  double? freqStartMHz;
  double? freqEndMHz;
  Rect? pixelRect;
  bool isSelected;
}
```

### 2.3 TENSORCADE Training (ARCADE/src/tensorcade/workers.py)

**This is the REAL training code from the beta software:**

```python
class TrainWorker(QObject):
    """Worker to train Faster R-CNN detection model."""

    def run(self):
        # Training dataset structure
        class DetectionDataset(Dataset):
            def __init__(self, data_dir):
                for fn in os.listdir(data_dir):
                    if fn.endswith("_bboxes.json"):
                        # JSON format:
                        # {
                        #   "image": "chunk_001.png",
                        #   "width": 1024,
                        #   "height": 1024,
                        #   "bboxes": [
                        #     {"label": "signal", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}
                        #   ]
                        # }

            def __getitem__(self, idx):
                img = Image.open(img_path).convert("RGB")
                target = {
                    "boxes": torch.tensor([[x_min, y_min, x_max, y_max], ...]),
                    "labels": torch.tensor([class_id, ...]),
                    "image_id": torch.tensor([idx])
                }
                return img, target

        # Model construction
        def build_model(num_classes, backbone, pretrained, trainable_layers):
            if backbone == "resnet50":
                m = fasterrcnn_resnet50_fpn(weights="DEFAULT")
            else:
                bb = resnet_fpn_backbone("resnet18", pretrained=pretrained, trainable_layers=trainable_layers)
                m = FasterRCNN(bb, num_classes=num_classes)
            return m

        # Training loop
        for epoch in range(epochs):
            model.train()
            for imgs, tgts in train_loader:
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()

            # Validation with IoU-based metrics
            model.eval()
            for imgs, tgts in val_loader:
                outputs = model(imgs)
                # Calculate TP/FP/FN with IoU >= 0.5

            # Early stopping based on F1
            if f1 > best_f1:
                best_f1 = f1
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= early_stop_patience:
                    break

        torch.save(model.state_dict(), out_model)
```

### 2.4 TENSORCADE ChunkWorker (Data Preparation)

```python
class ChunkWorker(QObject):
    """Convert IQ + labels to spectrogram images + bbox JSON."""

    def run(self):
        # Process each IQ file
        for datafile in data_files:
            # Load IQ data
            iq_data = np.frombuffer(raw, dtype=np.complex64)

            # Load labels (time/freq boxes)
            with open(labels_path) as f:
                signals = json.load(f).get("signals", [])
            # signals format:
            # [{"time_start": 0.1, "time_stop": 0.3, "freq_low": 915.0, "freq_high": 916.0, "label": "signal"}, ...]

            # Process each chunk
            for chunk_idx in range(n_chunks):
                chunk_data = iq_data[start:end]

                # Generate spectrogram
                f_vals, t_vals, Zxx = stft(chunk_data, fs=sr, nperseg=nfft, noverlap=noverlap)
                sxx_db = 10 * np.log10(np.abs(Zxx)**2 + 1e-12)

                # Clamp to dynamic range
                vmax = sxx_db.max()
                vmin = vmax - dynamic_range
                sxx_clamped = np.clip(sxx_db, vmin, vmax)

                # Resize to out_size x out_size
                resized = cv2.resize(sxx_clamped, (out_size, out_size))
                resized = np.flipud(resized)  # IMPORTANT: Flip vertical

                # Convert signal boxes to chunk-local pixel coordinates
                for sig in signals:
                    if signal_overlaps_chunk(sig, chunk_start, chunk_end):
                        bbox = convert_to_pixel_coords(sig, chunk, freqs)
                        tile_bboxes.append(bbox)

                # Only save if has labels
                if tile_bboxes:
                    plt.imsave(out_png, resized, cmap="gray")
                    json.dump({"image": ..., "bboxes": tile_bboxes}, f)
```

---

## SECTION 3: DATABASE SCHEMA

### 3.1 Signal Database (lib/core/database/signal_database.dart)

**Storage:** JSON file at `config/signals.json`

```dart
class SignalEntry {
  final String id;
  String name;                    // e.g., "creamy_chicken"
  String modType;                 // e.g., "OFDM", "GFSK", "--"
  double? modRate;                // Symbol rate in sps
  double? bandwidth;              // Signal bandwidth in kHz
  String? notes;

  // Training stats
  int totalDataLabels;            // Total labeled samples across all training
  double? f1Score;                // Best/latest F1 score
  double? precision;
  double? recall;
  int timesAbove90;               // Detection count with >90% confidence

  List<TrainingResult> trainingHistory;
  DateTime created;
  DateTime modified;
}

class TrainingResult {
  final DateTime timestamp;
  final int dataLabels;           // Number of labeled samples used
  final double f1Score;
  final double precision;
  final double recall;
  final int epochs;
  final double? loss;
  final String? modelPath;        // e.g., "models/creamy_chicken.engine"
}
```

### 3.2 Database Provider

```dart
final signalDatabaseProvider = StateNotifierProvider<SignalDatabaseNotifier, List<SignalEntry>>(
  (ref) => SignalDatabaseNotifier(),
);

// Update on training completion
void addTrainingResult(String signalName, TrainingResult result) {
  final existing = getByName(signalName);
  if (existing != null) {
    existing.addTrainingResult(result);
  } else {
    // Create new signal entry
    addSignal(SignalEntry(name: signalName, ...));
  }
  _saveToDisk();
}
```

---

## SECTION 4: MISSION SYSTEM

### 4.1 Mission Configuration Model

**Storage:** YAML files in `config/missions/*.mission.yaml`

```yaml
# Example mission file
name: "Urban RF Survey"
description: "Scan urban RF environment"
created: 2026-01-20T10:00:00Z
modified: 2026-01-25T15:30:00Z

# SDR Configuration
rx_bandwidth_mhz: 20.0
dwell_time_ms: 100
sample_rate_mhz: 20.0

# Frequency ranges to scan
frequency_ranges:
  - start_mhz: 800.0
    end_mhz: 900.0
  - start_mhz: 1800.0
    end_mhz: 1900.0

# Signal models to use (priority order)
signal_priority:
  - "creamy_chicken"
  - "lte_uplink"
  - "wifi_24"

# Display settings
colormap: "viridis"
fft_size: 32768
time_span_seconds: 5.0
```

### 4.2 Mission Provider (lib/features/config/providers/mission_provider.dart)

```dart
class MissionNotifier extends StateNotifier<MissionState> {
  Future<bool> loadMission(String filePath) async {
    final mission = await MissionConfig.loadFromFile(filePath);
    state = state.copyWith(activeMission: mission);
    return true;
  }

  Future<bool> saveMission({String? newPath}) async {
    final mission = state.activeMission;
    await mission.saveToFile(savePath);
    return true;
  }
}

final missionProvider = StateNotifierProvider<MissionNotifier, MissionState>(
  (ref) => MissionNotifier(),
);
```

---

## SECTION 5: LIVE VIEW & DETECTIONS

### 5.1 Detection Model (lib/features/live_detection/models/detection.dart)

```dart
class Detection {
  final String id;
  final int classId;
  final String className;
  final double confidence;
  final double x1, y1, x2, y2;     // Normalized 0-1
  final double freqMHz;
  final double bandwidthMHz;
  final String mgrsLocation;
  final double latitude, longitude;
  final DateTime timestamp;
  final bool isSelected;
  final bool isActive;
  final int trackId;
  final double pts;                // Presentation timestamp
  final int absoluteRow;           // Row index in waterfall
}
```

### 5.2 Video Stream Provider (lib/features/live_detection/providers/video_stream_provider.dart)

**WebSocket message types:**
```dart
class MessageType {
  static const int strip = 0x01;      // Row strip data (binary)
  static const int detection = 0x02;  // Detection JSON
  static const int metadata = 0x03;   // Stream metadata JSON
}
```

**Strip message format (binary):**
```
Header (17 bytes):
├─ frame_id:     uint32 (4 bytes)
├─ total_rows:   uint32 (4 bytes)  ← monotonic counter
├─ rows_in_strip: uint16 (2 bytes)
├─ strip_width:  uint16 (2 bytes)
├─ pts:          float32 (4 bytes)
└─ source_id:    uint8 (1 byte)    ← 0=SCAN, 1=RX1_REC, 2=RX2_REC, 3=MANUAL

Pixel data:
└─ RGBA bytes: rows_in_strip × strip_width × 4

PSD data:
└─ Float32 dB values: strip_width × 4 bytes
```

### 5.3 Detection Flow

```
Backend (Python)                    Frontend (Flutter)
─────────────────                   ──────────────────
IQ samples
    ↓
GPU FFT (waterfall)
    ↓
RGBA strip + PSD dB
    ↓
WebSocket 0x01 ─────────────────→ _handleStrip()
                                      ↓
                                  Scroll pixel buffer
                                  Paste new rows at bottom
                                      ↓
                                  Update state.pixelBuffer

Inference (every 6 frames)
    ↓
Faster R-CNN
    ↓
Detection boxes
    ↓
WebSocket 0x02 ─────────────────→ _handleDetection()
                                      ↓
                                  Parse JSON
                                  Add to _detectionBuffer
                                      ↓
                                  Update state.detections
```

---

## SECTION 6: MANUAL COLLECTION & LABELING

### 6.1 RFCAP File Format (lib/core/services/rfcap_service.dart)

```
G20 RFCAP File Format (512-byte header + IQ data)

| Offset | Size | Type     | Field              |
|--------|------|----------|--------------------|
| 0      | 4    | char[4]  | Magic ("G20\0")    |
| 4      | 4    | uint32   | Version            |
| 8      | 8    | float64  | Sample rate (Hz)   |
| 16     | 8    | float64  | Center freq (Hz)   |
| 24     | 8    | float64  | Bandwidth (Hz)     |
| 32     | 8    | uint64   | Number of samples  |
| 40     | 8    | float64  | Start time (epoch) |
| 48     | 32   | char[32] | Signal name        |
| 80     | 8    | float64  | Latitude           |
| 88     | 8    | float64  | Longitude          |
| 96     | 416  | reserved | (zeros)            |

After header: complex64 IQ data (float32 I, float32 Q pairs)
```

### 6.2 Manual Capture Flow

1. User draws box on waterfall → creates `LabelBox`
2. Backend captures IQ during box time range
3. Saves as `.rfcap` file with signal name from box label
4. File appears in Training screen dropdown

### 6.3 Label Storage

Labels are stored **per-file** in memory during Training session:

```dart
// TrainingScreen state
final Map<String, List<LabelBox>> _boxesByFile = {};

class LabelBox {
  int id;
  String className;
  double? timeStartSec;
  double? timeEndSec;
  double? freqStartMHz;
  double? freqEndMHz;
  Rect? pixelRect;
  bool isSelected;
}
```

**Not yet implemented:** Persisting labels to JSON files alongside `.rfcap` files.

---

## SECTION 7: PROVIDER ARCHITECTURE

### Key Riverpod Providers

```dart
// Backend management
final backendLauncherProvider = StateNotifierProvider<BackendLauncherNotifier, BackendLauncherState>

// Video/waterfall streaming
final videoStreamProvider = StateNotifierProvider<VideoStreamNotifier, VideoStreamState>

// Detection state
final detectionProvider = StateNotifierProvider<DetectionNotifier, List<Detection>>

// Signal database
final signalDatabaseProvider = StateNotifierProvider<SignalDatabaseNotifier, List<SignalEntry>>

// Mission configuration
final missionProvider = StateNotifierProvider<MissionNotifier, MissionState>

// SDR configuration
final sdrConfigProvider = StateNotifierProvider<SdrConfigNotifier, SdrConfigState>

// Settings (FFT size, colormap, etc.)
final settingsProvider = StateNotifierProvider<SettingsNotifier, SettingsState>
```

---

## SECTION 8: CURRENT MODEL ARCHITECTURE

### 8.1 Neural Network

**Architecture:** Faster R-CNN with ResNet18-FPN backbone

```python
# From inference.py
backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
model = torchvision.models.detection.FasterRCNN(backbone, num_classes=2)

# 2 classes: background (0), signal (1)
```

### 8.2 Model Input/Output

**Input:**
- Shape: `[batch, 3, 1024, 1024]`
- Type: FP16 (if GPU)
- Range: 0-1 normalized
- Grayscale expanded to 3 channels

**Output:**
- `boxes`: `[N, 4]` pixel coordinates (0-1024)
- `scores`: `[N]` confidence scores (0-1)
- `labels`: `[N]` class IDs

### 8.3 Post-processing

```python
# Score threshold
mask = scores >= score_threshold  # Default 0.5

# NMS is built into Faster R-CNN RoI heads
# Default IoU threshold ~0.5
```

---

## SECTION 9: FILE STRUCTURE

```
g20_demo/
├── lib/                              # Flutter app
│   ├── core/
│   │   ├── database/
│   │   │   └── signal_database.dart  # Signal metadata storage
│   │   ├── services/
│   │   │   ├── backend_launcher.dart # Python process management
│   │   │   └── rfcap_service.dart    # IQ file format
│   │   └── grpc/
│   │       └── connection_manager.dart
│   └── features/
│       ├── live_detection/
│       │   ├── providers/
│       │   │   └── video_stream_provider.dart  # WebSocket client
│       │   └── models/
│       │       └── detection.dart
│       ├── training/
│       │   └── training_screen.dart  # Labeling UI (training simulated)
│       ├── config/
│       │   └── providers/
│       │       └── mission_provider.dart
│       └── database/
│           └── database_screen.dart  # Signal catalog view
│
├── backend/                          # Python backend
│   ├── server.py                     # WebSocket + gRPC server
│   ├── unified_pipeline.py           # GPU FFT + inference
│   ├── gpu_fft.py                    # CUDA FFT processing
│   ├── inference.py                  # TensorRT/PyTorch engine
│   └── generated/                    # gRPC stubs
│
├── config/
│   ├── signals.json                  # Signal database
│   ├── spectrogram.yaml              # FFT parameters
│   └── missions/                     # .mission.yaml files
│
├── models/
│   └── creamy_chicken_fold3.pth      # Trained model
│
├── data/
│   └── captures/                     # .rfcap files
│
└── protos/
    ├── control.proto                 # SDR control
    └── inference.proto               # ML inference
```

---

## SECTION 10: CRITICAL PATHS

### 10.1 Detection End-to-End Flow

```
IQ File (.sigmf-data)
    │
    └─► UnifiedIQSource.read_chunk(33ms)
            │
            └─► TripleBufferedPipeline
                    │
                    ├─► compute_waterfall_rows() [GPU FFT]
                    │       │
                    │       └─► GPUSpectrogramProcessor.process()
                    │               - torch.fft.fft (batched)
                    │               - 20 rows decimated output
                    │               - Apply colormap → RGBA
                    │
                    └─► process_chunk() [Inference, every 6 frames]
                            │
                            └─► compute_spectrogram()
                            │       - 4096 FFT, 80dB
                            │       - Resize to 1024x1024
                            │       - Expand to 3 channels
                            │
                            └─► model(spec.half())
                                    │
                                    └─► Detection objects
    │
    └─► WebSocket
            │
            ├─► 0x01: Strip (RGBA + PSD)
            │
            └─► 0x02: Detections (JSON)
    │
    └─► Flutter
            │
            ├─► _handleStrip()
            │       - Scroll pixel buffer
            │       - Paste new rows
            │
            └─► _handleDetection()
                    - Parse JSON
                    - Add to detection list
```

### 10.2 Training End-to-End Flow (TENSORCADE)

```
IQ File + Labels JSON
    │
    └─► ChunkWorker
            │
            └─► For each chunk with labels:
                    │
                    ├─► Generate spectrogram (STFT)
                    ├─► Resize to 1024x1024
                    ├─► Flip vertical
                    ├─► Save as grayscale PNG
                    └─► Save bboxes as JSON
    │
    └─► TrainWorker
            │
            └─► DetectionDataset
                    - Load PNG images
                    - Load bbox JSON
                    - Build label map
            │
            └─► build_model()
                    - ResNet18/50 FPN backbone
                    - Faster R-CNN detector
            │
            └─► Training loop
                    - Forward pass (model returns losses)
                    - Backward pass
                    - Validation with IoU-based metrics
                    - Early stopping
            │
            └─► torch.save(model.state_dict(), out_model)
```

### 10.3 WebSocket Message Types

| Type | Direction | Format | Content |
|------|-----------|--------|---------|
| 0x01 (Strip) | Server→Client | Binary | 17-byte header + RGBA pixels + PSD dB |
| 0x02 (Detection) | Server→Client | JSON | `{type, frame_id, pts, inference_ms, base_row, detections}` |
| 0x03 (Metadata) | Server→Client | JSON | `{mode, strip_width, rows_per_strip, video_fps, ...}` |

**Commands (Client→Server):**
```json
{"command": "set_time_span", "seconds": 5.0}
{"command": "set_fps", "fps": 30}
{"command": "set_fft_size", "size": 32768}
{"command": "set_colormap", "colormap": 0}
{"command": "set_score_threshold", "threshold": 0.5}
{"command": "set_db_range", "min_db": -100, "max_db": -20}
{"command": "stop"}
```

---

## Summary

### What EXISTS and WORKS:
1. ✅ Faster R-CNN inference (PyTorch FP16, TensorRT fallback)
2. ✅ GPU FFT waterfall streaming (row-strip protocol)
3. ✅ Signal database (JSON persistence)
4. ✅ Mission configuration (YAML files)
5. ✅ Manual capture (.rfcap format)
6. ✅ Detection display with row-sync

### What is SIMULATED/PLACEHOLDER:
1. ⚠️ Training in Flutter (progress bar only, no actual training)
2. ⚠️ Label persistence (in-memory only per session)

### What NEEDS IMPLEMENTATION:
1. ❌ Actual PyTorch training in backend
2. ❌ Label file persistence (.json alongside .rfcap)
3. ❌ Multi-model simultaneous inference
4. ❌ DLA offloading (Orin NX)
5. ❌ INT8 quantization
6. ❌ Shared backbone + heads (HydraNet)

### CRITICAL PARAMETERS (DO NOT CHANGE WITHOUT RETRAINING):
```yaml
# Inference FFT (matches model training)
fft_size: 4096
hop_length: 2048
dynamic_range_db: 80.0
output_size: 1024x1024
```
