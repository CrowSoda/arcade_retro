# Current Codebase Analysis

## Backend Python Files

### `backend/inference.py`

**Purpose**: TensorRT-accelerated inference with automatic fallback

**Key Classes**:

```python
class InferenceEngine:
    """Single model inference with TRT → ONNX → PyTorch fallback"""
    
    def __init__(self, model_path, num_classes=2, device="cuda", precision="fp16")
    def infer(self, images: torch.Tensor, score_threshold=0.9) -> List[Dict]
    
class MultiModelEngine:
    """Parallel inference across N models using CUDA streams"""
    
    def __init__(self, model_configs: List[Tuple[str, int]], device, precision)
    def infer_parallel(self, images, score_threshold) -> List[List[Dict]]
    def infer_merged(self, images, score_threshold, nms_threshold) -> List[Dict]

class SpectrogramPipeline:
    """GPU-accelerated B&W spectrogram for ResNet detection"""
    # NFFT=4096, noverlap=2048, 80dB, 1024x1024 output
```

**Important Details**:
- Model is FasterRCNN with ResNet18-FPN backbone
- Uses `trainable_layers=5` in backbone (full backbone training)
- FP16 inference on CUDA
- Spectrogram is grayscale expanded to 3 channels (NOT colormap)

### `backend/unified_pipeline.py`

**Purpose**: Main detection + waterfall streaming pipeline

**Key Classes**:

```python
class TripleBufferedPipeline:
    """Separate params for INFERENCE vs WATERFALL"""
    
    # INFERENCE (LOCKED - matches tensorcade):
    inference_fft_size = 4096
    inference_noverlap = 2048  
    inference_hop_length = 2048
    inference_dynamic_range = 80.0
    
    # WATERFALL (configurable):
    waterfall_fft_size = 65536  # Default
    waterfall_dynamic_range = 60.0
    
    def _load_model(self, model_path)  # Loads FasterRCNN
    def compute_spectrogram(self, iq_data) -> torch.Tensor  # For inference
    def process_chunk(self, iq_data, pts, score_threshold) -> Dict
    def compute_waterfall_rows(self, iq_data) -> np.ndarray

class VideoStreamServer:
    """Row-strip streaming server"""
    
    def run_pipeline(self, websocket)
    # Sends MSG_STRIP (0x01) for waterfall rows
    # Sends MSG_DETECTION (0x02) for detection JSON
```

**Detection Output Format**:
```python
Detection(
    box_id=int,
    x1=float,  # Normalized 0-1
    y1=float,
    x2=float,
    y2=float,
    confidence=float,
    class_id=int,
    class_name=str,
    parent_pts=float,
)
```

### `backend/server.py`

**Purpose**: WebSocket + gRPC server orchestration

**Key Features**:
- WebSocket router: `/video`, `/unified`, `/inference` paths
- Parent process watchdog (auto-exit if Flutter dies)
- Dynamic port allocation (port 0 → OS picks)
- Signal handlers for graceful shutdown

**Current WebSocket Commands**:
```json
{"command": "set_time_span", "seconds": 5.0}
{"command": "set_fps", "fps": 30}
{"command": "set_fft_size", "size": 32768}
{"command": "set_colormap", "colormap": 0}
{"command": "set_score_threshold", "threshold": 0.5}
{"command": "set_db_range", "min_db": -100, "max_db": -20}
{"command": "stop"}
```

### `backend/gpu_fft.py`

**Purpose**: GPU-accelerated batched FFT for waterfall

**Key Class**:
```python
class GPUSpectrogramProcessor:
    """Batched FFT with full pipeline on GPU"""
    
    VALID_FFT_SIZES = {8192, 16384, 32768, 65536}
    
    def process(self, iq_data) -> np.ndarray  # Returns dB rows
    def update_fft_size(self, new_size) -> dict  # Includes cuFFT warmup
```

---

## Frontend Dart Files

### `lib/core/database/signal_database.dart`

**Purpose**: Signal metadata persistence

**Key Classes**:
```dart
class TrainingResult {
  DateTime timestamp;
  int dataLabels;
  double f1Score;
  double precision;
  double recall;
  int epochs;
  double? loss;
  String? modelPath;
}

class SignalEntry {
  String id;
  String name;
  String modType;
  double? modRate;
  double? bandwidth;
  int totalDataLabels;
  double? f1Score;
  List<TrainingResult> trainingHistory;
}

class SignalDatabaseNotifier extends StateNotifier<List<SignalEntry>> {
  // Persists to config/signals.json
  void addTrainingResult(String signalName, TrainingResult result);
  SignalEntry? getByName(String name);
}
```

### `lib/features/training/training_screen.dart`

**Purpose**: Current training UI (SIMULATED)

**Key State**:
```dart
class _TrainingScreenState {
  String? _selectedFile;
  RfcapHeader? _loadedHeader;
  bool _isTraining = false;
  double _trainingProgress = 0.0;
  Map<String, List<LabelBox>> _boxesByFile = {};  // Per-file labels
}
```

**Current Training Flow**:
1. Load `.rfcap` file from `data/captures/` (only `man_*` files shown)
2. Draw bounding boxes on spectrogram
3. Click "Train Model" → SIMULATED progress 0-100%
4. Save fake metrics to SignalDatabase

**What Needs to Change**:
- Replace simulation with real backend training
- Add version history panel
- Add training notes input
- Show comparison with previous version

### `lib/features/training/widgets/training_spectrogram.dart`

**Purpose**: Spectrogram display with labeling

**Key Class**:
```dart
class LabelBox {
  double x1, y1, x2, y2;  // Normalized 0-1
  String className;
  int id;
  double? freqStartMHz, freqEndMHz;
  double? timeStartSec, timeEndSec;
}

class TrainingSpectrogramState {
  // Computes FFT using fftea package (Dart-native)
  // Supports zoom, pan, auto-detect signal regions
  // Click-to-detect uses Otsu + region growing + Chan-Vese fallback
}
```

### `lib/features/config/providers/mission_provider.dart`

**Purpose**: Mission configuration management

**Key Classes**:
```dart
class MissionState {
  MissionConfig? activeMission;
  List<String> availableMissions;
  bool isLoading;
}

class MissionConfig {
  String name;
  String description;
  double bandwidthMhz;
  double dwellTimeSec;
  List<FreqRange> freqRanges;
  List<ModelConfig> models;  // <-- SIGNAL PRIORITY LIST
}
```

---

## Configuration Files

### `config/missions.json`

Current format (single mission example):
```json
[{
  "id": "mission_1769351848366",
  "name": "Creamy",
  "description": "Chicken",
  "bandwidthMhz": 20.0,
  "dwellTimeSec": 5.0,
  "freqRanges": [{"startMhz": 2400.0, "endMhz": 2480.0}],
  "models": [{
    "id": "models\\creamy_chicken_fold3.pth",
    "name": "creamy_chicken_fold3",
    "filePath": "models\\creamy_chicken_fold3.pth",
    "signalType": null,
    "priority": 0
  }],
  "created": "2026-01-25T07:37:28.366850",
  "modified": "2026-01-25T07:45:42.570369"
}]
```

### `models/` Directory

Current contents:
```
models/
├── creamy_chicken_fold3.pth   # ~100MB full FasterRCNN
└── README.md
```

---

## Model Architecture Details

### FasterRCNN Structure (from `inference.py`):

```python
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN

backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
model = FasterRCNN(backbone, num_classes=2)
```

**Weight Categories in state_dict**:

1. **Backbone** (~45MB):
   - `backbone.body.conv1.*`
   - `backbone.body.bn1.*`
   - `backbone.body.layer1.*` through `layer4.*`
   
2. **FPN Neck** (~10MB):
   - `backbone.fpn.inner_blocks.*`
   - `backbone.fpn.layer_blocks.*`

3. **RPN Head** (~5MB):
   - `rpn.head.conv.*`
   - `rpn.head.cls_logits.*`
   - `rpn.head.bbox_pred.*`

4. **ROI Head** (~35MB):
   - `roi_heads.box_head.fc6.*`
   - `roi_heads.box_head.fc7.*`
   - `roi_heads.box_predictor.cls_score.*`
   - `roi_heads.box_predictor.bbox_pred.*`

**For Hydra Architecture**:
- **Shared**: Backbone + FPN (~55MB, frozen after extraction)
- **Per-Signal**: RPN + ROI (~40MB → after pruning ~10MB)

---

## Data Flow Summary

```
[.rfcap file]
     │
     ▼
[RfcapService.readIqData()]  (Flutter - Dart)
     │
     ▼
[TrainingSpectrogram._computeSpectrogram()]  (Dart FFT)
     │
     ▼
[User draws bounding boxes]
     │
     ▼
[Click "Train"] ──────────────────────────────────────┐
     │                                                 │
     ▼                                                 ▼
[CURRENTLY: Simulated]                    [FUTURE: WebSocket to backend]
     │                                                 │
     ▼                                                 ▼
[SignalDatabase.addTrainingResult()]      [TrainingService.train_head()]
```
