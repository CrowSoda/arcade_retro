# Training Data Collection Flow

## Critical Design Decision

**Q: Who computes training spectrograms?**  
**A: The BACKEND must compute them using INFERENCE FFT parameters.**

Why? Because training data must match inference data exactly:
- **Inference FFT**: 4096 FFT, 2048 hop, 80dB dynamic range, 1024×1024 output
- **Flutter/Dart FFT**: Different params (2048 FFT, configurable), used for display only

If Flutter computed spectrograms, the model would be trained on different data than it sees during inference.

---

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLUTTER (Training Screen)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. User loads RFCAP file                                                   │
│     └─► RfcapService.readIqData(file, offset, samples)                     │
│                                                                             │
│  2. User sees spectrogram (computed with Dart FFT - display only)          │
│                                                                             │
│  3. User draws bounding box(es) on signal                                  │
│     └─► LabelBox { x1, y1, x2, y2, className }                             │
│                                                                             │
│  4. User clicks "Save Sample"                                              │
│     │                                                                       │
│     ├─► Extract IQ chunk from RFCAP:                                       │
│     │     - time_offset (seconds)                                          │
│     │     - duration (seconds)                                             │
│     │     - Raw IQ bytes (complex64)                                       │
│     │                                                                       │
│     ├─► Package box coordinates:                                           │
│     │     - Normalized 0-1 (relative to display window)                    │
│     │                                                                       │
│     └─► Send via WebSocket:                                                │
│           {                                                                 │
│             "command": "save_training_sample",                             │
│             "signal_name": "creamy_chicken",                               │
│             "iq_data_b64": "<base64 encoded IQ>",                          │
│             "boxes": [{ "x1": 0.4, "y1": 0.2, "x2": 0.6, "y2": 0.5 }],    │
│             "metadata": {                                                  │
│               "source_file": "MAN_024307ZJAN26_2430.rfcap",               │
│               "time_offset_sec": 0.05,                                     │
│               "duration_sec": 0.2,                                         │
│               "center_freq_mhz": 2430.0,                                   │
│               "sample_rate_mhz": 20.0                                      │
│             }                                                              │
│           }                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND (Training Service)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  5. Receive IQ data + boxes                                                │
│                                                                             │
│  6. Decode base64 IQ → numpy complex64 array                               │
│                                                                             │
│  7. Compute spectrogram using INFERENCE PARAMS:                            │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  NFFT = 4096                                                    │    │
│     │  hop_length = 2048                                              │    │
│     │  dynamic_range = 80.0 dB                                        │    │
│     │  output_size = (1024, 1024)                                     │    │
│     │                                                                 │    │
│     │  # Use SpectrogramPipeline from inference.py                    │    │
│     │  pipeline = SpectrogramPipeline(device)                         │    │
│     │  spec = pipeline.compute_spectrogram(iq_data)  # Returns tensor │    │
│     │  spec_np = (spec * 255).byte().cpu().numpy()   # uint8          │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  8. Convert normalized boxes to pixel coordinates:                         │
│     x_min_px = int(box["x1"] * 1024)                                       │
│     y_min_px = int(box["y1"] * 1024)                                       │
│     x_max_px = int(box["x2"] * 1024)                                       │
│     y_max_px = int(box["y2"] * 1024)                                       │
│                                                                             │
│  9. Generate unique sample ID:                                             │
│     sample_id = f"{len(existing_samples)+1:04d}"  # "0001", "0002", ...    │
│                                                                             │
│  10. Save files:                                                           │
│      training_data/signals/creamy_chicken/samples/                         │
│      ├── 0042.npz   (compressed spectrogram)                              │
│      └── 0042.json  (boxes + metadata)                                    │
│                                                                             │
│  11. Update manifest.json                                                  │
│                                                                             │
│  12. Send confirmation:                                                    │
│      {"type": "sample_saved", "signal_name": "...", "sample_id": "0042"}  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File: `backend/training/sample_manager.py`

```python
"""
SampleManager - Save and manage training samples.

Handles the flow: IQ data + boxes → computed spectrogram → saved files
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch

# Import inference spectrogram pipeline to ensure matching params
from ..inference import SpectrogramPipeline


class SampleManager:
    """Manages training sample storage."""
    
    def __init__(self, training_data_dir: str = "training_data/signals", device: str = "cuda"):
        self.base_dir = Path(training_data_dir)
        self.device = device
        
        # Use the SAME spectrogram computation as inference
        self.spec_pipeline = SpectrogramPipeline(device)
    
    def save_sample(
        self,
        signal_name: str,
        iq_data_b64: str,
        boxes: List[Dict],
        metadata: Dict
    ) -> str:
        """
        Save a training sample.
        
        Args:
            signal_name: Signal class name
            iq_data_b64: Base64-encoded IQ data (complex64)
            boxes: List of normalized box dicts [{x1, y1, x2, y2}, ...]
            metadata: Source file info, frequencies, etc.
        
        Returns:
            Sample ID (e.g., "0042")
        """
        # Create directories
        signal_dir = self.base_dir / signal_name
        samples_dir = signal_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Decode IQ data
        iq_bytes = base64.b64decode(iq_data_b64)
        iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
        
        # Compute spectrogram using INFERENCE params
        # This ensures training data matches inference data exactly
        spec_tensor = self.spec_pipeline.compute_spectrogram(iq_data)
        
        # Convert to uint8 for storage efficiency
        # spec_tensor is (1, 3, 1024, 1024) float32 normalized 0-1
        # Take first channel (grayscale), scale to 0-255
        spec_gray = spec_tensor[0, 0].cpu().numpy()  # (1024, 1024)
        spec_uint8 = (spec_gray * 255).astype(np.uint8)
        
        # Generate sample ID
        existing = list(samples_dir.glob("*.npz"))
        next_id = len(existing) + 1
        sample_id = f"{next_id:04d}"
        
        # Convert normalized boxes to pixel coordinates
        pixel_boxes = []
        for box in boxes:
            pixel_boxes.append({
                "x_min": int(box["x1"] * 1024),
                "y_min": int(box["y1"] * 1024),
                "x_max": int(box["x2"] * 1024),
                "y_max": int(box["y2"] * 1024),
                "label": "signal",
                "confidence": None,
            })
        
        # Save spectrogram
        npz_path = samples_dir / f"{sample_id}.npz"
        np.savez_compressed(npz_path, spectrogram=spec_uint8)
        
        # Save metadata
        sample_metadata = {
            "sample_id": sample_id,
            "source_capture": metadata.get("source_file"),
            "time_offset_sec": metadata.get("time_offset_sec"),
            "duration_sec": metadata.get("duration_sec"),
            "center_freq_mhz": metadata.get("center_freq_mhz"),
            "sample_rate_mhz": metadata.get("sample_rate_mhz"),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "created_by": "user_label",
            "spectrogram_params": {
                "fft_size": 4096,
                "hop_length": 2048,
                "dynamic_range_db": 80.0,
                "output_size": [1024, 1024],
            },
            "boxes": pixel_boxes,
        }
        
        json_path = samples_dir / f"{sample_id}.json"
        with open(json_path, "w") as f:
            json.dump(sample_metadata, f, indent=2)
        
        # Update manifest
        self._update_manifest(signal_name, sample_id, metadata)
        
        print(f"Saved sample {sample_id} for {signal_name}")
        return sample_id
    
    def _update_manifest(self, signal_name: str, sample_id: str, metadata: Dict):
        """Update the signal's manifest file."""
        signal_dir = self.base_dir / signal_name
        manifest_path = signal_dir / "manifest.json"
        
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {
                "signal_name": signal_name,
                "total_samples": 0,
                "sources": [],
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        
        manifest["total_samples"] += 1
        manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Track source file
        source_file = metadata.get("source_file")
        source_entry = next((s for s in manifest["sources"] if s["capture_file"] == source_file), None)
        if source_entry:
            source_entry["sample_count"] += 1
        else:
            manifest["sources"].append({
                "capture_file": source_file,
                "sample_count": 1,
                "added_at": datetime.utcnow().isoformat() + "Z",
            })
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    def get_sample_count(self, signal_name: str) -> int:
        """Get total sample count for a signal."""
        samples_dir = self.base_dir / signal_name / "samples"
        if not samples_dir.exists():
            return 0
        return len(list(samples_dir.glob("*.npz")))
    
    def delete_sample(self, signal_name: str, sample_id: str) -> bool:
        """Delete a sample."""
        samples_dir = self.base_dir / signal_name / "samples"
        npz_path = samples_dir / f"{sample_id}.npz"
        json_path = samples_dir / f"{sample_id}.json"
        
        deleted = False
        if npz_path.exists():
            npz_path.unlink()
            deleted = True
        if json_path.exists():
            json_path.unlink()
            deleted = True
        
        return deleted
```

---

## WebSocket Handler Addition

Add to `backend/server.py`:

```python
from training.sample_manager import SampleManager

class Server:
    def __init__(self):
        # ... existing init ...
        self.sample_manager = SampleManager()
    
    async def handle_command(self, websocket, data):
        command = data.get("command")
        
        # ... existing commands ...
        
        elif command == "save_training_sample":
            try:
                sample_id = self.sample_manager.save_sample(
                    signal_name=data["signal_name"],
                    iq_data_b64=data["iq_data_b64"],
                    boxes=data["boxes"],
                    metadata=data["metadata"]
                )
                await websocket.send(json.dumps({
                    "type": "sample_saved",
                    "signal_name": data["signal_name"],
                    "sample_id": sample_id,
                    "total_samples": self.sample_manager.get_sample_count(data["signal_name"])
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "sample_save_failed",
                    "error": str(e)
                }))
        
        elif command == "delete_training_sample":
            success = self.sample_manager.delete_sample(
                data["signal_name"],
                data["sample_id"]
            )
            await websocket.send(json.dumps({
                "type": "sample_deleted",
                "success": success
            }))
        
        elif command == "get_sample_count":
            count = self.sample_manager.get_sample_count(data["signal_name"])
            await websocket.send(json.dumps({
                "type": "sample_count",
                "signal_name": data["signal_name"],
                "count": count
            }))
```

---

## Flutter Integration

Add to `lib/features/training/training_screen.dart`:

```dart
Future<void> _saveCurrentSample() async {
  if (_selectedFile == null || _loadedHeader == null) return;
  
  final boxes = _boxesByFile[_selectedFile] ?? [];
  if (boxes.isEmpty) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('No boxes drawn')),
    );
    return;
  }
  
  // Read IQ chunk from current window
  final iqData = await RfcapService.readIqDataBytes(
    _selectedFile!,
    offsetSamples: (_windowStartSec * _sampleRate).toInt(),
    numSamples: (_windowLengthSec * _sampleRate).toInt(),
  );
  
  // Convert boxes to normalized coords
  final normalizedBoxes = boxes.map((box) => {
    'x1': box.x1,
    'y1': box.y1,
    'x2': box.x2,
    'y2': box.y2,
  }).toList();
  
  // Send to backend
  final ws = ref.read(webSocketProvider);
  ws.send({
    'command': 'save_training_sample',
    'signal_name': _selectedSignal,
    'iq_data_b64': base64Encode(iqData),
    'boxes': normalizedBoxes,
    'metadata': {
      'source_file': path.basename(_selectedFile!),
      'time_offset_sec': _windowStartSec,
      'duration_sec': _windowLengthSec,
      'center_freq_mhz': _loadedHeader!.centerFreqMHz,
      'sample_rate_mhz': _loadedHeader!.sampleRate / 1e6,
    },
  });
  
  // Clear boxes after save
  setState(() {
    _boxesByFile[_selectedFile!] = [];
  });
  
  ScaffoldMessenger.of(context).showSnackBar(
    const SnackBar(content: Text('Sample saved')),
  );
}
```

---

## Key Points Summary

1. **IQ data flows from Flutter → Backend** (not spectrograms)
2. **Backend computes spectrograms** using locked inference params
3. **Boxes are normalized 0-1** in Flutter, converted to pixels in backend
4. **Each sample is 2 files**: `.npz` (spectrogram) + `.json` (boxes/metadata)
5. **Manifest tracks sources** for auditing where samples came from
