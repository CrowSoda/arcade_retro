# Mission Configuration System Design

## Overview

A unified configuration system that manages all operational parameters for the G20 system. Users can create, load, save, and switch between named **missions**.

## Mission File Format

Mission files use `.mission.yaml` extension and a flat/semi-flat structure for easy hand-editing.

### Example: `ism_band_hunt.mission.yaml`
```yaml
# G20 Mission Configuration
schema_version: 1

# === IDENTITY ===
name: "ISM Band Hunt"
description: "Scanning 900 MHz ISM band for IoT devices"
created: "2026-01-25T06:40:00Z"
modified: "2026-01-25T06:40:00Z"

# === FREQUENCY ===
center_freq_mhz: 915.0
bandwidth_mhz: 20.0
sample_rate_mhz: 20.0

# === SCAN BEHAVIOR ===
scan_mode: "fixed"              # fixed | sweep | hop
dwell_time_sec: 5.0
sweep_step_mhz: 10.0            # Only used if scan_mode: sweep
hop_frequencies_mhz: []         # Only used if scan_mode: hop

# === MODEL ===
model_name: "creamy_chicken_fold3"
model_path: "models/creamy_chicken_fold3.pth"
confidence_threshold: 0.5
nms_iou_threshold: 0.45

# === CLASSES ===
classes:
  - name: "background"
    color: "#808080"
    enabled: true
  - name: "creamy_chicken"
    color: "#FF6B35"
    enabled: true

# === PROCESSING ===
inference_fft_size: 4096        # MUST MATCH TRAINING - don't change
inference_hop_length: 2048
inference_dynamic_range_db: 80.0

waterfall_fft_size: 65536       # 8192 | 16384 | 32768 | 65536
waterfall_dynamic_range_db: 60.0
colormap: "viridis"

# === RECORDING ===
auto_record_detections: true
pre_trigger_sec: 1.0
post_trigger_sec: 2.0
min_recording_sec: 3.0
max_recording_sec: 30.0
recording_format: "sigmf"       # sigmf | raw | wav
recording_dir: "recordings/"

# === DISPLAY ===
waterfall_time_span_sec: 2.5
waterfall_fps: 30
auto_tune_delay_sec: null       # null = disabled
show_psd_chart: true
show_detection_boxes: true
```

## File Structure

```
g20_demo/
├── config/
│   ├── missions/                 # Named mission files
│   │   ├── default.mission.yaml
│   │   ├── ism_band_hunt.mission.yaml
│   │   └── cellular_scan.mission.yaml
│   ├── inference.yaml            # (legacy - will migrate)
│   └── spectrogram.yaml          # (legacy - will migrate)
└── lib/
    └── features/
        └── config/
            ├── models/
            │   └── mission_config.dart    # Mission data model
            ├── providers/
            │   └── mission_provider.dart  # Mission state management
            └── widgets/
                ├── mission_screen.dart    # Main mission page
                └── mission_dialog.dart    # Create/edit dialog
```

## UI Design

### Mission Page (Simplified)
```
┌─────────────────────────────────────────────────┐
│  Mission Configuration                           │
├─────────────────────────────────────────────────┤
│                                                 │
│  Active Mission: [ISM Band Hunt        ▼]       │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │ 915.0 MHz  |  BW: 20 MHz  |  Dwell: 5s  │   │
│  │ Model: creamy_chicken_fold3 @ 50%       │   │
│  │ Auto-record: ON                         │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  [+ New]  [Edit]  [Duplicate]  [Delete]         │
│                                                 │
├─────────────────────────────────────────────────┤
│  Available Missions                              │
│  ─────────────────────────────────────────────  │
│  • default.mission.yaml                         │
│  • ism_band_hunt.mission.yaml           ← active│
│  • cellular_scan.mission.yaml                   │
│  • lora_sweep.mission.yaml                      │
└─────────────────────────────────────────────────┘
```

### New/Edit Mission Dialog
```
┌─────────────────────────────────────────────────┐
│  Create New Mission                     [×]     │
├─────────────────────────────────────────────────┤
│                                                 │
│  Name: [_________________]                      │
│  Description: [_________________]               │
│                                                 │
│  ─── Frequency ───────────────────────────────  │
│  Center (MHz): [915.0    ]  BW (MHz): [20.0  ] │
│                                                 │
│  ─── Scan ────────────────────────────────────  │
│  Mode: (•) Fixed  ( ) Sweep  ( ) Hop           │
│  Dwell (sec): [5.0      ]                      │
│                                                 │
│  ─── Model ───────────────────────────────────  │
│  Model: [creamy_chicken_fold3      ▼]           │
│  Confidence: [=======|======] 50%               │
│                                                 │
│  ─── Recording ───────────────────────────────  │
│  [✓] Auto-record detections                     │
│  Pre/Post trigger: [1.0] / [2.0] sec            │
│                                                 │
│             [Cancel]        [Save Mission]      │
└─────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Data Model & Provider
1. Create `MissionConfig` data class (Dart)
2. Create `MissionProvider` for state management
3. Add YAML serialization/deserialization
4. Mission file list scanning

### Phase 2: Backend Integration
1. Add mission loading endpoint in `server.py`
2. Update `unified_pipeline.py` to load from mission file
3. Hot-reload support for safe settings

### Phase 3: UI Implementation
1. Mission screen with dropdown + list
2. New/edit mission dialog
3. Wire up to backend

### Phase 4: Polish
1. Mission validation
2. Default mission on first run
3. Migration from legacy configs

## Settings Hot-Reload Matrix

| Setting | Hot Reload | Requires Restart |
|---------|------------|------------------|
| `waterfall_fft_size` | ✅ | |
| `confidence_threshold` | ✅ | |
| `auto_tune_delay_sec` | ✅ | |
| `waterfall_fps` | ✅ | |
| `center_freq_mhz` | ⚠️ | RX hardware |
| `bandwidth_mhz` | ⚠️ | RX hardware |
| `model_path` | | ✅ |
| `sample_rate_mhz` | | ✅ |

## Schema Versioning

The `schema_version: 1` field allows future migrations:

```yaml
# Version 1 (current)
schema_version: 1

# Future version might add fields
schema_version: 2
rx_gain_db: 30.0  # New field in v2
```

Migration logic will handle upgrading old mission files.

## Answers to Previous Questions

1. **Storage location**: Local `config/missions/` folder (easy to backup/share)
2. **Sharing**: Just copy `.mission.yaml` files
3. **Hardware profiles**: Included in mission (center_freq, bandwidth, etc.)
4. **Templates**: Start with `default.mission.yaml`, duplicate to create new
5. **Validation**: Warn on unusual combos, don't block

---

Ready to implement Phase 1?
