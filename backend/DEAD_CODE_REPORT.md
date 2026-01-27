# Dead Code Analysis Report
Generated: 2026-01-27

## Summary
- **Total dead code items found:** 172 items (vulture scan @ 60% confidence)
- **100% confidence issues:** 5 items (definite dead code)
- **Confirmed unused modules:** 2 files (280 lines, never imported)

---

## 🔴 CRITICAL: Entire Modules Never Imported (SAFE TO DELETE)

These files are **never imported anywhere** in the codebase:

| File | Lines | Notes |
|------|-------|-------|
| `api/grpc/device_control.py` | 93 | Duplicated in server.py |
| `api/grpc/inference_service.py` | 187 | Duplicated in server.py |

**Total: 280 lines of completely dead code**

### Verification Command
```bash
grep -rn "from api.grpc\|import api.grpc" . --include="*.py" | grep -v __pycache__
# Returns empty - confirming modules never imported
```

---

## 🔴 100% Confidence Issues (Definite Problems)

| File | Line | Issue |
|------|------|-------|
| `config/settings.py` | 56 | unused variable `__context` |
| `gpu_fft.py` | 191 | unused variable `decimate_to` |
| `logger_config.py` | 97 | unused variable `modules` |
| `server.py` | 128 | unused import `extract_subband_from_file` |
| `unified_pipeline.py` | 1028 | **unsatisfiable 'if' condition** (logic bug!) |

---

## 🟡 Standalone Scripts (May Be Intentionally Standalone)

| File | Lines | Purpose |
|------|-------|---------|
| `debug_anchors.py` | 144 | Debug visualization tool |
| `verify_sample.py` | 132 | Sample validation script |
| `dsp/tests/test_extraction.py` | 204 | Test file in wrong location |

**Total: 480 lines** - Review before removing

---

## 🟡 Unused Functions/Methods by Module

### server.py (gRPC methods - may be called externally)
- `SetFrequency`, `SetBandwidth`, `SetGain` (lines 177-209)
- `StartCapture`, `StopCapture` (lines 223-255)
- `GetStatus`, `GetDeviceInfo`, `SetMode` (lines 272-311)
- `LoadModel`, `UnloadModel`, `GetModels` (lines 385-429)
- `StartInference`, `StopInference`, `StreamDetections` (lines 442-514)
- `StartTraining`, `StopTraining`, `GetTrainingStatus` (lines 621-634)
- `video_pipeline_handler` (line 1095)

**Note:** gRPC methods are called externally by clients, so vulture marks them unused.

### dsp/filters.py
- `analyze_filter` (line 122)
- `verify_stopband` (line 176)
- `design_halfband_filter` (line 207)
- `design_cic_compensation_filter` (line 241)

### training/dataset.py
- `_print_debug_sample` (line 59)
- `RandomHorizontalFlip` class (line 244)
- `RandomVerticalFlip` class (line 263)
- `Compose` class (line 282)

### hydra/detector.py
- `_switch_head` (line 292)
- `detect_single` (line 444)
- `get_head_info` (line 458)
- `get_memory_usage` (line 474)

### logger_config.py
- `configure_development` (line 175)
- `reset_perf_counters` (line 256)
- `is_perf_enabled` (line 262)
- `get_log_storage_used` (line 267)

### colormaps.py
- `get_colormap_by_name` (line 181)
- `apply_colormap_db` (line 215)
- `VIRIDIS_BACKGROUND` (line 240)

---

## 🟢 Likely False Positives

### config/settings.py - Pydantic Fields
Fields like `model_config`, `inference_hop`, `waterfall_dynamic_range_db` are **Pydantic model fields** that are accessed via attribute access, not detected by vulture.

### config/capabilities.py - Config Variables
Variables like `inference_locked`, `fps_options`, `tensorrt_available` are **exported configuration** used by frontend/clients.

---

## Recommended Actions

### Immediate Cleanup (High Confidence)
1. **Delete** `api/grpc/device_control.py` (93 lines)
2. **Delete** `api/grpc/inference_service.py` (187 lines)
3. **Remove** unused import in `server.py:128`
4. **Fix** unsatisfiable if condition in `unified_pipeline.py:1028`
5. **Remove** unused variable `decimate_to` in `gpu_fft.py:191`

### Review Before Removing
1. Audit `debug_anchors.py`, `verify_sample.py` - standalone scripts
2. Audit augmentation classes in `training/dataset.py` - may be used in training
3. Audit filter functions in `dsp/filters.py` - may be used for DSP development

### Keep (External API)
- gRPC methods in `server.py` - called by external clients
- Pydantic fields in `config/settings.py` - used via attribute access
- Config exports in `config/capabilities.py` - used by frontend

---

## Line Count Summary

| Category | Lines | Action |
|----------|-------|--------|
| Confirmed dead modules | 280 | Delete |
| 100% confidence issues | 5 | Fix |
| Standalone scripts | 480 | Review |
| False positives (config) | ~200 | Keep |
| External API methods | ~500 | Keep |

**Net removable:** ~280-760 lines depending on review
