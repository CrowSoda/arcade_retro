# G20 Testing Plan - Path to 90% Coverage

## Current State
- **Overall**: ~35% (was 19.49%)
- **Backend (Python)**: 42% (3450 of 8192 lines covered) - was 29.54%
- **Flutter (Dart)**: 9.73% (726 of 7462 lines) - unchanged

### Recent Progress (Jan 2026)
| Module | Before | After | Change |
|--------|--------|-------|--------|
| colormaps.py | 0% | 100% | +43 lines |
| logger_config.py | 67% | 89% | +24 lines |
| dsp/subband_extractor.py | 16% | 59% | +117 lines |
| hydra/detector.py | 19% | 58% | +89 lines |
| training/dataset.py | 18% | 86% | +91 lines |
| **TOTAL** | 30% | 42% | +12% |

New test files added:
- `test_quick_wins.py` - colormaps, logger, runtime tests
- `test_dsp_extended.py` - SubbandExtractor, header I/O tests
- `test_hydra_extended.py` - HydraDetector, registry, config tests
- `test_training_extended.py` - SpectrogramDataset, transforms, data loaders

## Target: 90% Coverage
- Backend: 6530 lines covered (currently 2143 → need +4387 lines)
- Flutter: 6716 lines covered (currently 726 → need +5990 lines)

---

## Phase 1: Backend Python Tests (High ROI)

### 1.1 DSP Module Tests (Estimated +1500 lines covered)
**Files to test:**
- `dsp/subband_extractor.py` - Subband extraction logic
- `dsp/simple_extract.py` - Simple extraction utilities
- `gpu_fft.py` - FFT operations

**Test categories:**
```python
# tests/test_dsp_extended.py
- test_subband_extraction_center_frequency()
- test_subband_extraction_edge_cases()
- test_fft_window_functions()
- test_fft_output_shapes()
- test_decimation_factors()
- test_heterodyne_mixing()
```

### 1.2 Hydra Detector Tests (Estimated +800 lines covered)
**Files to test:**
- `hydra/detector.py` - Main detection logic
- `hydra/backbone_extractor.py` - Feature extraction

**Test categories:**
```python
# tests/test_hydra_extended.py
- test_detector_initialization()
- test_detection_with_mock_model()
- test_nms_filtering()
- test_confidence_thresholding()
- test_backbone_feature_shapes()
```

### 1.3 Training Module Tests (Estimated +600 lines covered)
**Files to test:**
- `training/dataset.py` - Dataset handling
- `training/sample_manager.py` - Sample management

**Test categories:**
```python
# tests/test_training_extended.py
- test_dataset_loading()
- test_sample_validation()
- test_label_parsing()
- test_augmentation_pipeline()
```

### 1.4 Inference Pipeline Tests (Estimated +500 lines covered)
**Files to test:**
- `inference.py` - Main inference logic
- `unified_pipeline.py` - Pipeline orchestration

**Test categories:**
```python
# tests/test_inference_extended.py
- test_inference_request_handling()
- test_pipeline_state_transitions()
- test_batch_processing()
```

### 1.5 API/Server Tests (Estimated +1000 lines covered)
**Files to test:**
- `server.py` - FastAPI server
- `api/ws/handlers/*.py` - WebSocket handlers
- `api/grpc/*.py` - gRPC services

**Test categories:**
```python
# tests/test_api_extended.py
- test_websocket_connection()
- test_websocket_message_routing()
- test_grpc_service_methods()
- test_error_handling()
```

---

## Phase 2: Flutter/Dart Tests (High Impact)

### 2.1 Provider Unit Tests (Estimated +2000 lines covered)
**Files to test:**
- `features/live_detection/providers/sdr_config_provider.dart`
- `features/live_detection/providers/video_stream_provider.dart`
- `features/live_detection/providers/inference_provider.dart`
- `features/settings/providers/settings_providers.dart`
- `features/config/providers/mission_provider.dart`

**Test categories:**
```dart
// test/providers/sdr_config_test.dart
void main() {
  group('SDRConfigProvider', () {
    test('initial state is correct');
    test('tune updates frequency');
    test('bandwidth clamps to valid range');
  });
}
```

### 2.2 Model Tests (Estimated +500 lines covered)
**Files to test:**
- `features/config/models/mission_config.dart`
- `features/live_detection/models/detection.dart`
- `core/database/signal_database.dart`

**Test categories:**
```dart
// test/models/mission_config_test.dart
void main() {
  test('MissionConfig.copyWith creates new instance');
  test('MissionConfig.defaultConfig has valid defaults');
  test('MissionConfig serialization roundtrip');
}
```

### 2.3 Widget Tests (Estimated +2000 lines covered)
**Files to test:**
- `features/live_detection/widgets/detection_table.dart`
- `features/live_detection/widgets/psd_chart.dart`
- `features/settings/settings_screen.dart`
- `features/config/config_screen.dart`

**Test categories:**
```dart
// test/widgets/detection_table_test.dart
void main() {
  testWidgets('DetectionTable shows detections', (tester) async {
    await tester.pumpWidget(/* ... */);
    expect(find.text('Detection'), findsWidgets);
  });
}
```

### 2.4 Utility/Core Tests (Estimated +1500 lines covered)
**Files to test:**
- `core/utils/colormap.dart`
- `core/utils/dtg_formatter.dart`
- `core/grpc/connection_manager.dart`
- `core/services/backend_launcher.dart`

---

## Phase 3: Integration Tests

### 3.1 Backend Integration Tests
```python
# tests/test_e2e.py
- test_full_inference_pipeline()
- test_training_workflow()
- test_capture_workflow()
```

### 3.2 Flutter Integration Tests
```dart
// integration_test/app_test.dart
- test_navigation_between_screens()
- test_settings_persistence()
- test_mission_load_save()
```

---

## Implementation Priority

### Week 1: Backend Core (Target: 50% backend coverage)
1. [ ] `test_dsp_extended.py` - FFT, subband extraction
2. [ ] `test_hydra_extended.py` - Detection logic
3. [ ] `test_api_extended.py` - WebSocket handlers

### Week 2: Backend Complete (Target: 70% backend coverage)
4. [ ] `test_training_extended.py` - Training pipeline
5. [ ] `test_inference_extended.py` - Inference pipeline
6. [ ] `test_integration_extended.py` - E2E flows

### Week 3: Flutter Providers (Target: 40% Flutter coverage)
7. [ ] `test/providers/sdr_config_test.dart`
8. [ ] `test/providers/video_stream_test.dart`
9. [ ] `test/providers/inference_test.dart`

### Week 4: Flutter Widgets (Target: 70% Flutter coverage)
10. [ ] `test/widgets/detection_table_test.dart`
11. [ ] `test/widgets/psd_chart_test.dart`
12. [ ] `test/widgets/settings_test.dart`

### Week 5: Polish (Target: 90% overall)
13. [ ] Fill coverage gaps
14. [ ] Integration tests
15. [ ] Edge case tests

---

## Quick Wins (Immediate Impact)

### Backend - Add these tests now:
```python
# tests/test_quick_wins.py

def test_colormap_lookup():
    """Test colormap functions - easy coverage"""
    from colormaps import get_colormap
    cmap = get_colormap('viridis')
    assert cmap is not None

def test_waterfall_buffer_operations():
    """Test waterfall buffer - simple state machine"""
    from waterfall_buffer import WaterfallBuffer
    buf = WaterfallBuffer(width=512, height=256)
    buf.add_row([0.0] * 512)
    assert buf.row_count == 1

def test_logger_config():
    """Test logger setup - trivial coverage"""
    from logger_config import setup_logging
    logger = setup_logging()
    assert logger is not None
```

### Flutter - Add these tests now:
```dart
// test/quick_wins_test.dart

void main() {
  test('DTG formatter formats correctly', () {
    final formatted = formatDtg(DateTime(2024, 1, 15, 14, 30));
    expect(formatted, contains('15'));
  });

  test('Colormap generates valid colors', () {
    final color = getColormapColor('viridis', 0.5);
    expect(color.alpha, 255);
  });
}
```

---

## Test Infrastructure Needed

### Backend:
- [x] pytest configured (`pytest.ini`)
- [x] Coverage reporting (`--cov`)
- [ ] Mock fixtures for GPU/CUDA
- [ ] Test data fixtures (sample IQ files)

### Flutter:
- [x] Test directory structure
- [ ] Mock providers using `mocktail`
- [ ] Golden tests for complex widgets
- [ ] Test fixtures for sample data

---

## Running Tests

### Backend:
```bash
cd backend
pytest tests/ -v --cov=. --cov-report=html
# Open htmlcov/index.html to see coverage gaps
```

### Flutter:
```bash
flutter test --coverage
# Open coverage/lcov-report/index.html
```

---

## Coverage Targets by Module

| Module | Current | Target | Gap |
|--------|---------|--------|-----|
| backend/dsp | ~20% | 85% | +65% |
| backend/hydra | ~30% | 90% | +60% |
| backend/training | ~25% | 85% | +60% |
| backend/api | ~35% | 90% | +55% |
| lib/providers | ~10% | 90% | +80% |
| lib/widgets | ~5% | 80% | +75% |
| lib/core | ~15% | 90% | +75% |

---

## Next Immediate Actions

1. **Create test fixtures** - Sample data for testing
2. **Add mock infrastructure** - MockTensorRT, MockSDR
3. **Write quick-win tests** - Logger, colormap, simple utilities
4. **Iterate on high-impact modules** - Providers, detection logic
