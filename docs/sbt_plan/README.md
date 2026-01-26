# Sub-Band Tuning (SBT) Implementation Plan

This directory contains the detailed implementation plan for proper sub-band extraction for CNN training data.

## Document Index

| Part | File | Description |
|------|------|-------------|
| 1 | [01_overview.md](01_overview.md) | Executive summary, current state analysis, goals |
| 2 | [02_dsp_pipeline.md](02_dsp_pipeline.md) | DSP theory: mix → filter → decimate |
| 3 | [03_implementation.md](03_implementation.md) | Backend Python implementation |
| 4 | [04_flutter_integration.md](04_flutter_integration.md) | Flutter UI and provider changes |
| 5 | [05_testing.md](05_testing.md) | Test cases, validation, benchmarks |

## Quick Summary

### Problem
Currently G20 captures full 20 MHz wideband IQ. For training, we need narrowband sub-bands extracted with proper DSP to avoid aliasing artifacts that corrupt CNN training.

### Solution
Implement proper sub-band extraction pipeline:
1. **Frequency shift** - NCO mix to move signal to baseband
2. **Anti-aliasing filter** - 60-80 dB stopband FIR
3. **Decimation** - Rational resampling to target rate

### Key Specs
| Parameter | Value |
|-----------|-------|
| Source rate | 20 MHz |
| Target rate | 2.5 × signal bandwidth |
| Stopband attenuation | 60-80 dB |
| DC offset removal | Yes |
| Normalization | Unit power |

### Files to Create/Modify

**New:**
```
backend/dsp/
├── __init__.py
├── subband_extractor.py    # Main extraction class
├── filters.py              # Filter design helpers
└── tests/
    └── test_extraction.py  # Unit tests
```

**Modify:**
```
backend/scripts/capture_subband.py  # Use new DSP module
backend/server.py                   # Add WebSocket commands
lib/features/live_detection/providers/sdr_config_provider.dart  # Add extraction params
```

## Timeline Estimate

| Phase | Duration |
|-------|----------|
| DSP module implementation | 2 days |
| Backend integration | 1 day |
| Flutter integration | 1 day |
| Testing & validation | 1 day |
| **Total** | **5 days** |
