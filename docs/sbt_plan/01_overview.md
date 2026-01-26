# Sub-Band Tuning Implementation Plan

## Executive Summary

This plan implements proper sub-band extraction for CNN training data. Currently, G20 captures full 20 MHz wideband IQ and stores it directly. For efficient training, we need to extract narrowband sub-bands around detected signals with proper DSP: frequency translation, anti-aliasing filtering, and decimation.

## Current State Analysis

### What Exists

**capture_subband.py** (Python script):
- Basic frequency shift via `np.exp(-2j * np.pi * f0 * t)`
- 101-tap FIR lowpass filter (scipy.signal.firwin)
- scipy.signal.resample_poly for rational resampling
- Reads/writes RFCAP format

**ISSUES:**
1. Filter stopband attenuation not specified (default ~40dB, needs 60-80dB)
2. No DC offset removal for direct-conversion receivers
3. No sample rate margin (needs 2.5-3x oversampling)
4. Filter order (101 taps) may be insufficient for large decimation

**gpu_fft.py** (Python, GPU):
- Batched FFT on GPU for waterfall display
- No sub-band tuning capability
- Used for display, not training data extraction

**sdr_config_provider.dart** (Flutter capture):
- Writes raw wideband IQ to RFCAP files
- No sub-band extraction (captures full 20 MHz)
- Box coordinates available for signal region

### Key Numbers

| Parameter | Current | Recommended |
|-----------|---------|-------------|
| Source sample rate | 20 MHz | 20 MHz |
| Source bandwidth | 20 MHz | 20 MHz |
| Target bandwidth | Full 20 MHz | Signal BW × 2.5 |
| Filter stopband | ~40 dB | 60-80 dB |
| Filter order | 101 taps | Dynamic |
| DC offset handling | None | Required |

## Goals

1. Extract narrowband sub-bands around detected signals
2. Proper DSP: mix → filter → decimate (in that order!)
3. 60-80 dB stopband attenuation for clean training data
4. Support arbitrary center frequencies and bandwidths
5. GPU acceleration for real-time extraction
6. Flutter UI for user-controlled extraction
