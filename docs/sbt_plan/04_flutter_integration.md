# Part 4: Flutter Integration

## Current Capture Flow (To Be Modified)

```
User draws box → confirmAndStart() → _simulateCapture() → Writes raw 20 MHz IQ
```

## New Flow: Capture → Extract → Store

```
User draws box 
    │
    ▼
confirmAndStart() 
    │
    ├─ Calculate: freq offset from box X position
    ├─ Calculate: bandwidth from box width
    │
    ▼
_simulateCapture()
    │
    ├─ Writes wideband capture to temp file
    │
    ▼
Backend extract_subband command (via WebSocket)
    │
    ├─ Mix → Filter → Decimate
    ├─ Writes narrowband RFCAP
    │
    ▼
Store in training_data/signals/{name}/samples/
```

## Modifications to sdr_config_provider.dart

### Add Extraction Parameters

```dart
class CaptureRequest {
  final String signalName;
  final String targetFreqMHz;
  final int durationMinutes;
  final double? boxX1;  // Normalized 0-1
  final double? boxY1;
  final double? boxX2;
  final double? boxY2;
  final DateTime queuedAt;
  
  // NEW: Extraction parameters
  final double? centerOffsetHz;  // Calculated from box center
  final double? bandwidthHz;     // Calculated from box width
  final bool extractSubband;     // Whether to extract narrowband
  
  CaptureRequest({
    required this.signalName,
    required this.targetFreqMHz,
    required this.durationMinutes,
    this.boxX1,
    this.boxY1,
    this.boxX2,
    this.boxY2,
    this.centerOffsetHz,
    this.bandwidthHz,
    this.extractSubband = true,  // Default: extract
    DateTime? queuedAt,
  }) : queuedAt = queuedAt ?? DateTime.now();
}
```

### Calculate Extraction Params from Box

```dart
/// Calculate sub-band parameters from drawn box
/// Box X: 0=left edge, 1=right edge (maps to frequency)
/// Box Y: 0=top, 1=bottom (maps to time - not used for extraction)
Map<String, double> _calculateExtractionParams() {
  final sourceBandwidthHz = 20e6;  // Source capture bandwidth
  final sourceCenterHz = double.tryParse(state.targetFreqMHz ?? '825') ?? 825;
  
  if (state.boxX1 == null || state.boxX2 == null) {
    return {
      'center_offset_hz': 0.0,
      'bandwidth_hz': sourceBandwidthHz,
    };
  }
  
  // Box center (normalized 0-1, where 0.5 = DC)
  final boxCenterNorm = (state.boxX1! + state.boxX2!) / 2;
  
  // Convert to frequency offset from center
  // 0.0 → -10 MHz, 0.5 → 0 MHz (DC), 1.0 → +10 MHz
  final centerOffsetHz = (boxCenterNorm - 0.5) * sourceBandwidthHz;
  
  // Box width in Hz
  final boxWidthNorm = (state.boxX2! - state.boxX1!).abs();
  final signalBandwidthHz = boxWidthNorm * sourceBandwidthHz;
  
  // Add margin for filter transition (2.5x signal bandwidth)
  final extractBandwidthHz = signalBandwidthHz * 1.2;  // 20% margin
  
  return {
    'center_offset_hz': centerOffsetHz,
    'bandwidth_hz': extractBandwidthHz.clamp(500e3, sourceBandwidthHz),  // Min 500 kHz
  };
}
```

### Modified _simulateCapture()

```dart
void _simulateCapture() async {
  // ... existing capture code ...
  
  // After capture complete, request sub-band extraction
  if (state.extractSubband) {
    final params = _calculateExtractionParams();
    
    // Send extraction request to backend
    await _requestSubbandExtraction(
      sourceFile: filepath,
      centerOffsetHz: params['center_offset_hz']!,
      bandwidthHz: params['bandwidth_hz']!,
    );
  }
}

Future<void> _requestSubbandExtraction({
  required String sourceFile,
  required double centerOffsetHz,
  required double bandwidthHz,
}) async {
  // This would be sent via WebSocket to backend
  final request = {
    'command': 'extract_subband',
    'source_file': sourceFile,
    'output_file': _generateOutputPath(),
    'center_offset_hz': centerOffsetHz,
    'bandwidth_hz': bandwidthHz,
  };
  
  debugPrint('[Capture] Requesting sub-band extraction: $request');
  // TODO: Send via WebSocket service
}
```

## New Provider: subband_extraction_provider.dart

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// State for sub-band extraction
class SubbandExtractionState {
  final bool isExtracting;
  final double progress;  // 0-1
  final String? currentFile;
  final String? error;
  final ExtractionResult? lastResult;

  const SubbandExtractionState({
    this.isExtracting = false,
    this.progress = 0.0,
    this.currentFile,
    this.error,
    this.lastResult,
  });
}

class ExtractionResult {
  final String sourceFile;
  final String outputFile;
  final double sourceRateHz;
  final double outputRateHz;
  final double bandwidthHz;
  final int outputSamples;
  final Duration processingTime;
  
  const ExtractionResult({
    required this.sourceFile,
    required this.outputFile,
    required this.sourceRateHz,
    required this.outputRateHz,
    required this.bandwidthHz,
    required this.outputSamples,
    required this.processingTime,
  });
  
  double get decimationRatio => sourceRateHz / outputRateHz;
  double get durationSec => outputSamples / outputRateHz;
}

class SubbandExtractionNotifier extends StateNotifier<SubbandExtractionState> {
  SubbandExtractionNotifier() : super(const SubbandExtractionState());
  
  Future<ExtractionResult?> extractSubband({
    required String sourceFile,
    required double centerOffsetHz,
    required double bandwidthHz,
    double? startSec,
    double? durationSec,
    String? outputPath,
  }) async {
    state = SubbandExtractionState(
      isExtracting: true,
      progress: 0.0,
      currentFile: sourceFile,
    );
    
    try {
      // TODO: Send WebSocket command and wait for result
      // For now, placeholder
      await Future.delayed(const Duration(seconds: 2));
      
      final result = ExtractionResult(
        sourceFile: sourceFile,
        outputFile: outputPath ?? 'extracted.rfcap',
        sourceRateHz: 20e6,
        outputRateHz: bandwidthHz * 2.5,
        bandwidthHz: bandwidthHz,
        outputSamples: 1000000,
        processingTime: const Duration(seconds: 2),
      );
      
      state = SubbandExtractionState(
        isExtracting: false,
        progress: 1.0,
        lastResult: result,
      );
      
      return result;
    } catch (e) {
      state = SubbandExtractionState(
        isExtracting: false,
        error: e.toString(),
      );
      return null;
    }
  }
  
  void reset() {
    state = const SubbandExtractionState();
  }
}

final subbandExtractionProvider = 
    StateNotifierProvider<SubbandExtractionNotifier, SubbandExtractionState>((ref) {
  return SubbandExtractionNotifier();
});
```

## UI Changes

### Capture Duration Dialog Enhancement

Add checkbox: "Extract narrowband for training"

```dart
// In duration_dialog.dart
CheckboxListTile(
  title: const Text('Extract narrowband for training'),
  subtitle: Text(
    'Reduces ${(extractParams['bandwidth_hz']! / 1e6).toStringAsFixed(1)} MHz '
    'to ${(extractParams['bandwidth_hz']! * 2.5 / 1e6).toStringAsFixed(2)} MHz sample rate'
  ),
  value: extractSubband,
  onChanged: (v) => setState(() => extractSubband = v ?? true),
),
```

### Extraction Progress Indicator

Show during extraction:

```dart
if (extractionState.isExtracting)
  LinearProgressIndicator(
    value: extractionState.progress,
  ),
  Text('Extracting sub-band: ${(extractionState.progress * 100).toInt()}%'),
```
