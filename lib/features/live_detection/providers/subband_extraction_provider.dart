import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// State for sub-band extraction
class SubbandExtractionState {
  final bool isExtracting;
  final double progress; // 0-1
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

  SubbandExtractionState copyWith({
    bool? isExtracting,
    double? progress,
    String? currentFile,
    String? error,
    ExtractionResult? lastResult,
  }) {
    return SubbandExtractionState(
      isExtracting: isExtracting ?? this.isExtracting,
      progress: progress ?? this.progress,
      currentFile: currentFile ?? this.currentFile,
      error: error,
      lastResult: lastResult ?? this.lastResult,
    );
  }
}

/// Result from a sub-band extraction operation
class ExtractionResult {
  final String sourceFile;
  final String outputFile;
  final double sourceRateHz;
  final double outputRateHz;
  final double bandwidthHz;
  final double centerOffsetHz;
  final double outputCenterFreqHz;
  final int inputSamples;
  final int outputSamples;
  final double decimationRatio;
  final int filterTaps;
  final double processingTimeSec;

  const ExtractionResult({
    required this.sourceFile,
    required this.outputFile,
    required this.sourceRateHz,
    required this.outputRateHz,
    required this.bandwidthHz,
    required this.centerOffsetHz,
    required this.outputCenterFreqHz,
    required this.inputSamples,
    required this.outputSamples,
    required this.decimationRatio,
    required this.filterTaps,
    required this.processingTimeSec,
  });

  /// Factory to create from WebSocket response JSON
  factory ExtractionResult.fromJson(Map<String, dynamic> json, String sourceFile) {
    return ExtractionResult(
      sourceFile: sourceFile,
      outputFile: json['output_file'] ?? '',
      sourceRateHz: (json['source_rate_hz'] ?? 20e6).toDouble(),
      outputRateHz: (json['output_rate_hz'] ?? 5e6).toDouble(),
      bandwidthHz: (json['bandwidth_hz'] ?? 2e6).toDouble(),
      centerOffsetHz: (json['center_offset_hz'] ?? 0).toDouble(),
      outputCenterFreqHz: (json['output_center_freq_hz'] ?? 825e6).toDouble(),
      inputSamples: json['input_samples'] ?? 0,
      outputSamples: json['output_samples'] ?? 0,
      decimationRatio: (json['decimation_ratio'] ?? 4.0).toDouble(),
      filterTaps: json['filter_taps'] ?? 0,
      processingTimeSec: (json['processing_time_sec'] ?? 0).toDouble(),
    );
  }

  /// Duration of output in seconds
  double get durationSec => outputSamples / outputRateHz;

  /// Size reduction factor
  double get sizeReduction => inputSamples > 0 ? outputSamples / inputSamples : 1.0;

  /// Output file size estimate in MB
  double get outputSizeMB => outputSamples * 8 / 1e6;

  @override
  String toString() {
    return 'ExtractionResult('
        'output=$outputFile, '
        'rate=${(outputRateHz / 1e6).toStringAsFixed(2)} MHz, '
        'bw=${(bandwidthHz / 1e6).toStringAsFixed(2)} MHz, '
        'samples=$outputSamples, '
        'time=${processingTimeSec.toStringAsFixed(2)}s)';
  }
}

/// Parameters for sub-band extraction request
class ExtractionRequest {
  final String sourceFile;
  final String? outputFile;
  final double centerOffsetHz;
  final double bandwidthHz;
  final double? startSec;
  final double? durationSec;
  final double stopbandDb;

  const ExtractionRequest({
    required this.sourceFile,
    this.outputFile,
    required this.centerOffsetHz,
    required this.bandwidthHz,
    this.startSec,
    this.durationSec,
    this.stopbandDb = 60.0,
  });

  /// Convert to JSON for WebSocket command
  Map<String, dynamic> toJson() {
    return {
      'command': 'extract_subband',
      'source_file': sourceFile,
      if (outputFile != null) 'output_file': outputFile,
      'center_offset_hz': centerOffsetHz,
      'bandwidth_hz': bandwidthHz,
      if (startSec != null) 'start_sec': startSec,
      if (durationSec != null) 'duration_sec': durationSec,
      'stopband_db': stopbandDb,
    };
  }

  /// Calculate from normalized box coordinates (0-1) and source bandwidth
  factory ExtractionRequest.fromBox({
    required String sourceFile,
    required double boxX1,
    required double boxX2,
    required double sourceBandwidthHz,
    String? outputFile,
    double? startSec,
    double? durationSec,
    double stopbandDb = 60.0,
  }) {
    // Box center (normalized 0-1, where 0.5 = DC)
    final boxCenterNorm = (boxX1 + boxX2) / 2;

    // Convert to frequency offset from center
    // 0.0 → -BW/2, 0.5 → 0 Hz (DC), 1.0 → +BW/2
    final centerOffsetHz = (boxCenterNorm - 0.5) * sourceBandwidthHz;

    // Box width in Hz
    final boxWidthNorm = (boxX2 - boxX1).abs();
    final signalBandwidthHz = boxWidthNorm * sourceBandwidthHz;

    // Add margin for filter transition (20% extra)
    final extractBandwidthHz = signalBandwidthHz * 1.2;

    // Clamp to reasonable range (min 500 kHz, max source BW)
    final clampedBandwidth = extractBandwidthHz.clamp(500e3, sourceBandwidthHz);

    return ExtractionRequest(
      sourceFile: sourceFile,
      outputFile: outputFile,
      centerOffsetHz: centerOffsetHz,
      bandwidthHz: clampedBandwidth,
      startSec: startSec,
      durationSec: durationSec,
      stopbandDb: stopbandDb,
    );
  }

  @override
  String toString() {
    return 'ExtractionRequest('
        'source=$sourceFile, '
        'offset=${(centerOffsetHz / 1e6).toStringAsFixed(2)} MHz, '
        'bw=${(bandwidthHz / 1e6).toStringAsFixed(2)} MHz)';
  }
}

/// Notifier for sub-band extraction operations
class SubbandExtractionNotifier extends StateNotifier<SubbandExtractionState> {
  SubbandExtractionNotifier() : super(const SubbandExtractionState());

  /// Update progress during extraction (called from WebSocket handler)
  void updateProgress(double progress, String? sourceFile) {
    state = state.copyWith(
      progress: progress,
      currentFile: sourceFile,
    );
  }

  /// Start extraction (call this when sending WebSocket command)
  void startExtraction(String sourceFile) {
    state = SubbandExtractionState(
      isExtracting: true,
      progress: 0.0,
      currentFile: sourceFile,
    );
    debugPrint('[SubbandExtraction] Started: $sourceFile');
  }

  /// Complete extraction with result
  void completeExtraction(ExtractionResult result) {
    state = SubbandExtractionState(
      isExtracting: false,
      progress: 1.0,
      lastResult: result,
    );
    debugPrint('[SubbandExtraction] Complete: $result');
  }

  /// Handle extraction error
  void extractionError(String error) {
    state = state.copyWith(
      isExtracting: false,
      error: error,
    );
    debugPrint('[SubbandExtraction] Error: $error');
  }

  /// Reset state
  void reset() {
    state = const SubbandExtractionState();
  }

  /// Calculate expected output sample rate for given bandwidth
  static double calculateOutputRate(double bandwidthHz) {
    // 2.5x oversampling for filter transition margin
    return bandwidthHz * 2.5;
  }

  /// Calculate expected decimation ratio
  static double calculateDecimationRatio(double sourceRateHz, double bandwidthHz) {
    final targetRate = calculateOutputRate(bandwidthHz);
    return sourceRateHz / targetRate;
  }

  /// Estimate output file size in MB
  static double estimateOutputSizeMB(
    double sourceRateHz,
    double bandwidthHz,
    double durationSec,
  ) {
    final outputRate = calculateOutputRate(bandwidthHz);
    final outputSamples = outputRate * durationSec;
    // 8 bytes per complex sample
    return outputSamples * 8 / 1e6;
  }
}

/// Provider for sub-band extraction state
final subbandExtractionProvider =
    StateNotifierProvider<SubbandExtractionNotifier, SubbandExtractionState>((ref) {
  return SubbandExtractionNotifier();
});

/// Helper provider to get estimated output info for current capture settings
final extractionEstimateProvider = Provider.family<Map<String, dynamic>, ExtractionRequest?>((ref, request) {
  if (request == null) {
    return {
      'output_rate_mhz': 0.0,
      'decimation_ratio': 1.0,
      'output_size_mb': 0.0,
    };
  }

  const sourceRateHz = 20e6; // Standard source rate
  final outputRateHz = SubbandExtractionNotifier.calculateOutputRate(request.bandwidthHz);
  final decimationRatio = SubbandExtractionNotifier.calculateDecimationRatio(sourceRateHz, request.bandwidthHz);

  // Estimate duration if not specified (assume 60 seconds)
  final durationSec = request.durationSec ?? 60.0;
  final outputSizeMB = SubbandExtractionNotifier.estimateOutputSizeMB(
    sourceRateHz,
    request.bandwidthHz,
    durationSec,
  );

  return {
    'output_rate_mhz': outputRateHz / 1e6,
    'decimation_ratio': decimationRatio,
    'output_size_mb': outputSizeMB,
    'bandwidth_mhz': request.bandwidthHz / 1e6,
    'center_offset_mhz': request.centerOffsetHz / 1e6,
  };
});
