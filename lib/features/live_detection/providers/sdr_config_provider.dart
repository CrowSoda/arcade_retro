import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../../../core/services/rfcap_service.dart';
import '../../../core/services/backend_launcher.dart';
import '../../../core/utils/dtg_formatter.dart';

/// SDR Configuration state
/// Based on Sidekiq NV100 specifications
class SDRConfig {
  /// Center frequency in MHz (30 MHz - 6 GHz)
  final double centerFreqMHz;
  
  /// Instantaneous bandwidth in MHz (0.1 - 50 MHz)
  final double bandwidthMHz;
  
  /// Currently loaded config file path
  final String? configFilePath;
  
  /// Config file name for display
  final String? configFileName;
  
  /// RX Gain in dB (0-34 dB in 0.5 dB steps)
  final double gainDb;
  
  /// Sample rate in Msps (up to 61.44)
  final double sampleRateMsps;
  
  /// Connection status
  final SDRConnectionStatus connectionStatus;
  
  /// Temperature in Celsius
  final double? temperatureC;
  
  /// Last error message
  final String? errorMessage;

  const SDRConfig({
    this.centerFreqMHz = 825.0,
    this.bandwidthMHz = 20.0,
    this.configFilePath,
    this.configFileName,
    this.gainDb = 20.0,
    this.sampleRateMsps = 20.0,
    this.connectionStatus = SDRConnectionStatus.disconnected,
    this.temperatureC,
    this.errorMessage,
  });

  SDRConfig copyWith({
    double? centerFreqMHz,
    double? bandwidthMHz,
    String? configFilePath,
    String? configFileName,
    double? gainDb,
    double? sampleRateMsps,
    SDRConnectionStatus? connectionStatus,
    double? temperatureC,
    String? errorMessage,
  }) {
    return SDRConfig(
      centerFreqMHz: centerFreqMHz ?? this.centerFreqMHz,
      bandwidthMHz: bandwidthMHz ?? this.bandwidthMHz,
      configFilePath: configFilePath ?? this.configFilePath,
      configFileName: configFileName ?? this.configFileName,
      gainDb: gainDb ?? this.gainDb,
      sampleRateMsps: sampleRateMsps ?? this.sampleRateMsps,
      connectionStatus: connectionStatus ?? this.connectionStatus,
      temperatureC: temperatureC ?? this.temperatureC,
      errorMessage: errorMessage,
    );
  }

  /// Validation constants for Sidekiq NV100
  static const double minFreqMHz = 30.0;
  static const double maxFreqMHz = 6000.0;
  static const double minBandwidthMHz = 0.1;
  static const double maxBandwidthMHz = 50.0;
  static const double minGainDb = 0.0;
  static const double maxGainDb = 34.0;
  static const double gainStepDb = 0.5;
  static const double maxSampleRateMsps = 61.44;

  /// Validate frequency
  static String? validateFrequency(double freqMHz) {
    if (freqMHz < minFreqMHz) {
      return 'Frequency must be at least $minFreqMHz MHz';
    }
    if (freqMHz > maxFreqMHz) {
      return 'Frequency must be at most ${maxFreqMHz / 1000} GHz';
    }
    return null;
  }

  /// Validate bandwidth
  static String? validateBandwidth(double bwMHz) {
    if (bwMHz < minBandwidthMHz) {
      return 'Bandwidth must be at least $minBandwidthMHz MHz';
    }
    if (bwMHz > maxBandwidthMHz) {
      return 'Bandwidth must be at most $maxBandwidthMHz MHz';
    }
    return null;
  }

  /// Snap bandwidth to Nyquist-friendly value (power of 2 friendly)
  /// Common FFT-friendly bandwidths: 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 50 MHz
  static double snapToNyquist(double bwMHz) {
    const nyquistFriendly = [0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 32.0, 40.0, 50.0];
    
    // Find closest Nyquist-friendly value
    double closest = nyquistFriendly[0];
    double minDiff = (bwMHz - closest).abs();
    
    for (final nq in nyquistFriendly) {
      final diff = (bwMHz - nq).abs();
      if (diff < minDiff) {
        minDiff = diff;
        closest = nq;
      }
    }
    
    return closest;
  }

  /// Validate gain
  static String? validateGain(double gainDb) {
    if (gainDb < minGainDb || gainDb > maxGainDb) {
      return 'Gain must be between $minGainDb and $maxGainDb dB';
    }
    return null;
  }
}

enum SDRConnectionStatus {
  disconnected,
  connecting,
  connected,
  error,
}

/// Queued capture request with optional sub-band extraction parameters
class CaptureRequest {
  final String signalName;
  final String targetFreqMHz;
  final int durationMinutes;
  final double? boxX1;
  final double? boxY1;
  final double? boxX2;
  final double? boxY2;
  final DateTime queuedAt;

  // Sub-band extraction parameters (calculated from box)
  final double? centerOffsetHz;  // Calculated from box center
  final double? bandwidthHz;     // Calculated from box width
  final bool extractSubband;     // Whether to extract narrowband after capture

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
    this.extractSubband = true,  // Default: extract narrowband for training
    DateTime? queuedAt,
  }) : queuedAt = queuedAt ?? DateTime.now();

  /// Has valid extraction parameters
  bool get hasExtractionParams => centerOffsetHz != null && bandwidthHz != null;

  /// Output sample rate if extraction is performed (2.5x bandwidth)
  double? get outputRateHz => bandwidthHz != null ? bandwidthHz! * 2.5 : null;

  /// Decimation ratio if extraction is performed (source / output rate)
  double? get decimationRatio {
    if (bandwidthHz == null) return null;
    const sourceRateHz = 20e6;  // Standard 20 MHz source
    return sourceRateHz / (bandwidthHz! * 2.5);
  }
}

/// Manual capture state for UNK signal labeling
/// Drawing state is SEPARATE from capture state so drawing can happen while capturing
class ManualCaptureState {
  // Capture state (current capture)
  final CapturePhase phase;
  final double? boxX1;  // Current capture box (normalized 0-1)
  final double? boxY1;
  final double? boxX2;
  final double? boxY2;
  final String? signalName;
  final int captureDurationMinutes;
  final double captureProgress;  // 0-1
  final String? targetFreqMHz;
  final List<CaptureRequest> queue;  // Queue of pending captures

  // Drawing state (SEPARATE from capture - can draw while capturing)
  final bool isDrawing;           // Drawing overlay active
  final double? pendingBoxX1;     // Pending box being drawn
  final double? pendingBoxY1;
  final double? pendingBoxX2;
  final double? pendingBoxY2;
  final String? pendingFreqMHz;   // Freq for pending capture
  final int pendingDuration;      // Duration for pending capture
  
  // Sub-band extraction parameters (set from duration dialog)
  final double? pendingCenterOffsetHz;  // Frequency offset from DC
  final double? pendingBandwidthHz;     // Target bandwidth for extraction
  final bool pendingExtractSubband;     // Whether to extract narrowband

  const ManualCaptureState({
    this.phase = CapturePhase.idle,
    this.boxX1,
    this.boxY1,
    this.boxX2,
    this.boxY2,
    this.signalName,
    this.captureDurationMinutes = 5,
    this.captureProgress = 0.0,
    this.targetFreqMHz,
    this.queue = const [],
    // Drawing state
    this.isDrawing = false,
    this.pendingBoxX1,
    this.pendingBoxY1,
    this.pendingBoxX2,
    this.pendingBoxY2,
    this.pendingFreqMHz,
    this.pendingDuration = 1,
    // Extraction state
    this.pendingCenterOffsetHz,
    this.pendingBandwidthHz,
    this.pendingExtractSubband = true,  // Default: extract for training
  });

  ManualCaptureState copyWith({
    CapturePhase? phase,
    double? boxX1,
    double? boxY1,
    double? boxX2,
    double? boxY2,
    String? signalName,
    int? captureDurationMinutes,
    double? captureProgress,
    String? targetFreqMHz,
    List<CaptureRequest>? queue,
    // Drawing state
    bool? isDrawing,
    double? pendingBoxX1,
    double? pendingBoxY1,
    double? pendingBoxX2,
    double? pendingBoxY2,
    String? pendingFreqMHz,
    int? pendingDuration,
    // Extraction state
    double? pendingCenterOffsetHz,
    double? pendingBandwidthHz,
    bool? pendingExtractSubband,
  }) {
    return ManualCaptureState(
      phase: phase ?? this.phase,
      boxX1: boxX1 ?? this.boxX1,
      boxY1: boxY1 ?? this.boxY1,
      boxX2: boxX2 ?? this.boxX2,
      boxY2: boxY2 ?? this.boxY2,
      signalName: signalName ?? this.signalName,
      captureDurationMinutes: captureDurationMinutes ?? this.captureDurationMinutes,
      captureProgress: captureProgress ?? this.captureProgress,
      targetFreqMHz: targetFreqMHz ?? this.targetFreqMHz,
      queue: queue ?? this.queue,
      // Drawing state
      isDrawing: isDrawing ?? this.isDrawing,
      pendingBoxX1: pendingBoxX1 ?? this.pendingBoxX1,
      pendingBoxY1: pendingBoxY1 ?? this.pendingBoxY1,
      pendingBoxX2: pendingBoxX2 ?? this.pendingBoxX2,
      pendingBoxY2: pendingBoxY2 ?? this.pendingBoxY2,
      pendingFreqMHz: pendingFreqMHz ?? this.pendingFreqMHz,
      pendingDuration: pendingDuration ?? this.pendingDuration,
      // Extraction state
      pendingCenterOffsetHz: pendingCenterOffsetHz ?? this.pendingCenterOffsetHz,
      pendingBandwidthHz: pendingBandwidthHz ?? this.pendingBandwidthHz,
      pendingExtractSubband: pendingExtractSubband ?? this.pendingExtractSubband,
    );
  }

  /// Has valid pending box (for drawing overlay)
  bool get hasPendingBox => pendingBoxX1 != null && pendingBoxY1 != null && 
                            pendingBoxX2 != null && pendingBoxY2 != null;

  /// Get the bounding box in frequency/time terms (for current capture)
  bool get hasValidBox => boxX1 != null && boxY1 != null && boxX2 != null && boxY2 != null;
  
  /// Check if currently capturing
  bool get isCapturing => phase == CapturePhase.capturing;
  
  /// Get queue length
  int get queueLength => queue.length;
}

enum CapturePhase {
  idle,           // Not capturing
  promptTune,     // Asking user to confirm tune
  drawing,        // User is drawing bounding box
  confirmBox,     // User confirming bounding box
  naming,         // User naming the signal
  selectDuration, // User selecting capture duration
  capturing,      // RX-2 is capturing
  complete,       // Capture complete
  error,          // Error occurred
}

/// SDR Configuration notifier
class SDRConfigNotifier extends StateNotifier<SDRConfig> {
  SDRConfigNotifier() : super(const SDRConfig()) {
    // Simulate connection on startup
    _simulateConnection();
  }

  void _simulateConnection() async {
    state = state.copyWith(connectionStatus: SDRConnectionStatus.connecting);
    await Future.delayed(const Duration(milliseconds: 500));
    state = state.copyWith(
      connectionStatus: SDRConnectionStatus.connected,
      temperatureC: 42.5,
    );
  }

  /// Set center frequency with validation
  bool setFrequency(double freqMHz) {
    final error = SDRConfig.validateFrequency(freqMHz);
    if (error != null) {
      state = state.copyWith(errorMessage: error);
      return false;
    }
    state = state.copyWith(
      centerFreqMHz: freqMHz,
      errorMessage: null,
    );
    // TODO: Send to backend via gRPC
    return true;
  }

  /// Set bandwidth with validation
  bool setBandwidth(double bwMHz) {
    final error = SDRConfig.validateBandwidth(bwMHz);
    if (error != null) {
      state = state.copyWith(errorMessage: error);
      return false;
    }
    state = state.copyWith(
      bandwidthMHz: bwMHz,
      errorMessage: null,
    );
    // TODO: Send to backend via gRPC
    return true;
  }

  /// Set gain with validation
  bool setGain(double gainDb) {
    final error = SDRConfig.validateGain(gainDb);
    if (error != null) {
      state = state.copyWith(errorMessage: error);
      return false;
    }
    // Round to nearest 0.5 dB step
    final roundedGain = (gainDb * 2).round() / 2;
    state = state.copyWith(
      gainDb: roundedGain,
      errorMessage: null,
    );
    // TODO: Send to backend via gRPC
    return true;
  }

  /// Load config file
  void loadConfigFile(String path, String fileName) {
    state = state.copyWith(
      configFilePath: path,
      configFileName: fileName,
      errorMessage: null,
    );
    // TODO: Parse config and apply settings
  }

  /// Clear error message
  void clearError() {
    state = state.copyWith(errorMessage: null);
  }

  /// Update temperature (called from telemetry)
  void updateTemperature(double tempC) {
    state = state.copyWith(temperatureC: tempC);
  }
}

/// Manual capture state notifier with queue support
/// Drawing is SEPARATE from capturing - can draw while a capture is running
class ManualCaptureNotifier extends StateNotifier<ManualCaptureState> {
  final Ref _ref;
  
  ManualCaptureNotifier(this._ref) : super(const ManualCaptureState());
  
  /// Get WebSocket port from backend launcher
  int _getWsPort() {
    final backendState = _ref.read(backendLauncherProvider);
    return backendState.wsPort ?? 8765;
  }

  /// Add a capture request to the queue
  void _addToQueue(CaptureRequest request) {
    final newQueue = [...state.queue, request];
    state = state.copyWith(queue: newQueue);
    debugPrint('📡 Queued capture #${newQueue.length}: ${request.signalName}');
  }
  
  /// Process the next item in the queue (called after a capture completes)
  void _processNextInQueue() {
    if (state.queue.isEmpty) {
      debugPrint('Queue empty, resetting to idle');
      state = const ManualCaptureState();
      return;
    }
    
    final next = state.queue.first;
    final remainingQueue = state.queue.sublist(1);
    
    debugPrint('📡 Starting queued capture: ${next.signalName} (${remainingQueue.length} remaining)');
    
    // Start the next capture directly
    state = ManualCaptureState(
      phase: CapturePhase.capturing,
      targetFreqMHz: next.targetFreqMHz,
      signalName: next.signalName,
      captureDurationMinutes: next.durationMinutes,
      boxX1: next.boxX1,
      boxY1: next.boxY1,
      boxX2: next.boxX2,
      boxY2: next.boxY2,
      captureProgress: 0.0,
      queue: remainingQueue,
    );
    
    _simulateCapture();
  }

  /// Start drawing mode - ALWAYS works regardless of capturing state
  /// Uses SEPARATE pending box fields so current capture continues
  void startDrawingMode(String targetFreqMHz, {int durationMinutes = 1}) {
    // FIXED: Use explicit new state to ensure pending box is cleared
    // (copyWith with null values doesn't clear - it preserves old values!)
    state = ManualCaptureState(
      // Preserve capture state if a capture is in progress
      phase: state.phase,
      boxX1: state.boxX1,
      boxY1: state.boxY1,
      boxX2: state.boxX2,
      boxY2: state.boxY2,
      signalName: state.signalName,
      captureDurationMinutes: state.captureDurationMinutes,
      captureProgress: state.captureProgress,
      targetFreqMHz: state.targetFreqMHz,
      queue: state.queue,
      // EXPLICITLY SET drawing state (clears any previous pending box)
      isDrawing: true,
      pendingBoxX1: null,
      pendingBoxY1: null,
      pendingBoxX2: null,
      pendingBoxY2: null,
      pendingFreqMHz: targetFreqMHz,
      pendingDuration: durationMinutes,
    );
    debugPrint('[Manual Capture] Drawing mode started @ $targetFreqMHz MHz (capturing: ${state.isCapturing})');
  }

  /// User started drawing box (pending)
  void startDrawing(double x, double y) {
    state = state.copyWith(
      pendingBoxX1: x,
      pendingBoxY1: y,
      pendingBoxX2: x,
      pendingBoxY2: y,
    );
  }

  /// Update pending box while drawing
  void updateDrawing(double x, double y) {
    state = state.copyWith(pendingBoxX2: x, pendingBoxY2: y);
  }

  /// Finish drawing - user can now confirm
  void finishDrawing() {
    // hasPendingBox checks if pending box is valid
    debugPrint('📐 Drawing finished, hasPendingBox: ${state.hasPendingBox}');
  }

  /// Adjust box (clear pending box to redraw)
  void adjustBox() {
    state = state.copyWith(
      pendingBoxX1: null,
      pendingBoxY1: null,
      pendingBoxX2: null,
      pendingBoxY2: null,
    );
  }

  /// Update pending duration (called from duration dialog after drawing)
  void setPendingDuration(int minutes) {
    state = state.copyWith(pendingDuration: minutes);
  }
  
  /// Set extraction parameters (called from duration dialog)
  void setExtractionParams({
    required double centerOffsetHz,
    required double bandwidthHz,
    required bool extractSubband,
  }) {
    state = state.copyWith(
      pendingCenterOffsetHz: centerOffsetHz,
      pendingBandwidthHz: bandwidthHz,
      pendingExtractSubband: extractSubband,
    );
    debugPrint('[Manual Capture] Extraction params: centerOffset=${centerOffsetHz/1e6}MHz, bw=${bandwidthHz/1e6}MHz, extract=$extractSubband');
  }

  /// Generate signal name for manual capture
  /// Returns "MAN" - the filename generator handles the rest (DTG, freq)
  String _generateSignalName() {
    return 'MAN';
  }

  /// Confirm the drawn box and start/queue capture
  /// If already capturing: QUEUE this capture (current continues)
  /// If idle: START this capture immediately
  void confirmAndStart() {
    if (!state.hasPendingBox) {
      debugPrint('No pending box to confirm');
      return;
    }
    
    final freqMHz = state.pendingFreqMHz ?? '825.0';
    final signalName = _generateSignalName();  // Just "MAN" - filename includes DTG and freq
    
    if (state.isCapturing) {
      // QUEUE - current capture continues uninterrupted
      _addToQueue(CaptureRequest(
        signalName: signalName,
        targetFreqMHz: state.pendingFreqMHz ?? '825.0',
        durationMinutes: state.pendingDuration,
        boxX1: state.pendingBoxX1,
        boxY1: state.pendingBoxY1,
        boxX2: state.pendingBoxX2,
        boxY2: state.pendingBoxY2,
      ));
      
      // Clear drawing state completely - use explicit nulls in new state
      state = ManualCaptureState(
        phase: state.phase,
        boxX1: state.boxX1,
        boxY1: state.boxY1,
        boxX2: state.boxX2,
        boxY2: state.boxY2,
        signalName: state.signalName,
        captureDurationMinutes: state.captureDurationMinutes,
        captureProgress: state.captureProgress,
        targetFreqMHz: state.targetFreqMHz,
        queue: state.queue,
        // CLEAR drawing state
        isDrawing: false,
        pendingBoxX1: null,
        pendingBoxY1: null,
        pendingBoxX2: null,
        pendingBoxY2: null,
        pendingFreqMHz: null,
        pendingDuration: 1,
      );
      debugPrint('📡 Queued $signalName (queue: ${state.queueLength})');
    } else {
      // START - nothing running, start immediately
      // CRITICAL: Preserve extraction params!
      state = ManualCaptureState(
        phase: CapturePhase.capturing,
        targetFreqMHz: state.pendingFreqMHz ?? '825.0',
        signalName: signalName,
        captureDurationMinutes: state.pendingDuration,
        boxX1: state.pendingBoxX1,
        boxY1: state.pendingBoxY1,
        boxX2: state.pendingBoxX2,
        boxY2: state.pendingBoxY2,
        captureProgress: 0.0,
        queue: state.queue,
        isDrawing: false,
        // Ensure pending box is cleared
        pendingBoxX1: null,
        pendingBoxY1: null,
        pendingBoxX2: null,
        pendingBoxY2: null,
        // PRESERVE extraction params for _simulateCapture!
        pendingCenterOffsetHz: state.pendingCenterOffsetHz,
        pendingBandwidthHz: state.pendingBandwidthHz,
        pendingExtractSubband: state.pendingExtractSubband,
      );
      debugPrint('📡 Starting capture: $signalName (extractSubband=${state.pendingExtractSubband}, bw=${state.pendingBandwidthHz})');
      _simulateCapture();
    }
  }

  /// Cancel drawing - FULLY reset drawing state including any saved box
  /// This ensures canceling from duration dialog doesn't leave stale box data
  void cancelDrawing() {
    // Use explicit new state to ensure all pending fields are null (not preserved by copyWith)
    state = ManualCaptureState(
      // Preserve capture state if a capture is in progress
      phase: state.phase,
      boxX1: state.boxX1,
      boxY1: state.boxY1,
      boxX2: state.boxX2,
      boxY2: state.boxY2,
      signalName: state.signalName,
      captureDurationMinutes: state.captureDurationMinutes,
      captureProgress: state.captureProgress,
      targetFreqMHz: state.targetFreqMHz,
      queue: state.queue,
      // FULLY CLEAR drawing state
      isDrawing: false,
      pendingBoxX1: null,
      pendingBoxY1: null,
      pendingBoxX2: null,
      pendingBoxY2: null,
      pendingFreqMHz: null,
      pendingDuration: 1,
    );
    debugPrint('📐 Drawing cancelled and cleared');
  }

  /// Cancel current capture and clear queue
  void cancel() {
    state = const ManualCaptureState();
  }

  /// Reset after complete
  void reset() {
    state = const ManualCaptureState();
  }

  /// Simulates streaming capture with PROPER SUBBAND EXTRACTION
  /// Flow:
  /// 1. Capture raw 20 MHz to TEMP file
  /// 2. If subband extraction requested, call Python backend (shift → filter → decimate)
  /// 3. Python saves final file with correct header (sample_rate = bandwidth for complex IQ)
  /// 4. Delete temp file
  void _simulateCapture() async {
    final totalSeconds = state.captureDurationMinutes * 60;
    final startTime = DateTime.now();
    
    // Capture parameters
    final signalName = state.signalName ?? 'UNKNOWN';
    final centerFreqMHz = double.tryParse(state.targetFreqMHz ?? '825.0') ?? 825.0;
    
    // Source parameters (full 20 MHz capture)
    const sourceSampleRate = 20e6;  // 20 MHz sample rate
    const sourceBandwidth = 20e6;   // Full bandwidth
    
    // Check if subband extraction is requested
    final extractSubband = state.pendingExtractSubband && 
                           state.pendingBandwidthHz != null &&
                           state.pendingBandwidthHz! < sourceBandwidth;
    final targetBandwidthHz = state.pendingBandwidthHz ?? sourceBandwidth;
    final centerOffsetHz = state.pendingCenterOffsetHz ?? 0.0;
    
    // DEBUG: Log why extraction might be skipped
    if (!extractSubband) {
      debugPrint('[Capture] ⚠️ SUBBAND EXTRACTION DISABLED because:');
      if (!state.pendingExtractSubband) {
        debugPrint('         - pendingExtractSubband = false (not enabled in dialog)');
      }
      if (state.pendingBandwidthHz == null) {
        debugPrint('         - pendingBandwidthHz = null (no bandwidth set)');
      } else if (state.pendingBandwidthHz! >= sourceBandwidth) {
        debugPrint('         - pendingBandwidthHz (${state.pendingBandwidthHz!/1e6}MHz) >= sourceBandwidth (${sourceBandwidth/1e6}MHz)');
      }
      debugPrint('[Capture] Will save FULL 20MHz raw capture');
    }
    
    // Calculate target center frequency for extracted subband
    final targetCenterHz = centerFreqMHz * 1e6 + centerOffsetHz;
    
    debugPrint('[Capture] Starting capture: ${totalSeconds}s');
    debugPrint('[Capture] Source: ${sourceSampleRate/1e6} Msps, center=${centerFreqMHz} MHz');
    if (extractSubband) {
      debugPrint('[Capture] SUBBAND: bw=${targetBandwidthHz/1e6} MHz, offset=${centerOffsetHz/1e6} MHz, target_center=${targetCenterHz/1e6} MHz');
    }
    
    // Chunk size: 0.1 sec at 20 MHz = 2M samples * 8 bytes = 16 MB
    const samplesPerChunk = 2000000;
    const chunkSizeBytes = samplesPerChunk * 8;
    final totalChunks = totalSeconds * 10;
    final totalExpectedSamples = totalSeconds * 20000000;
    
    // Prepare directories and filenames
    final currentDir = Directory.current.path;
    var capturesDir = Directory('$currentDir/data/captures');
    if (!await capturesDir.exists()) {
      capturesDir = Directory('$currentDir/g20_demo/data/captures');
      if (!await capturesDir.exists()) {
        await capturesDir.create(recursive: true);
      }
    }
    
    // File paths
    final finalFilename = RfcapService.generateFilename(signalName, null, targetCenterHz / 1e6);
    final tempFilename = '_temp_${DateTime.now().millisecondsSinceEpoch}.rfcap';
    final tempFilepath = '${capturesDir.path}/$tempFilename';
    final finalFilepath = '${capturesDir.path}/$finalFilename';
    
    // Open source file for reading
    final sourceFile = await _openSourceIqFile();
    if (sourceFile == null) {
      debugPrint('[Capture] ERROR: Cannot open source IQ file');
      state = state.copyWith(phase: CapturePhase.error);
      return;
    }
    
    // Open temp file for streaming writes (always write raw 20 MHz first)
    final outputFile = await File(tempFilepath).open(mode: FileMode.write);
    var samplesWritten = 0;
    
    // CRITICAL FIX: Calculate source offset from the box TIME coordinate
    // The box X coordinate (0-1) represents position in the VISIBLE time window
    // For now, we use a time offset based on the current time modulo file duration
    // This ensures different captures get different data instead of always reading from 40s
    final sourceFileSize = await sourceFile.length();
    final totalSourceSamples = (sourceFileSize / 8).floor();  // 8 bytes per complex sample
    final totalSourceSeconds = totalSourceSamples / sourceSampleRate;
    
    // Use current timestamp to pick a pseudo-random start point in the source file
    // This prevents always reading the same stale data
    final randomStartSec = (DateTime.now().millisecondsSinceEpoch / 1000) % (totalSourceSeconds - totalSeconds - 1);
    var sourceOffset = (randomStartSec * sourceSampleRate * 8).toInt();
    
    // Align to 8-byte boundary (complex sample size)
    sourceOffset = (sourceOffset ~/ 8) * 8;
    
    debugPrint('[Capture] Source file: ${totalSourceSeconds.toStringAsFixed(1)}s total');
    debugPrint('[Capture] Starting at: ${(sourceOffset / 8 / sourceSampleRate).toStringAsFixed(1)}s into source');
    
    debugPrint('[Capture] Writing raw capture to TEMP: $tempFilepath');
    
    try {
      // STEP 1: Write raw 20 MHz RFCAP header
      final header = RfcapService.createHeader(
        sampleRate: sourceSampleRate,
        centerFreqHz: centerFreqMHz * 1e6,
        bandwidthHz: sourceBandwidth,  // Full 20 MHz
        numSamples: totalExpectedSamples,
        signalName: signalName,
        latitude: 35.0,
        longitude: -106.0,
        startTime: startTime,
      );
      await outputFile.writeFrom(header);
      
      // STEP 2: Stream raw IQ data
      // OPTIMIZATION: Cache source file size - don't call length() every chunk!
      final sourceFileSizeForLoop = sourceFileSize;  // Already calculated above
      
      for (int chunk = 0; chunk < totalChunks && state.phase == CapturePhase.capturing; chunk++) {
        await sourceFile.setPosition(sourceOffset);
        final chunkData = await sourceFile.read(chunkSizeBytes);
        sourceOffset += chunkSizeBytes;
        
        // Wrap around if we exceed source file
        if (sourceOffset >= sourceFileSizeForLoop) {
          sourceOffset = 0;
        }
        
        await outputFile.writeFrom(chunkData);
        samplesWritten += chunkData.length ~/ 8;
        
        final progress = (chunk + 1) / totalChunks;
        state = state.copyWith(captureProgress: extractSubband ? progress * 0.8 : progress);  // Reserve 20% for extraction
        
        // Reduce delay from 100ms to 10ms for faster capture (still allows UI updates)
        await Future.delayed(const Duration(milliseconds: 10));
        
        if (chunk % 100 == 0) {  // Log less frequently
          debugPrint('[Capture] Progress: ${(progress * 100).toInt()}% ($samplesWritten samples)');
        }
      }
      
      // STEP 3: Update header with actual sample count
      await outputFile.setPosition(32);
      final sampleCountBytes = ByteData(8);
      sampleCountBytes.setUint64(0, samplesWritten, Endian.little);
      await outputFile.writeFrom(sampleCountBytes.buffer.asUint8List());
      
      await sourceFile.close();
      await outputFile.close();
      
      if (state.phase != CapturePhase.capturing) {
        debugPrint('[Capture] Cancelled');
        await File(tempFilepath).delete().catchError((_) {});
        return;
      }
      
      // STEP 4: If subband extraction requested, call Python backend
      if (extractSubband) {
        debugPrint('[Capture] Raw capture complete. Calling Python for subband extraction...');
        state = state.copyWith(captureProgress: 0.85);
        
        final extracted = await _callSubbandExtraction(
          tempFilepath, 
          finalFilepath,
          centerFreqMHz * 1e6,  // original center
          sourceSampleRate,     // original sample rate
          targetCenterHz,       // new center
          targetBandwidthHz,    // new bandwidth
        );
        
        if (extracted) {
          // Delete temp file
          await File(tempFilepath).delete().catchError((_) {});
          debugPrint('[Capture] Subband extraction complete: $finalFilepath');
        } else {
          // Extraction failed - rename temp to final as fallback
          debugPrint('[Capture] Subband extraction failed, using raw capture');
          await File(tempFilepath).rename(finalFilepath).catchError((_) {});
        }
      } else {
        // No extraction - rename temp to final
        await File(tempFilepath).rename(finalFilepath);
        debugPrint('[Capture] Raw capture saved: $finalFilepath');
      }
      
      state = state.copyWith(phase: CapturePhase.complete, captureProgress: 1.0);
      
      // Process next in queue
      await Future.delayed(const Duration(milliseconds: 500));
      if (state.queue.isNotEmpty) {
        _processNextInQueue();
      } else {
        state = const ManualCaptureState();
      }
      
    } catch (e) {
      debugPrint('[Capture] ERROR: $e');
      await sourceFile.close();
      await outputFile.close();
      await File(tempFilepath).delete().catchError((_) {});
      state = state.copyWith(phase: CapturePhase.error);
      
      await Future.delayed(const Duration(milliseconds: 500));
      if (state.queue.isNotEmpty) {
        _processNextInQueue();
      }
    }
  }
  
  /// Call Python backend to extract subband (shift → filter → decimate)
  /// Returns true on success, false on failure
  Future<bool> _callSubbandExtraction(
    String inputPath,
    String outputPath,
    double originalCenterHz,
    double originalSampleRate,
    double newCenterHz,
    double newBandwidthHz,
  ) async {
    try {
      // Connect to training WebSocket (which has extract_subband handler)
      final wsPort = _getWsPort();
      final wsUrl = 'ws://127.0.0.1:$wsPort/training';
      
      debugPrint('[Extract] Connecting to $wsUrl');
      final channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      
      // Send extract_subband command
      final command = jsonEncode({
        'command': 'extract_subband',
        'source_file': inputPath,
        'output_file': outputPath,
        'center_offset_hz': newCenterHz - originalCenterHz,  // Offset from original center
        'bandwidth_hz': newBandwidthHz,
      });
      
      debugPrint('[Extract] Sending: $command');
      channel.sink.add(command);
      
      // Wait for response with timeout
      // NOTE: Extraction of 60s @ 20MHz can take several minutes!
      try {
        await for (final message in channel.stream.timeout(const Duration(seconds: 300))) {
          final data = jsonDecode(message as String);
          final type = data['type'];
          
          if (type == 'subband_extracted') {
            debugPrint('[Extract] Success: ${data['output_file']}');
            debugPrint('[Extract] Output rate: ${data['output_rate_hz']/1e6} Msps, BW: ${data['bandwidth_hz']/1e6} MHz');
            await channel.sink.close();
            return true;
          } else if (type == 'error') {
            debugPrint('[Extract] Error: ${data['message']}');
            await channel.sink.close();
            return false;
          } else if (type == 'extraction_progress') {
            final progress = data['progress'] as double;
            state = state.copyWith(captureProgress: 0.8 + progress * 0.2);  // 80-100%
          }
        }
      } catch (e) {
        debugPrint('[Extract] Timeout or error: $e');
        await channel.sink.close();
        return false;
      }
      
      await channel.sink.close();
      return false;
    } catch (e) {
      debugPrint('[Extract] Connection error: $e');
      return false;
    }
  }

  /// Open the source IQ file for reading
  Future<RandomAccessFile?> _openSourceIqFile() async {
    try {
      final currentDir = Directory.current.path;
      var iqPath = '$currentDir/data/825MHz.sigmf-data';
      var file = File(iqPath);
      
      if (!await file.exists()) {
        iqPath = '$currentDir/g20_demo/data/825MHz.sigmf-data';
        file = File(iqPath);
      }
      
      if (!await file.exists()) {
        debugPrint('[Capture] Source IQ file not found');
        return null;
      }
      
      return await file.open(mode: FileMode.read);
    } catch (e) {
      debugPrint('[Capture] Error opening source: $e');
      return null;
    }
  }

}

/// Provider for SDR configuration
final sdrConfigProvider = StateNotifierProvider<SDRConfigNotifier, SDRConfig>((ref) {
  return SDRConfigNotifier();
});

/// Provider for manual capture state
final manualCaptureProvider = StateNotifierProvider<ManualCaptureNotifier, ManualCaptureState>((ref) {
  return ManualCaptureNotifier(ref);
});

/// Provider for connection status
final sdrConnectionStatusProvider = Provider<SDRConnectionStatus>((ref) {
  return ref.watch(sdrConfigProvider).connectionStatus;
});
