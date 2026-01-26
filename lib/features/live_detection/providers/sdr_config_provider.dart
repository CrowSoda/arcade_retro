import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import '../../../core/services/rfcap_service.dart';
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

/// Queued capture request
class CaptureRequest {
  final String signalName;
  final String targetFreqMHz;
  final int durationMinutes;
  final double? boxX1;
  final double? boxY1;
  final double? boxX2;
  final double? boxY2;
  final DateTime queuedAt;

  CaptureRequest({
    required this.signalName,
    required this.targetFreqMHz,
    required this.durationMinutes,
    this.boxX1,
    this.boxY1,
    this.boxX2,
    this.boxY2,
    DateTime? queuedAt,
  }) : queuedAt = queuedAt ?? DateTime.now();
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
  ManualCaptureNotifier() : super(const ManualCaptureState());

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

  /// Generate signal name: man_[DTG]_[FREQ]MHz
  String _generateSignalName(String freqMHz) {
    final now = DateTime.now().toUtc();
    final dtg = DateFormat('HHmmss\'Z\'MMMdd').format(now).toUpperCase();
    final freqInt = double.tryParse(freqMHz)?.round() ?? 825;
    return 'man_${dtg}_${freqInt}MHz';
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
    final signalName = _generateSignalName(freqMHz);
    
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
        // Ensure pending is cleared
        pendingBoxX1: null,
        pendingBoxY1: null,
        pendingBoxX2: null,
        pendingBoxY2: null,
      );
      debugPrint('📡 Starting capture: $signalName');
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

  /// Simulates streaming capture: reads chunks from source, writes chunks to disk
  /// Uses G20 RFCAP format with 512-byte binary header for proper file loading
  /// This mimics real SDR behavior where DMA streams to a ring buffer and we write to disk incrementally
  void _simulateCapture() async {
    final totalSeconds = state.captureDurationMinutes * 60;
    final startTime = DateTime.now();
    
    // Capture parameters
    final signalName = state.signalName ?? 'UNKNOWN';
    final centerFreqMHz = double.tryParse(state.targetFreqMHz ?? '825.0') ?? 825.0;
    
    // Calculate bandwidth from box width (if available)
    // Box x1/x2 are normalized 0-1, representing the 20 MHz capture bandwidth
    const sourceBandwidthMHz = 20.0;  // Source file bandwidth
    double bandwidthMHz = sourceBandwidthMHz;  // Default to full bandwidth
    if (state.boxX1 != null && state.boxX2 != null) {
      final boxWidth = (state.boxX2! - state.boxX1!).abs();
      bandwidthMHz = boxWidth * sourceBandwidthMHz;
      if (bandwidthMHz < 0.5) bandwidthMHz = 0.5;  // Minimum 500 kHz
    }
    
    // Source file sample rate - MUST match actual source file!
    const sampleRate = 20e6;  // 20 MHz sample rate (matches 825MHz.sigmf-data)
    
    // Chunk size: 1 second of data = 20M samples * 8 bytes = 160 MB
    // That's too big, use smaller chunks (1/10 second = 16 MB)
    const samplesPerChunk = 2000000;  // 2M samples = 0.1 sec at 20 MHz
    const chunkSizeBytes = samplesPerChunk * 8;  // 16 MB per chunk
    final totalChunks = totalSeconds * 10;  // 10 chunks per second
    final totalExpectedSamples = totalSeconds * 20000000;  // 20M samples/sec
    
    debugPrint('[Capture] Starting streaming capture: ${totalSeconds}s, ${totalChunks} chunks');
    debugPrint('[Capture] Sample rate: ${(sampleRate / 1e6).toStringAsFixed(1)} MHz');
    debugPrint('[Capture] Bandwidth: ${bandwidthMHz.toStringAsFixed(2)} MHz');
    debugPrint('[Capture] Expected samples: $totalExpectedSamples');
    
    // Prepare output file with RFCAP format
    final filename = RfcapService.generateFilename(signalName);
    final currentDir = Directory.current.path;
    var capturesDir = Directory('$currentDir/data/captures');
    if (!await capturesDir.exists()) {
      capturesDir = Directory('$currentDir/g20_demo/data/captures');
      if (!await capturesDir.exists()) {
        await capturesDir.create(recursive: true);
      }
    }
    final filepath = '${capturesDir.path}/$filename';
    
    // Open source file for reading
    final sourceFile = await _openSourceIqFile();
    if (sourceFile == null) {
      debugPrint('[Capture] ERROR: Cannot open source IQ file');
      state = state.copyWith(phase: CapturePhase.error);
      return;
    }
    
    // Open output file for streaming writes
    final outputFile = await File(filepath).open(mode: FileMode.write);
    var samplesWritten = 0;
    var sourceOffset = 40 * 20000000 * 8;  // Start at 40s into source
    
    debugPrint('[Capture] Writing RFCAP to: $filepath');
    
    try {
      // STEP 1: Write G20 RFCAP header (512 bytes) with estimated sample count
      // We'll update the numSamples field at the end with actual count
      final header = RfcapService.createHeader(
        sampleRate: sampleRate,
        centerFreqHz: centerFreqMHz * 1e6,
        bandwidthHz: bandwidthMHz * 1e6,
        numSamples: totalExpectedSamples,  // Initial estimate, updated later
        signalName: signalName,
        latitude: 35.0,   // Demo location
        longitude: -106.0,
        startTime: startTime,
      );
      await outputFile.writeFrom(header);
      debugPrint('[Capture] Wrote 512-byte RFCAP header');
      
      // STEP 2: Stream IQ data chunks after header
      for (int chunk = 0; chunk < totalChunks && state.phase == CapturePhase.capturing; chunk++) {
        // Read chunk from source file
        await sourceFile.setPosition(sourceOffset);
        final chunkData = await sourceFile.read(chunkSizeBytes);
        sourceOffset += chunkSizeBytes;
        
        // Wrap around if we hit end of source file
        final sourceFileSize = await sourceFile.length();
        if (sourceOffset >= sourceFileSize) {
          sourceOffset = 0;
        }
        
        // Write chunk to output (after header)
        await outputFile.writeFrom(chunkData);
        samplesWritten += chunkData.length ~/ 8;  // 8 bytes per complex sample
        
        // Update progress
        final progress = (chunk + 1) / totalChunks;
        state = state.copyWith(captureProgress: progress);
        
        // Wait 0.1 seconds (10 chunks/second = real-time capture)
        await Future.delayed(const Duration(milliseconds: 100));
        
        // Log every second (every 10 chunks)
        if (chunk % 10 == 0) {
          final secondsElapsed = chunk ~/ 10;
          debugPrint('[Capture] Progress: ${(progress * 100).toInt()}% (${secondsElapsed}s, $samplesWritten samples)');
        }
      }
      
      // STEP 3: Update header with actual sample count (seek back to offset 32)
      await outputFile.setPosition(32);  // numSamples field is at offset 32
      final sampleCountBytes = ByteData(8);
      sampleCountBytes.setUint64(0, samplesWritten, Endian.little);
      await outputFile.writeFrom(sampleCountBytes.buffer.asUint8List());
      debugPrint('[Capture] Updated header with actual sample count: $samplesWritten');
      
      // Close files
      await sourceFile.close();
      await outputFile.close();
      
      if (state.phase == CapturePhase.capturing) {
        final fileSizeMB = (512 + samplesWritten * 8) / 1e6;
        debugPrint('[Capture] Complete: $filepath (${fileSizeMB.toStringAsFixed(1)} MB, $samplesWritten samples)');
        
        state = state.copyWith(phase: CapturePhase.complete);
        
        // Process next in queue
        await Future.delayed(const Duration(milliseconds: 500));
        if (state.queue.isNotEmpty) {
          _processNextInQueue();
        } else {
          state = const ManualCaptureState();
        }
      } else {
        // Capture was cancelled - header still valid with partial data
        debugPrint('[Capture] Cancelled after $samplesWritten samples');
      }
      
    } catch (e) {
      debugPrint('[Capture] ERROR: $e');
      await sourceFile.close();
      await outputFile.close();
      state = state.copyWith(phase: CapturePhase.error);
      
      // Try next in queue even on error
      await Future.delayed(const Duration(milliseconds: 500));
      if (state.queue.isNotEmpty) {
        _processNextInQueue();
      }
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
  return ManualCaptureNotifier();
});

/// Provider for connection status
final sdrConnectionStatusProvider = Provider<SDRConnectionStatus>((ref) {
  return ref.watch(sdrConfigProvider).connectionStatus;
});
