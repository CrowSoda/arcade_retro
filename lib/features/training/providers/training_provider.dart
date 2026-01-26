import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/io.dart';
import '../../../core/services/rfcap_service.dart';
import '../../../core/services/backend_launcher.dart';
import '../../../core/database/signal_database.dart' as db;

/// Training state
class TrainingState {
  final bool isConnected;
  final bool isTraining;
  final bool isSavingSamples;
  final int samplesSaved;
  final int totalSamplesToSave;
  final TrainingProgress? progress;
  final TrainingResult? lastResult;
  final String? error;
  
  const TrainingState({
    this.isConnected = false,
    this.isTraining = false,
    this.isSavingSamples = false,
    this.samplesSaved = 0,
    this.totalSamplesToSave = 0,
    this.progress,
    this.lastResult,
    this.error,
  });
  
  TrainingState copyWith({
    bool? isConnected,
    bool? isTraining,
    bool? isSavingSamples,
    int? samplesSaved,
    int? totalSamplesToSave,
    TrainingProgress? progress,
    TrainingResult? lastResult,
    String? error,
  }) {
    return TrainingState(
      isConnected: isConnected ?? this.isConnected,
      isTraining: isTraining ?? this.isTraining,
      isSavingSamples: isSavingSamples ?? this.isSavingSamples,
      samplesSaved: samplesSaved ?? this.samplesSaved,
      totalSamplesToSave: totalSamplesToSave ?? this.totalSamplesToSave,
      progress: progress ?? this.progress,
      lastResult: lastResult ?? this.lastResult,
      error: error,
    );
  }
  
  double get overallProgress {
    if (isSavingSamples && totalSamplesToSave > 0) {
      // Saving phase: 0-30% of total progress
      return 0.3 * (samplesSaved / totalSamplesToSave);
    }
    if (isTraining && progress != null) {
      // Training phase: 30-100% of total progress
      return 0.3 + 0.7 * (progress!.epoch / progress!.totalEpochs);
    }
    return 0.0;
  }
  
  String get statusText {
    if (error != null) return 'Error: $error';
    if (isSavingSamples) return 'Saving samples... ($samplesSaved/$totalSamplesToSave)';
    if (isTraining && progress != null) {
      return 'Training epoch ${progress!.epoch}/${progress!.totalEpochs} - F1: ${progress!.f1Score.toStringAsFixed(3)}';
    }
    if (lastResult != null) return 'Complete! F1: ${lastResult!.f1Score.toStringAsFixed(3)}';
    return 'Ready';
  }
}

/// Training progress from backend
class TrainingProgress {
  final int epoch;
  final int totalEpochs;
  final double trainLoss;
  final double valLoss;
  final double f1Score;
  final double precision;
  final double recall;
  final bool isBest;
  final double elapsedSec;
  
  const TrainingProgress({
    required this.epoch,
    required this.totalEpochs,
    required this.trainLoss,
    required this.valLoss,
    required this.f1Score,
    required this.precision,
    required this.recall,
    required this.isBest,
    required this.elapsedSec,
  });
  
  factory TrainingProgress.fromJson(Map<String, dynamic> json) {
    return TrainingProgress(
      epoch: json['epoch'] ?? 0,
      totalEpochs: json['total_epochs'] ?? 50,
      trainLoss: (json['train_loss'] ?? 0.0).toDouble(),
      valLoss: (json['val_loss'] ?? 0.0).toDouble(),
      f1Score: (json['f1_score'] ?? 0.0).toDouble(),
      precision: (json['precision'] ?? 0.0).toDouble(),
      recall: (json['recall'] ?? 0.0).toDouble(),
      isBest: json['is_best'] ?? false,
      elapsedSec: (json['elapsed_sec'] ?? 0.0).toDouble(),
    );
  }
}

/// Training result from backend
class TrainingResult {
  final String signalName;
  final int version;
  final int sampleCount;
  final int epochsTrained;
  final bool earlyStopped;
  final double f1Score;
  final double precision;
  final double recall;
  final double trainingTimeSec;
  final bool autoPromoted;
  final String? promotionReason;
  
  const TrainingResult({
    required this.signalName,
    required this.version,
    required this.sampleCount,
    required this.epochsTrained,
    required this.earlyStopped,
    required this.f1Score,
    required this.precision,
    required this.recall,
    required this.trainingTimeSec,
    required this.autoPromoted,
    this.promotionReason,
  });
  
  factory TrainingResult.fromJson(Map<String, dynamic> json) {
    final metrics = json['metrics'] as Map<String, dynamic>? ?? {};
    return TrainingResult(
      signalName: json['signal_name'] ?? '',
      version: json['version'] ?? 0,
      sampleCount: json['sample_count'] ?? 0,
      epochsTrained: json['epochs_trained'] ?? 0,
      earlyStopped: json['early_stopped'] ?? false,
      f1Score: (metrics['f1_score'] ?? 0.0).toDouble(),
      precision: (metrics['precision'] ?? 0.0).toDouble(),
      recall: (metrics['recall'] ?? 0.0).toDouble(),
      trainingTimeSec: (json['training_time_sec'] ?? 0.0).toDouble(),
      autoPromoted: json['auto_promoted'] ?? false,
      promotionReason: json['promotion_reason'],
    );
  }
}

/// Training preset options (research-based)
enum TrainingPreset {
  /// Quick validation (~1-2 min)
  /// High LR (0.005), batch 8, patience 2
  fast('fast', 'Fast', '~1-2 min', 15),
  
  /// Production default (~3-5 min)
  /// TFA/CFA standard LR (0.001), batch 4, patience 5
  balanced('balanced', 'Balanced', '~3-5 min', 30),
  
  /// Maximum accuracy (~10-15 min)
  /// Lower LR (0.0005), batch 2, patience 10
  quality('quality', 'Quality', '~10-15 min', 75);
  
  final String value;
  final String label;
  final String description;
  final int expectedEpochs;
  
  const TrainingPreset(this.value, this.label, this.description, this.expectedEpochs);
}

/// Box to send to backend (REAL-WORLD UNITS - seconds and MHz)
/// 
/// CRITICAL: We send real units (not normalized 0-1) so Python can:
/// 1. Extract IQ data centered on each box independently
/// 2. Compute its own spectrogram with locked inference FFT params
/// 3. Convert real units to pixel coords for ITS spectrogram
/// 
/// This fixes the F1=0 bug where Flutter's spectrogram differs from Python's.
class TrainingBox {
  final double timeStartSec;
  final double timeEndSec;
  final double freqStartMHz;
  final double freqEndMHz;
  
  const TrainingBox({
    required this.timeStartSec,
    required this.timeEndSec,
    required this.freqStartMHz,
    required this.freqEndMHz,
  });
  
  Map<String, dynamic> toJson() => {
    'time_start_sec': timeStartSec,
    'time_end_sec': timeEndSec,
    'freq_start_mhz': freqStartMHz,
    'freq_end_mhz': freqEndMHz,
  };
}

/// Training notifier - handles WebSocket communication with backend
class TrainingNotifier extends StateNotifier<TrainingState> {
  final Ref _ref;
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  Completer<void>? _trainCompleter;
  Completer<bool>? _sampleSaveCompleter;  // For waiting on sample_saved response
  bool _isCancelled = false;  // Flag to stop sample saving loop
  
  TrainingNotifier(this._ref) : super(const TrainingState());
  
  /// Get the dynamic WebSocket port from backend_launcher
  int _getWsPort() {
    final backendState = _ref.read(backendLauncherProvider);
    // Use discovered port if available, fallback to 8765
    return backendState.wsPort ?? 8765;
  }
  
  /// Connect to training WebSocket
  Future<bool> connect({String host = '127.0.0.1', int? port}) async {
    // Use dynamic port if not specified
    final usePort = port ?? _getWsPort();
    print('[Training] 🔌 Connecting to port: $usePort');
    
    if (_channel != null) {
      await disconnect();
    }
    
    try {
      final uri = Uri.parse('ws://$host:$usePort/training');
      print('[Training] URI: $uri');
      
      // Use raw WebSocket.connect to get immediate connection errors
      final socket = await WebSocket.connect(uri.toString())
          .timeout(const Duration(seconds: 5));
      
      print('[Training] ✅ Socket connected! readyState=${socket.readyState}');
      
      _channel = IOWebSocketChannel(socket);
      
      _subscription = _channel!.stream.listen(
        _onMessage,
        onError: (error) {
          print('[Training] ❌ WebSocket error: $error');
          state = state.copyWith(isConnected: false, error: error.toString());
        },
        onDone: () {
          print('[Training] 🔴 WebSocket closed (onDone)');
          state = state.copyWith(isConnected: false);
        },
      );
      
      state = state.copyWith(isConnected: true, error: null);
      print('[Training] ✅ Connected! state.isConnected=${state.isConnected}');
      
      // Give the event loop a moment to process
      await Future.delayed(const Duration(milliseconds: 100));
      
      // Verify still connected
      if (!state.isConnected) {
        print('[Training] ⚠️ Connection was immediately closed by server!');
        return false;
      }
      
      return true;
    } on TimeoutException {
      print('[Training] ❌ Connection timeout');
      state = state.copyWith(isConnected: false, error: 'Connection timeout');
      return false;
    } on SocketException catch (e) {
      print('[Training] ❌ Connection refused: $e');
      state = state.copyWith(isConnected: false, error: 'Backend not running');
      return false;
    } catch (e) {
      print('[Training] ❌ Connection failed: $e');
      state = state.copyWith(isConnected: false, error: e.toString());
      return false;
    }
  }
  
  /// Disconnect from WebSocket
  Future<void> disconnect() async {
    await _subscription?.cancel();
    _subscription = null;
    await _channel?.sink.close();
    _channel = null;
    state = state.copyWith(isConnected: false);
  }
  
  /// Handle incoming WebSocket messages
  void _onMessage(dynamic message) {
    try {
      final data = jsonDecode(message as String) as Map<String, dynamic>;
      final type = data['type'] as String?;
      
      debugPrint('[Training] Received: $type');
      
      switch (type) {
        case 'init_error':
          // Full initialization error from backend - display immediately
          final fullError = data['message'] ?? 'Unknown init error';
          final sampleManagerOk = data['sample_manager_ok'] ?? false;
          final trainingServiceOk = data['training_service_ok'] ?? false;
          print('[Training] ⚠️ INIT ERROR from backend:');
          print('========================');
          print(fullError);
          print('========================');
          print('[Training] sample_manager_ok=$sampleManagerOk, training_service_ok=$trainingServiceOk');
          state = state.copyWith(error: fullError);
          break;
          
        case 'sample_saved':
          // Count ALL samples processed for progress bar (even duplicates)
          final isNew = data['is_new'] as bool? ?? true;
          final totalOnDisk = data['total_samples'] as int? ?? state.samplesSaved;
          
          // Always increment to show progress (counter = samples processed, not just new)
          state = state.copyWith(samplesSaved: state.samplesSaved + 1);
          
          if (isNew) {
            print('[Training] ✓ NEW sample: ${data['sample_id']} (on disk: $totalOnDisk)');
          } else {
            print('[Training] ⏭ Duplicate skipped: ${data['sample_id']} (on disk: $totalOnDisk)');
          }
          
          // Complete the wait completer
          if (_sampleSaveCompleter != null && !_sampleSaveCompleter!.isCompleted) {
            _sampleSaveCompleter!.complete(true);
          }
          break;
          
        case 'training_progress':
          final progress = TrainingProgress.fromJson(data);
          state = state.copyWith(progress: progress, isTraining: true, isSavingSamples: false);
          break;
          
        case 'training_complete':
          final result = TrainingResult.fromJson(data);
          state = state.copyWith(
            isTraining: false,
            isSavingSamples: false,
            lastResult: result,
            progress: null,
          );
          
          // Update signal database with training result
          _ref.read(db.signalDatabaseProvider.notifier).addTrainingResult(
            result.signalName,
            db.TrainingResult(
              timestamp: DateTime.now(),
              dataLabels: result.sampleCount,
              f1Score: result.f1Score,
              precision: result.precision,
              recall: result.recall,
              epochs: result.epochsTrained,
            ),
          );
          debugPrint('[Training] Complete! F1=${result.f1Score} - Updated database for ${result.signalName}');
          
          _trainCompleter?.complete();
          _trainCompleter = null;
          break;
          
        case 'training_failed':
          state = state.copyWith(
            isTraining: false,
            isSavingSamples: false,
            error: data['error'] ?? 'Training failed',
          );
          _trainCompleter?.complete();
          _trainCompleter = null;
          break;
          
        case 'training_cancelled':
          state = state.copyWith(isTraining: false, isSavingSamples: false);
          _trainCompleter?.complete();
          _trainCompleter = null;
          break;
          
        case 'error':
          state = state.copyWith(error: data['message']);
          // Complete sample completer with failure
          if (_sampleSaveCompleter != null && !_sampleSaveCompleter!.isCompleted) {
            _sampleSaveCompleter!.complete(false);
          }
          break;
      }
    } catch (e) {
      debugPrint('[Training] Message parse error: $e');
    }
  }
  
  /// Send command to backend
  void _send(Map<String, dynamic> command) {
    if (_channel == null) {
      debugPrint('[Training] Not connected!');
      return;
    }
    _channel!.sink.add(jsonEncode(command));
  }
  
  /// Save a training sample and WAIT for response
  /// 
  /// Returns true if saved, false if error or timeout
  Future<bool> saveSampleAndWait({
    required Uint8List iqData,
    required List<TrainingBox> boxes,
    required String signalName,
    required Map<String, dynamic> metadata,
    Duration timeout = const Duration(seconds: 30),
  }) async {
    if (!state.isConnected) {
      print('[Training] ❌ saveSampleAndWait called but not connected!');
      return false;
    }
    
    // Create completer for this request
    _sampleSaveCompleter = Completer<bool>();
    
    // Send the save command
    final iqB64 = base64Encode(iqData);
    _send({
      'command': 'save_sample',
      'signal_name': signalName,
      'iq_data': iqB64,
      'boxes': boxes.map((b) => b.toJson()).toList(),
      'metadata': metadata,
    });
    
    // Wait for response with timeout
    try {
      final result = await _sampleSaveCompleter!.future.timeout(timeout);
      return result;
    } on TimeoutException {
      print('[Training] ⏰ Timeout waiting for sample save response');
      return false;
    } catch (e) {
      print('[Training] ❌ Error waiting for sample save: $e');
      return false;
    }
  }
  
  /// Save a training sample (fire and forget - no wait)
  /// 
  /// [iqData] - Raw IQ bytes from RFCAP file
  /// [boxes] - Normalized bounding boxes (0-1)
  /// [signalName] - Class name for this signal
  /// [metadata] - Source file info
  /// 
  /// NOTE: Assumes already connected. Call connect() before this.
  void saveSample({
    required Uint8List iqData,
    required List<TrainingBox> boxes,
    required String signalName,
    required Map<String, dynamic> metadata,
  }) {
    if (!state.isConnected) {
      print('[Training] ❌ saveSample called but not connected!');
      return;
    }
    
    // Encode IQ data as base64
    final iqB64 = base64Encode(iqData);
    
    _send({
      'command': 'save_sample',
      'signal_name': signalName,
      'iq_data': iqB64,
      'boxes': boxes.map((b) => b.toJson()).toList(),
      'metadata': metadata,
    });
  }
  
  /// Train a signal (new or extend existing)
  /// 
  /// [preset] - Training preset: fast, balanced (default), or quality
  Future<void> trainSignal({
    required String signalName,
    TrainingPreset preset = TrainingPreset.balanced,
    bool isNew = true,
    String? notes,
  }) async {
    if (!state.isConnected) {
      final connected = await connect();
      if (!connected) throw Exception('Failed to connect to training backend');
    }
    
    state = state.copyWith(
      isTraining: true,
      isSavingSamples: false,
      progress: null,
      error: null,
    );
    
    _trainCompleter = Completer<void>();
    
    debugPrint('[Training] Starting with preset: ${preset.label}');
    
    _send({
      'command': 'train_signal',
      'signal_name': signalName,
      'preset': preset.value,  // Send preset to backend
      'is_new': isNew,
      'notes': notes,
    });
    
    // Wait for training to complete
    await _trainCompleter!.future;
  }
  
  /// Cancel running training
  void cancelTraining() {
    print('[Training] 🛑 cancelTraining called');
    _isCancelled = true;  // Stop the sample saving loop
    
    // Only send if connected
    if (_channel != null && state.isConnected) {
      try {
        _send({'command': 'cancel_training'});
      } catch (e) {
        debugPrint('[Training] Failed to send cancel: $e');
      }
    }
    // Complete any pending sample save completer
    if (_sampleSaveCompleter != null && !_sampleSaveCompleter!.isCompleted) {
      _sampleSaveCompleter!.complete(false);
    }
    // Always reset state
    state = state.copyWith(
      isTraining: false, 
      isSavingSamples: false,
      error: null,
    );
    _trainCompleter?.complete();
    _trainCompleter = null;
  }
  
  /// Full training flow: save samples from RFCAP + labels, then train
  /// 
  /// [rfcapPath] - Path to RFCAP file
  /// [signalName] - Class name for the signal
  /// [boxes] - List of bounding boxes with time/coords
  /// [preset] - Training preset: fast, balanced (default), or quality
  Future<TrainingResult?> trainFromFile({
    required String rfcapPath,
    required String signalName,
    required List<Map<String, dynamic>> boxes,
    TrainingPreset preset = TrainingPreset.balanced,
    RfcapHeader? header,
  }) async {
    print('[Training] 🚀 trainFromFile called: ${boxes.length} boxes, preset=${preset.label}');
    
    // Reset cancellation flag at start
    _isCancelled = false;
    
    if (boxes.isEmpty) {
      state = state.copyWith(error: 'No bounding boxes to train');
      print('[Training] ❌ No boxes to train');
      return null;
    }
    
    // Try to connect to backend
    try {
      if (!state.isConnected) {
        final connected = await connect();
        if (!connected) {
          state = state.copyWith(
            error: 'Backend not running. Start server with: python backend/server.py',
            isSavingSamples: false,
            isTraining: false,
          );
          return null;
        }
      }
    } catch (e) {
      state = state.copyWith(
        error: 'Connection failed: $e',
        isSavingSamples: false,
        isTraining: false,
      );
      return null;
    }
    
    state = state.copyWith(
      isSavingSamples: true,
      samplesSaved: 0,
      totalSamplesToSave: boxes.length,
      error: null,
    );
    
    // Read RFCAP header if not provided
    header ??= await RfcapService.readHeader(rfcapPath);
    if (header == null) {
      state = state.copyWith(isSavingSamples: false, error: 'Failed to read RFCAP header');
      return null;
    }
    
    // For each box, extract the IQ window and send to backend
    // CRITICAL: Python only uses a 0.1s window centered on the box.
    // We should NOT read the entire box duration - just send metadata and let Python read!
    const trainingWindowSec = 0.15;  // Match Python's TRAINING_WINDOW_SEC (0.1s) + small margin
    
    for (int i = 0; i < boxes.length; i++) {
      // Check for cancellation at start of each iteration
      if (_isCancelled) {
        print('[Training] 🛑 Cancelled during sample save loop (before box ${i+1})');
        return null;
      }
      
      final box = boxes[i];
      
      // Get time window for this box
      final timeStartSec = (box['time_start_sec'] ?? 0.0) as double;
      final timeEndSec = (box['time_end_sec'] ?? 0.2) as double;
      
      // Calculate CENTER of box - Python centers its window around this point
      final boxCenterSec = (timeStartSec + timeEndSec) / 2;
      
      // Read ONLY a small window around the center (not the full box duration!)
      // This prevents reading huge amounts of data for large boxes
      final windowStartSec = (boxCenterSec - trainingWindowSec / 2).clamp(0.0, header.durationSec);
      final windowEndSec = (boxCenterSec + trainingWindowSec / 2).clamp(0.0, header.durationSec);
      final windowDurationSec = windowEndSec - windowStartSec;
      
      // Calculate sample offsets for the SMALL window
      final offsetSamples = (windowStartSec * header.sampleRate).toInt();
      final numSamples = (windowDurationSec * header.sampleRate).toInt();
      
      debugPrint('[Training] Box ${i+1}/${boxes.length}: center=${boxCenterSec.toStringAsFixed(2)}s, '
          'reading ${windowDurationSec.toStringAsFixed(3)}s ($numSamples samples, ${(numSamples * 8 / 1024 / 1024).toStringAsFixed(1)} MB)');
      
      // Read IQ data for the SMALL window only
      final iqData = await RfcapService.readIqDataRaw(
        rfcapPath,
        offsetSamples: offsetSamples,
        numSamples: numSamples,
      );
      
      if (iqData == null || iqData.isEmpty) {
        debugPrint('[Training] Warning: No IQ data for box ${i+1}');
        continue;
      }
      
      // Get frequency bounds from box (real units from LabelBox)
      final freqStartMHz = (box['freq_start_mhz'] ?? header.centerFreqMHz - header.bandwidthMHz / 2) as double;
      final freqEndMHz = (box['freq_end_mhz'] ?? header.centerFreqMHz + header.bandwidthMHz / 2) as double;
      
      // Create TrainingBox with REAL UNITS (not normalized)
      // Python will use these to:
      // 1. Extract IQ centered on the box
      // 2. Compute its own spectrogram
      // 3. Convert real units to pixel coords for THAT spectrogram
      final trainingBox = TrainingBox(
        timeStartSec: timeStartSec,
        timeEndSec: timeEndSec,
        freqStartMHz: freqStartMHz,
        freqEndMHz: freqEndMHz,
      );
      
      // Save to backend - WAIT for response before continuing
      // NOTE: We now send rfcap_path so Python can extract IQ itself
      final boxDurationSec = timeEndSec - timeStartSec;
      final success = await saveSampleAndWait(
        iqData: iqData,  // Still send IQ for backwards compat, but Python will re-extract centered
        boxes: [trainingBox],
        signalName: signalName,
        metadata: {
          'source_file': rfcapPath,
          'rfcap_path': rfcapPath,  // Full path for Python to read
          'time_offset_sec': timeStartSec,
          'duration_sec': boxDurationSec,
          'center_freq_hz': header.centerFreqHz,  // Hz for precision
          'center_freq_mhz': header.centerFreqMHz,
          'sample_rate': header.sampleRate,  // Hz
          'sample_rate_mhz': header.sampleRate / 1e6,
          'bandwidth_hz': header.bandwidthHz,
          'bandwidth_mhz': header.bandwidthMHz,
        },
      );
      
      if (!success) {
        print('[Training] ❌ Sample ${i+1} failed to save!');
        // Check if we're still connected
        if (!state.isConnected) {
          print('[Training] ❌ Lost connection during sample save!');
          state = state.copyWith(
            isSavingSamples: false,
            error: 'Lost connection to backend during sample save. Check Python logs.',
          );
          return null;
        }
      } else {
        print('[Training] ✅ Saved sample ${i+1}/${boxes.length}');
      }
    }
    
    debugPrint('[Training] All samples saved, starting training with preset: ${preset.label}');
    
    // Now train with the selected preset
    await trainSignal(signalName: signalName, preset: preset, isNew: true);
    
    return state.lastResult;
  }
  
  /// Reset state
  void reset() {
    state = TrainingState(isConnected: state.isConnected);
  }
  
  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}

/// Provider for training state and actions
final trainingProvider = StateNotifierProvider<TrainingNotifier, TrainingState>((ref) {
  return TrainingNotifier(ref);
});
