/// Crop classifier provider - handles WebSocket communication with /crop endpoint.
///
/// Uses the EXISTING backend (auto-started by app) - gets dynamic port from backendLauncherProvider.
///
/// Workflow:
/// 1. Connect to ws://localhost:{dynamic_port}/crop
/// 2. Send spectrogram → get crops with auto-categorization
/// 3. User labels uncertain crops
/// 4. Send labels → train Siamese model
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/io.dart';
import '../../../core/services/backend_launcher.dart';
import '../widgets/crop_review_dialog.dart';

/// State for crop classifier
class CropClassifierState {
  final bool isConnected;
  final bool isProcessing;
  final bool isTraining;
  final String? error;
  final List<String> availableModels;
  final String? loadedModel;
  final CropTrainingResult? lastResult;

  const CropClassifierState({
    this.isConnected = false,
    this.isProcessing = false,
    this.isTraining = false,
    this.error,
    this.availableModels = const [],
    this.loadedModel,
    this.lastResult,
  });

  CropClassifierState copyWith({
    bool? isConnected,
    bool? isProcessing,
    bool? isTraining,
    String? error,
    List<String>? availableModels,
    String? loadedModel,
    CropTrainingResult? lastResult,
  }) {
    return CropClassifierState(
      isConnected: isConnected ?? this.isConnected,
      isProcessing: isProcessing ?? this.isProcessing,
      isTraining: isTraining ?? this.isTraining,
      error: error,
      availableModels: availableModels ?? this.availableModels,
      loadedModel: loadedModel ?? this.loadedModel,
      lastResult: lastResult ?? this.lastResult,
    );
  }
}

/// Result of crop classifier training
class CropTrainingResult {
  final int epochsTrained;
  final double bestLoss;
  final int pairsUsed;
  final String? modelPath;

  const CropTrainingResult({
    required this.epochsTrained,
    required this.bestLoss,
    required this.pairsUsed,
    this.modelPath,
  });

  factory CropTrainingResult.fromJson(Map<String, dynamic> json) {
    return CropTrainingResult(
      epochsTrained: json['epochs_trained'] as int? ?? 0,
      bestLoss: (json['best_loss'] as num?)?.toDouble() ?? 0.0,
      pairsUsed: json['pairs_used'] as int? ?? 0,
      modelPath: json['model_path'] as String?,
    );
  }
}

/// Detection result from crop inference
class CropDetection {
  final int x1, y1, x2, y2;
  final double confidence;
  final String label;

  const CropDetection({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.confidence,
    required this.label,
  });

  factory CropDetection.fromJson(Map<String, dynamic> json) {
    return CropDetection(
      x1: json['x1'] as int? ?? 0,
      y1: json['y1'] as int? ?? 0,
      x2: json['x2'] as int? ?? 0,
      y2: json['y2'] as int? ?? 0,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      label: json['label'] as String? ?? 'unknown',
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// BOOTSTRAP FLOW DATA CLASSES
// ─────────────────────────────────────────────────────────────────

/// A candidate signal found by template matching
class BootstrapCandidate {
  final int index;
  final Uint8List imageBytes;
  final Map<String, int> box;
  final double score;

  const BootstrapCandidate({
    required this.index,
    required this.imageBytes,
    required this.box,
    required this.score,
  });

  factory BootstrapCandidate.fromJson(Map<String, dynamic> json) {
    final imageBase64 = json['image'] as String? ?? '';
    final imageBytes = imageBase64.isNotEmpty
        ? Uint8List.fromList(base64Decode(imageBase64))
        : Uint8List(0);

    final boxJson = json['box'] as Map<String, dynamic>? ?? {};
    final box = <String, int>{
      'x_min': (boxJson['x_min'] as num?)?.toInt() ?? 0,
      'y_min': (boxJson['y_min'] as num?)?.toInt() ?? 0,
      'x_max': (boxJson['x_max'] as num?)?.toInt() ?? 0,
      'y_max': (boxJson['y_max'] as num?)?.toInt() ?? 0,
    };

    return BootstrapCandidate(
      index: json['index'] as int? ?? 0,
      imageBytes: imageBytes,
      box: box,
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
    );
  }

  /// Convert to CropReviewData for swipe UI
  CropReviewData toCropReviewData() {
    return CropReviewData(
      id: 'bootstrap_$index',
      imageBytes: imageBytes,
      modelConfidence: score, // Use template match score as confidence
      metadata: box.map((k, v) => MapEntry(k, v)),
    );
  }
}

/// Statistics from bootstrapper
class BootstrapStats {
  final int positives;
  final int negatives;
  final int total;
  final bool readyToTrain;

  const BootstrapStats({
    this.positives = 0,
    this.negatives = 0,
    this.total = 0,
    this.readyToTrain = false,
  });

  factory BootstrapStats.fromJson(Map<String, dynamic> json) {
    return BootstrapStats(
      positives: json['positives'] as int? ?? 0,
      negatives: json['negatives'] as int? ?? 0,
      total: json['total'] as int? ?? 0,
      readyToTrain: json['ready_to_train'] as bool? ?? false,
    );
  }
}

/// Result from bootstrap() call
class BootstrapResult {
  final List<BootstrapCandidate> candidates;
  final int seedCount;
  final BootstrapStats stats;

  const BootstrapResult({
    required this.candidates,
    required this.seedCount,
    required this.stats,
  });

  /// Convert candidates to CropReviewData for existing swipe UI
  List<CropReviewData> toCropReviewDataList() {
    return candidates.map((c) => c.toCropReviewData()).toList();
  }
}

/// Crop classifier provider
class CropClassifierNotifier extends StateNotifier<CropClassifierState> {
  final Ref _ref;
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  Completer<Map<String, dynamic>>? _responseCompleter;

  CropClassifierNotifier(this._ref) : super(const CropClassifierState());

  /// Get WebSocket port from the already-running backend
  int _getWsPort() {
    final backendState = _ref.read(backendLauncherProvider);
    return backendState.wsPort ?? 8765; // Fallback if not yet discovered
  }

  /// Connect to crop classifier backend
  Future<bool> connect({int? port}) async {
    final wsPort = port ?? _getWsPort();
    if (state.isConnected) return true;

    try {
      // Connect to crop classifier endpoint using dynamic port
      final uri = Uri.parse('ws://127.0.0.1:$wsPort/crop');

      debugPrint('[CropClassifier] Connecting to $uri');
      _channel = IOWebSocketChannel.connect(uri);

      _subscription = _channel!.stream.listen(
        _onMessage,
        onError: _onError,
        onDone: _onDone,
      );

      state = state.copyWith(isConnected: true, error: null);
      debugPrint('[CropClassifier] Connected');

      // Get initial status
      await getStatus();

      return true;
    } catch (e) {
      debugPrint('[CropClassifier] Connection failed: $e');
      state = state.copyWith(error: 'Connection failed: $e');
      return false;
    }
  }

  /// Disconnect from backend
  void disconnect() {
    _subscription?.cancel();
    _channel?.sink.close();
    _channel = null;
    _subscription = null;
    state = state.copyWith(isConnected: false);
    debugPrint('[CropClassifier] Disconnected');
  }

  /// Get status from backend
  Future<void> getStatus() async {
    final response = await _sendCommand({'command': 'crop_status'});
    if (response != null && response['status'] == 'success') {
      state = state.copyWith(
        availableModels:
            List<String>.from(response['available_models'] as List? ?? []),
        loadedModel: response['loaded_model'] as String?,
      );
    }
  }

  /// Load available models from local models/heads directory
  /// Call this on screen init to populate dropdown even without backend connection
  Future<void> loadAvailableModelsFromDisk() async {
    try {
      final headsDir = Directory('models/heads');
      if (!await headsDir.exists()) {
        // Try alternate path
        final altDir = Directory('g20_demo/models/heads');
        if (await altDir.exists()) {
          await _scanHeadsDir(altDir);
          return;
        }
        debugPrint('[CropClassifier] No models/heads directory found');
        return;
      }
      await _scanHeadsDir(headsDir);
    } catch (e) {
      debugPrint('[CropClassifier] Error scanning heads: $e');
    }
  }

  Future<void> _scanHeadsDir(Directory dir) async {
    final models = <String>[];
    await for (final entity in dir.list()) {
      if (entity is Directory) {
        final name = entity.path.split('/').last.split('\\').last;
        models.add(name);
      }
    }
    models.sort();
    state = state.copyWith(availableModels: models);
    debugPrint('[CropClassifier] Found models: $models');
  }

  /// Detect crops from RFCAP file for specified duration
  ///
  /// [referenceBoxes] - User's drawn boxes used as templates for matching.
  /// Only blobs similar to these will be returned.
  /// Each box should have: {x_min, y_min, x_max, y_max}
  ///
  /// [similarityThreshold] - 0.0 to 1.0, higher = stricter matching
  ///
  /// Returns list of [CropReviewData] for the review dialog
  /// [progressCallback] receives progress updates (0.0 - 1.0)
  Future<List<CropReviewData>> detectCropsFromFile({
    required String rfcapPath,
    required double scanDurationSec,
    double chunkSec = 0.5,
    List<Map<String, int>>? referenceBoxes,
    double similarityThreshold = 0.3,
    void Function(double progress, int cropsFound)? progressCallback,
  }) async {
    state = state.copyWith(isProcessing: true, error: null);

    // Use a FRESH WebSocket connection for this operation
    // (Dart streams are single-subscription, can't reuse)
    WebSocketChannel? cropChannel;
    StreamSubscription? cropSubscription;

    try {
      final crops = <CropReviewData>[];
      final resultCompleter = Completer<List<CropReviewData>>();

      // Message handler for this connection
      void handleMessage(dynamic message) {
        try {
          final data = jsonDecode(message as String) as Map<String, dynamic>;
          final cmd = data['command'] as String?;

          if (cmd == 'crop_detect_progress') {
            final progress = (data['progress'] as num?)?.toDouble() ?? 0.0;
            final cropsFound = data['crops_found'] as int? ?? 0;
            progressCallback?.call(progress, cropsFound);
          } else if (cmd == 'crop_detect_file') {
            // Final result
            final cropsJson = data['crops'] as List? ?? [];

            for (final crop in cropsJson) {
              final imageBase64 = crop['image'] as String?;
              if (imageBase64 == null) continue;

              final imageBytes = base64Decode(imageBase64);
              crops.add(CropReviewData(
                id: crop['id'] as String,
                imageBytes: Uint8List.fromList(imageBytes),
                modelConfidence: (crop['confidence'] as num?)?.toDouble(),
                metadata: {
                  'time_sec': crop['time_sec'],
                  'chunk_idx': crop['chunk_idx'],
                  ...crop['bbox'] as Map<String, dynamic>? ?? {},
                },
              ));
            }

            if (!resultCompleter.isCompleted) {
              resultCompleter.complete(crops);
            }
          } else if (data['status'] == 'error') {
            if (!resultCompleter.isCompleted) {
              resultCompleter.completeError(Exception(data['error']));
            }
          }
        } catch (e) {
          debugPrint('[CropClassifier] Parse error: $e');
        }
      }

      // Create fresh connection using EXISTING backend's dynamic port
      final wsPort = _getWsPort();
      final uri = Uri.parse('ws://127.0.0.1:$wsPort/crop');
      debugPrint('[CropClassifier] Creating fresh connection to $uri for crop detection');
      cropChannel = IOWebSocketChannel.connect(uri);

      cropSubscription = cropChannel.stream.listen(
        handleMessage,
        onError: (e) {
          if (!resultCompleter.isCompleted) {
            resultCompleter.completeError(e);
          }
        },
        onDone: () {
          if (!resultCompleter.isCompleted) {
            resultCompleter.completeError(Exception('Connection closed unexpectedly'));
          }
        },
      );

      // Send command
      final cmd = {
        'command': 'crop_detect_file',
        'rfcap_path': rfcapPath,
        'scan_duration_sec': scanDurationSec,
        'chunk_sec': chunkSec,
        if (referenceBoxes != null && referenceBoxes.isNotEmpty)
          'reference_boxes': referenceBoxes,
        'similarity_threshold': similarityThreshold,
      };
      debugPrint('[CropClassifier] → crop_detect_file (${referenceBoxes?.length ?? 0} reference boxes)');
      cropChannel.sink.add(jsonEncode(cmd));

      // Wait for result with generous timeout
      final result = await resultCompleter.future.timeout(
        Duration(seconds: (scanDurationSec * 2 + 30).toInt()),
        onTimeout: () => throw TimeoutException('Crop detection timed out'),
      );

      state = state.copyWith(isProcessing: false);
      return result;
    } catch (e) {
      state = state.copyWith(isProcessing: false, error: e.toString());
      rethrow;
    } finally {
      // Always clean up the dedicated connection
      cropSubscription?.cancel();
      cropChannel?.sink.close();
    }
  }

  /// Detect crops in spectrogram (base64 PNG)
  ///
  /// Returns list of [CropReviewData] for the review dialog
  Future<List<CropReviewData>> detectCrops(String spectrogramBase64) async {
    state = state.copyWith(isProcessing: true, error: null);

    try {
      final response = await _sendCommand({
        'command': 'crop_detect',
        'spectrogram': spectrogramBase64,
      });

      if (response == null || response['status'] != 'success') {
        throw Exception(response?['error'] ?? 'Detection failed');
      }

      final crops = <CropReviewData>[];
      final cropsJson = response['crops'] as List? ?? [];

      for (final crop in cropsJson) {
        // Decode base64 image to bytes
        final imageBase64 = crop['image'] as String;
        final imageBytes = base64Decode(imageBase64);

        crops.add(CropReviewData(
          id: crop['id'] as String,
          imageBytes: Uint8List.fromList(imageBytes),
          modelConfidence: (crop['confidence'] as num?)?.toDouble(),
          metadata: {
            'x': crop['x'],
            'y': crop['y'],
            'width': crop['width'],
            'height': crop['height'],
          },
        ));
      }

      state = state.copyWith(isProcessing: false);
      return crops;
    } catch (e) {
      state = state.copyWith(isProcessing: false, error: e.toString());
      rethrow;
    }
  }

  /// Train model with labeled crops
  ///
  /// [labels] maps crop ID → true (signal) / false (not signal)
  Future<CropTrainingResult> train({
    required String modelName,
    required Map<String, bool> labels,
    required bool isNew,
  }) async {
    state = state.copyWith(isTraining: true, error: null);

    try {
      final response = await _sendCommand({
        'command': 'crop_train',
        'signal_name': modelName,
        'labels': labels,
        'is_new': isNew,
      });

      if (response == null || response['status'] != 'success') {
        throw Exception(response?['error'] ?? 'Training failed');
      }

      final result =
          CropTrainingResult.fromJson(response['result'] as Map<String, dynamic>);
      state = state.copyWith(
        isTraining: false,
        lastResult: result,
        loadedModel: modelName,
      );

      // Refresh status to get updated model list
      await getStatus();

      return result;
    } catch (e) {
      state = state.copyWith(isTraining: false, error: e.toString());
      rethrow;
    }
  }

  /// Load an existing model
  Future<bool> loadModel(String modelName) async {
    try {
      final response = await _sendCommand({
        'command': 'crop_load_model',
        'signal_name': modelName,
      });

      if (response?['status'] == 'success') {
        state = state.copyWith(loadedModel: modelName);
        return true;
      }
      return false;
    } catch (e) {
      state = state.copyWith(error: e.toString());
      return false;
    }
  }

  /// Run inference on spectrogram
  Future<List<CropDetection>> infer(String spectrogramBase64) async {
    state = state.copyWith(isProcessing: true, error: null);

    try {
      final response = await _sendCommand({
        'command': 'crop_infer',
        'spectrogram': spectrogramBase64,
      });

      if (response == null || response['status'] != 'success') {
        throw Exception(response?['error'] ?? 'Inference failed');
      }

      final detections = <CropDetection>[];
      final detsJson = response['detections'] as List? ?? [];

      for (final det in detsJson) {
        detections.add(CropDetection.fromJson(det as Map<String, dynamic>));
      }

      state = state.copyWith(isProcessing: false);
      return detections;
    } catch (e) {
      state = state.copyWith(isProcessing: false, error: e.toString());
      rethrow;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // BOOTSTRAP FLOW - Seed-based signal expansion
  // ─────────────────────────────────────────────────────────────────

  /// Find candidates similar to user's seed boxes using template matching.
  ///
  /// Flow:
  /// 1. User draws ~20 boxes on their signal → [seedBoxes]
  /// 2. This finds similar-looking candidates using NCC
  /// 3. Returns candidates ranked by similarity for swipe UI
  ///
  /// [spectrogramBase64] - the visible spectrogram as PNG base64
  /// [seedBoxes] - user's drawn boxes [{x_min, y_min, x_max, y_max}, ...]
  /// [topK] - how many candidates to return (default 50)
  ///
  /// Returns list of [BootstrapCandidate] sorted by similarity score
  Future<BootstrapResult> bootstrap({
    required String spectrogramBase64,
    required List<Map<String, int>> seedBoxes,
    int topK = 50,
  }) async {
    state = state.copyWith(isProcessing: true, error: null);

    try {
      final response = await _sendCommand({
        'command': 'bootstrap',
        'spectrogram': spectrogramBase64,
        'seed_boxes': seedBoxes,
        'top_k': topK,
      });

      if (response == null || response['status'] != 'success') {
        throw Exception(response?['error'] ?? 'Bootstrap failed');
      }

      final candidates = <BootstrapCandidate>[];
      final candidatesJson = response['candidates'] as List? ?? [];

      for (final c in candidatesJson) {
        candidates.add(BootstrapCandidate.fromJson(c as Map<String, dynamic>));
      }

      final stats = BootstrapStats.fromJson(
        response['stats'] as Map<String, dynamic>? ?? {},
      );

      state = state.copyWith(isProcessing: false);

      return BootstrapResult(
        candidates: candidates,
        seedCount: response['seed_count'] as int? ?? 0,
        stats: stats,
      );
    } catch (e) {
      state = state.copyWith(isProcessing: false, error: e.toString());
      rethrow;
    }
  }

  /// Bootstrap from RFCAP file directly - EASIER than bootstrap().
  ///
  /// Backend generates spectrogram internally, no screenshot needed.
  ///
  /// [rfcapPath] - Path to RFCAP file
  /// [seedBoxes] - User's drawn boxes in spectrogram pixel coords
  /// [timeStartSec] - Start time of visible window
  /// [timeDurationSec] - Duration of visible window
  /// [topK] - How many candidates to return
  /// Bootstrap from RFCAP file directly - EASIER than bootstrap().
  ///
  /// Backend generates spectrogram internally, no screenshot needed.
  /// Accepts normalized (0-1) or pixel coordinates.
  Future<BootstrapResult> bootstrapFromFile({
    required String rfcapPath,
    required List<Map<String, num>> seedBoxes,
    double timeStartSec = 0.0,
    double timeDurationSec = 0.5,
    int topK = 50,
  }) async {
    state = state.copyWith(isProcessing: true, error: null);

    try {
      final response = await _sendCommand({
        'command': 'bootstrap_file',
        'rfcap_path': rfcapPath,
        'seed_boxes': seedBoxes,
        'time_start_sec': timeStartSec,
        'time_duration_sec': timeDurationSec,
        'top_k': topK,
      });

      if (response == null || response['status'] != 'success') {
        throw Exception(response?['error'] ?? 'Bootstrap failed');
      }

      final candidates = <BootstrapCandidate>[];
      final candidatesJson = response['candidates'] as List? ?? [];

      for (final c in candidatesJson) {
        candidates.add(BootstrapCandidate.fromJson(c as Map<String, dynamic>));
      }

      final stats = BootstrapStats.fromJson(
        response['stats'] as Map<String, dynamic>? ?? {},
      );

      state = state.copyWith(isProcessing: false);

      return BootstrapResult(
        candidates: candidates,
        seedCount: response['seed_count'] as int? ?? 0,
        stats: stats,
      );
    } catch (e) {
      state = state.copyWith(isProcessing: false, error: e.toString());
      rethrow;
    }
  }

  /// Record user's confirmations from swipe UI.
  ///
  /// Call this after user finishes reviewing candidates from [bootstrap].
  /// Seeds are automatically added as positives by the backend.
  ///
  /// [confirmed] - indices that user accepted (swipe right)
  /// [rejected] - indices that user rejected (swipe left)
  ///
  /// Returns updated stats with ready_to_train flag
  Future<BootstrapStats> confirmLabels({
    required List<int> confirmed,
    required List<int> rejected,
  }) async {
    try {
      final response = await _sendCommand({
        'command': 'confirm',
        'confirmed': confirmed,
        'rejected': rejected,
      });

      if (response == null || response['status'] != 'success') {
        throw Exception(response?['error'] ?? 'Confirm failed');
      }

      return BootstrapStats.fromJson(
        response['stats'] as Map<String, dynamic>? ?? {},
      );
    } catch (e) {
      state = state.copyWith(error: e.toString());
      rethrow;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // Private methods
  // ─────────────────────────────────────────────────────────────────

  void _send(Map<String, dynamic> data) {
    if (_channel == null) return;
    final json = jsonEncode(data);
    debugPrint('[CropClassifier] → ${json.substring(0, json.length.clamp(0, 200))}...');
    _channel!.sink.add(json);
  }

  Future<Map<String, dynamic>?> _sendCommand(Map<String, dynamic> data) async {
    if (!state.isConnected) {
      final connected = await connect();
      if (!connected) return null;
    }

    _responseCompleter = Completer<Map<String, dynamic>>();
    _send(data);

    try {
      return await _responseCompleter!.future.timeout(
        const Duration(seconds: 60),
        onTimeout: () => throw TimeoutException('Command timed out'),
      );
    } catch (e) {
      debugPrint('[CropClassifier] Command error: $e');
      return null;
    }
  }

  void _onMessage(dynamic message) {
    try {
      final data = jsonDecode(message as String) as Map<String, dynamic>;
      debugPrint('[CropClassifier] ← ${data['command'] ?? data['type']}');

      // Complete pending command
      if (_responseCompleter != null && !_responseCompleter!.isCompleted) {
        _responseCompleter!.complete(data);
      }
    } catch (e) {
      debugPrint('[CropClassifier] Parse error: $e');
    }
  }

  void _onError(dynamic error) {
    debugPrint('[CropClassifier] Error: $error');
    state = state.copyWith(error: error.toString());
    _responseCompleter?.completeError(error);
  }

  void _onDone() {
    debugPrint('[CropClassifier] Connection closed');
    state = state.copyWith(isConnected: false);
  }

  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}

/// Provider for crop classifier
final cropClassifierProvider =
    StateNotifierProvider<CropClassifierNotifier, CropClassifierState>((ref) {
  return CropClassifierNotifier(ref);
});
