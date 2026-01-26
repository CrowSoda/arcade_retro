import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Training progress from backend
class TrainingProgress {
  final String signalName;
  final int epoch;
  final int totalEpochs;
  final double trainLoss;
  final double valLoss;
  final double f1Score;
  final double precision;
  final double recall;
  final bool isBest;
  final double elapsedSec;

  TrainingProgress({
    required this.signalName,
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
      signalName: json['signal_name'] ?? '',
      epoch: json['epoch'] ?? 0,
      totalEpochs: json['total_epochs'] ?? 50,
      trainLoss: (json['train_loss'] ?? 0).toDouble(),
      valLoss: (json['val_loss'] ?? 0).toDouble(),
      f1Score: (json['f1_score'] ?? 0).toDouble(),
      precision: (json['precision'] ?? 0).toDouble(),
      recall: (json['recall'] ?? 0).toDouble(),
      isBest: json['is_best'] ?? false,
      elapsedSec: (json['elapsed_sec'] ?? 0).toDouble(),
    );
  }

  double get progressPercent => totalEpochs > 0 ? epoch / totalEpochs : 0;
}

/// Training result after completion
class TrainingResult {
  final String signalName;
  final int version;
  final int sampleCount;
  final int epochsTrained;
  final bool earlyStopped;
  final Map<String, double> metrics;
  final double trainingTimeSec;
  final int? previousVersion;
  final Map<String, double>? previousMetrics;
  final bool autoPromoted;
  final String? promotionReason;

  TrainingResult({
    required this.signalName,
    required this.version,
    required this.sampleCount,
    required this.epochsTrained,
    required this.earlyStopped,
    required this.metrics,
    required this.trainingTimeSec,
    this.previousVersion,
    this.previousMetrics,
    required this.autoPromoted,
    this.promotionReason,
  });

  factory TrainingResult.fromJson(Map<String, dynamic> json) {
    return TrainingResult(
      signalName: json['signal_name'] ?? '',
      version: json['version'] ?? 1,
      sampleCount: json['sample_count'] ?? 0,
      epochsTrained: json['epochs_trained'] ?? 0,
      earlyStopped: json['early_stopped'] ?? false,
      metrics: _parseMetrics(json['metrics']),
      trainingTimeSec: (json['training_time_sec'] ?? 0).toDouble(),
      previousVersion: json['previous_version'],
      previousMetrics: json['previous_metrics'] != null 
          ? _parseMetrics(json['previous_metrics']) 
          : null,
      autoPromoted: json['auto_promoted'] ?? false,
      promotionReason: json['promotion_reason'],
    );
  }

  static Map<String, double> _parseMetrics(dynamic m) {
    if (m == null) return {};
    final map = <String, double>{};
    if (m is Map) {
      for (final e in m.entries) {
        if (e.value != null) {
          map[e.key.toString()] = (e.value as num).toDouble();
        }
      }
    }
    return map;
  }

  double get f1Score => metrics['f1_score'] ?? 0;
  double get previousF1 => previousMetrics?['f1_score'] ?? 0;
  double get f1Improvement => f1Score - previousF1;
}

/// State for training operations
class TrainingState {
  final bool isTraining;
  final String? currentSignal;
  final TrainingProgress? progress;
  final TrainingResult? lastResult;
  final String? error;
  final bool isConnected;

  const TrainingState({
    this.isTraining = false,
    this.currentSignal,
    this.progress,
    this.lastResult,
    this.error,
    this.isConnected = false,
  });

  TrainingState copyWith({
    bool? isTraining,
    String? currentSignal,
    TrainingProgress? progress,
    TrainingResult? lastResult,
    String? error,
    bool? isConnected,
  }) {
    return TrainingState(
      isTraining: isTraining ?? this.isTraining,
      currentSignal: currentSignal ?? this.currentSignal,
      progress: progress ?? this.progress,
      lastResult: lastResult ?? this.lastResult,
      error: error,
      isConnected: isConnected ?? this.isConnected,
    );
  }
}

/// Provider for training operations
class TrainingNotifier extends StateNotifier<TrainingState> {
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  final String _wsUrl;

  TrainingNotifier({String? wsUrl}) 
      : _wsUrl = wsUrl ?? 'ws://127.0.0.1:8765/training',
        super(const TrainingState());

  Future<void> connect() async {
    if (_channel != null) return;

    try {
      _channel = WebSocketChannel.connect(Uri.parse(_wsUrl));
      state = state.copyWith(isConnected: true, error: null);

      _subscription = _channel!.stream.listen(
        _handleMessage,
        onError: (e) {
          state = state.copyWith(error: e.toString(), isConnected: false);
        },
        onDone: () {
          state = state.copyWith(isConnected: false);
          _channel = null;
        },
      );
    } catch (e) {
      state = state.copyWith(error: e.toString(), isConnected: false);
    }
  }

  void disconnect() {
    _subscription?.cancel();
    _channel?.sink.close();
    _channel = null;
    state = state.copyWith(isConnected: false);
  }

  void _handleMessage(dynamic message) {
    try {
      final data = jsonDecode(message as String) as Map<String, dynamic>;
      final type = data['type'] as String?;

      switch (type) {
        case 'training_progress':
          final progress = TrainingProgress.fromJson(data);
          state = state.copyWith(
            isTraining: true,
            currentSignal: progress.signalName,
            progress: progress,
          );
          break;

        case 'training_complete':
          final result = TrainingResult.fromJson(data);
          state = state.copyWith(
            isTraining: false,
            lastResult: result,
            progress: null,
          );
          break;

        case 'training_failed':
          state = state.copyWith(
            isTraining: false,
            error: data['error'] as String?,
            progress: null,
          );
          break;

        case 'training_cancelled':
          state = state.copyWith(
            isTraining: false,
            progress: null,
          );
          break;

        case 'error':
          state = state.copyWith(error: data['message'] as String?);
          break;
      }
    } catch (e) {
      state = state.copyWith(error: 'Parse error: $e');
    }
  }

  void _send(Map<String, dynamic> data) {
    if (_channel == null) {
      state = state.copyWith(error: 'Not connected');
      return;
    }
    _channel!.sink.add(jsonEncode(data));
  }

  /// Train a signal (new or extend)
  Future<void> trainSignal(String signalName, {String? notes, bool isNew = false}) async {
    if (!state.isConnected) await connect();
    
    state = state.copyWith(
      isTraining: true,
      currentSignal: signalName,
      error: null,
    );

    _send({
      'command': 'train_signal',
      'signal_name': signalName,
      'notes': notes,
      'is_new': isNew,
    });
  }

  /// Cancel current training
  void cancelTraining() {
    _send({'command': 'cancel_training'});
    state = state.copyWith(isTraining: false, progress: null);
  }

  /// Get training status
  void getTrainingStatus() {
    _send({'command': 'get_training_status'});
  }

  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}

/// Global provider
final trainingProvider = StateNotifierProvider<TrainingNotifier, TrainingState>((ref) {
  return TrainingNotifier();
});
