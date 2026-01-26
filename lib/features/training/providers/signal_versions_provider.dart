import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Version info for a signal head
class VersionInfo {
  final int version;
  final DateTime createdAt;
  final int sampleCount;
  final int? splitVersion;
  final int? epochsTrained;
  final bool? earlyStopped;
  final double? f1Score;
  final double? precision;
  final double? recall;
  final double? valLoss;
  final double? trainingTimeSec;
  final int? parentVersion;
  final bool isActive;
  final String? notes;
  final String? promotionReason;

  VersionInfo({
    required this.version,
    required this.createdAt,
    required this.sampleCount,
    this.splitVersion,
    this.epochsTrained,
    this.earlyStopped,
    this.f1Score,
    this.precision,
    this.recall,
    this.valLoss,
    this.trainingTimeSec,
    this.parentVersion,
    required this.isActive,
    this.notes,
    this.promotionReason,
  });

  factory VersionInfo.fromJson(Map<String, dynamic> json) {
    final metrics = json['metrics'] as Map<String, dynamic>? ?? {};
    return VersionInfo(
      version: json['version'] ?? 1,
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
      sampleCount: json['sample_count'] ?? 0,
      splitVersion: json['split_version'],
      epochsTrained: json['epochs_trained'],
      earlyStopped: json['early_stopped'],
      f1Score: _toDouble(metrics['f1_score']),
      precision: _toDouble(metrics['precision']),
      recall: _toDouble(metrics['recall']),
      valLoss: _toDouble(metrics['val_loss']),
      trainingTimeSec: _toDouble(json['training_time_sec']),
      parentVersion: json['parent_version'],
      isActive: json['is_active'] ?? false,
      notes: json['notes'],
      promotionReason: json['promotion_reason'],
    );
  }

  static double? _toDouble(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    return null;
  }
}

/// Signal info from registry
class SignalInfo {
  final String name;
  final int activeVersion;
  final int sampleCount;
  final double? f1Score;
  final DateTime? lastTrained;
  final bool isLoaded;
  List<VersionInfo> versions;

  SignalInfo({
    required this.name,
    required this.activeVersion,
    required this.sampleCount,
    this.f1Score,
    this.lastTrained,
    this.isLoaded = false,
    this.versions = const [],
  });

  factory SignalInfo.fromJson(String name, Map<String, dynamic> json) {
    return SignalInfo(
      name: name,
      activeVersion: json['active_head_version'] ?? 1,
      sampleCount: json['sample_count'] ?? 0,
      f1Score: json['f1_score']?.toDouble(),
      lastTrained: DateTime.tryParse(json['last_trained'] ?? ''),
      isLoaded: json['is_loaded'] ?? false,
    );
  }
}

/// State for signal versions
class SignalVersionsState {
  final Map<String, SignalInfo> signals;
  final bool isLoading;
  final String? error;
  final int? backboneVersion;
  final bool isConnected;

  const SignalVersionsState({
    this.signals = const {},
    this.isLoading = false,
    this.error,
    this.backboneVersion,
    this.isConnected = false,
  });

  SignalVersionsState copyWith({
    Map<String, SignalInfo>? signals,
    bool? isLoading,
    String? error,
    int? backboneVersion,
    bool? isConnected,
  }) {
    return SignalVersionsState(
      signals: signals ?? this.signals,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      backboneVersion: backboneVersion ?? this.backboneVersion,
      isConnected: isConnected ?? this.isConnected,
    );
  }
}

/// Provider for signal version management
class SignalVersionsNotifier extends StateNotifier<SignalVersionsState> {
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  final String _wsUrl;
  Completer<void>? _pendingRequest;

  SignalVersionsNotifier({String? wsUrl})
      : _wsUrl = wsUrl ?? 'ws://127.0.0.1:8765/training',
        super(const SignalVersionsState());

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
        case 'registry':
          _handleRegistry(data);
          break;

        case 'version_history':
          _handleVersionHistory(data);
          break;

        case 'version_promoted':
        case 'version_rollback':
          // Refresh registry after promotion/rollback
          loadRegistry();
          break;

        case 'error':
          state = state.copyWith(error: data['message'] as String?);
          break;
      }

      _pendingRequest?.complete();
      _pendingRequest = null;
    } catch (e) {
      state = state.copyWith(error: 'Parse error: $e');
    }
  }

  void _handleRegistry(Map<String, dynamic> data) {
    final signalsJson = data['signals'] as Map<String, dynamic>? ?? {};
    final signals = <String, SignalInfo>{};

    for (final entry in signalsJson.entries) {
      signals[entry.key] = SignalInfo.fromJson(
        entry.key,
        entry.value as Map<String, dynamic>,
      );
    }

    state = state.copyWith(
      signals: signals,
      backboneVersion: data['backbone_version'],
      isLoading: false,
    );
  }

  void _handleVersionHistory(Map<String, dynamic> data) {
    final signalName = data['signal_name'] as String;
    final versionsJson = data['versions'] as List<dynamic>? ?? [];
    final activeVersion = data['active_version'];

    final versions = versionsJson
        .map((v) => VersionInfo.fromJson(v as Map<String, dynamic>))
        .toList();

    // Update signal with version history
    final signals = Map<String, SignalInfo>.from(state.signals);
    if (signals.containsKey(signalName)) {
      signals[signalName] = SignalInfo(
        name: signalName,
        activeVersion: activeVersion ?? signals[signalName]!.activeVersion,
        sampleCount: signals[signalName]!.sampleCount,
        f1Score: signals[signalName]!.f1Score,
        lastTrained: signals[signalName]!.lastTrained,
        isLoaded: signals[signalName]!.isLoaded,
        versions: versions,
      );
    }

    state = state.copyWith(signals: signals, isLoading: false);
  }

  void _send(Map<String, dynamic> data) {
    if (_channel == null) {
      state = state.copyWith(error: 'Not connected');
      return;
    }
    _channel!.sink.add(jsonEncode(data));
  }

  /// Load the signal registry
  Future<void> loadRegistry() async {
    if (!state.isConnected) await connect();
    state = state.copyWith(isLoading: true, error: null);
    _send({'command': 'get_registry'});
  }

  /// Load version history for a signal
  Future<void> loadVersionHistory(String signalName) async {
    if (!state.isConnected) await connect();
    state = state.copyWith(isLoading: true, error: null);
    _send({
      'command': 'get_version_history',
      'signal_name': signalName,
    });
  }

  /// Promote a version
  Future<void> promoteVersion(String signalName, int version) async {
    if (!state.isConnected) await connect();
    _send({
      'command': 'promote_version',
      'signal_name': signalName,
      'version': version,
    });
  }

  /// Rollback to previous version
  Future<void> rollback(String signalName) async {
    if (!state.isConnected) await connect();
    _send({
      'command': 'rollback_signal',
      'signal_name': signalName,
    });
  }

  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}

/// Global provider
final signalVersionsProvider =
    StateNotifierProvider<SignalVersionsNotifier, SignalVersionsState>((ref) {
  return SignalVersionsNotifier();
});

/// Get versions for a specific signal
final signalVersionsFamily = Provider.family<SignalInfo?, String>((ref, name) {
  final state = ref.watch(signalVersionsProvider);
  return state.signals[name];
});
