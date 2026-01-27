import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// G20 System connection state
enum G20ConnectionState {
  disconnected,
  connecting,
  connected,
  error,
}

/// G20 System status snapshot
class G20SystemStatus {
  final G20ConnectionState connectionState;
  final double? temperatureC;
  final double? cpuUsagePercent;
  final double? gpuUsagePercent;
  final int? activeRxCount;
  final String? errorMessage;
  final DateTime timestamp;

  const G20SystemStatus({
    required this.connectionState,
    this.temperatureC,
    this.cpuUsagePercent,
    this.gpuUsagePercent,
    this.activeRxCount,
    this.errorMessage,
    required this.timestamp,
  });

  G20SystemStatus copyWith({
    G20ConnectionState? connectionState,
    double? temperatureC,
    double? cpuUsagePercent,
    double? gpuUsagePercent,
    int? activeRxCount,
    String? errorMessage,
    DateTime? timestamp,
  }) {
    return G20SystemStatus(
      connectionState: connectionState ?? this.connectionState,
      temperatureC: temperatureC ?? this.temperatureC,
      cpuUsagePercent: cpuUsagePercent ?? this.cpuUsagePercent,
      gpuUsagePercent: gpuUsagePercent ?? this.gpuUsagePercent,
      activeRxCount: activeRxCount ?? this.activeRxCount,
      errorMessage: errorMessage,
      timestamp: timestamp ?? this.timestamp,
    );
  }
}

/// Tuning command result
class TuningResult {
  final bool success;
  final double centerMHz;
  final double bandwidthMHz;
  final String? errorMessage;
  final DateTime timestamp;

  const TuningResult({
    required this.success,
    required this.centerMHz,
    required this.bandwidthMHz,
    this.errorMessage,
    required this.timestamp,
  });
}

/// RX assignment result
class RxAssignmentResult {
  final bool success;
  final int rxIndex;
  final bool assigned;  // true = assigned to manual, false = released
  final String? errorMessage;
  final DateTime timestamp;

  const RxAssignmentResult({
    required this.success,
    required this.rxIndex,
    required this.assigned,
    this.errorMessage,
    required this.timestamp,
  });
}

/// Collection result
class CollectionResult {
  final bool success;
  final String? filename;
  final int? sampleCount;
  final double? durationSec;
  final String? errorMessage;
  final DateTime timestamp;

  const CollectionResult({
    required this.success,
    this.filename,
    this.sampleCount,
    this.durationSec,
    this.errorMessage,
    required this.timestamp,
  });
}

/// G20 API Service - Unified interface for G20 platform communication
///
/// This service provides a clean interface for communicating with the G20 backend
/// running on Nvidia Jetson Orin with Epiq Sidekiq NV100 SDR.
///
/// Currently uses simulated responses for development.
/// TODO: Replace with real gRPC calls when hardware is available.
class G20ApiService {
  // Connection settings
  String _host = 'localhost';
  int _port = 50051;

  // Connection state
  G20ConnectionState _connectionState = G20ConnectionState.disconnected;
  final _connectionStateController = StreamController<G20ConnectionState>.broadcast();

  // Simulation settings
  bool _simulateErrors = false;
  double _simulateErrorRate = 0.0;  // 0.0 to 1.0
  int _minDelayMs = 100;
  int _maxDelayMs = 300;

  // Random for simulation
  final _random = math.Random();

  /// Stream of connection state changes
  Stream<G20ConnectionState> get connectionState => _connectionStateController.stream;

  /// Current connection state
  G20ConnectionState get currentConnectionState => _connectionState;

  /// Configure simulation behavior (for testing)
  void configureSimulation({
    bool? simulateErrors,
    double? errorRate,
    int? minDelayMs,
    int? maxDelayMs,
  }) {
    if (simulateErrors != null) _simulateErrors = simulateErrors;
    if (errorRate != null) _simulateErrorRate = errorRate.clamp(0.0, 1.0);
    if (minDelayMs != null) _minDelayMs = minDelayMs;
    if (maxDelayMs != null) _maxDelayMs = maxDelayMs;
  }

  /// Simulate network delay
  Future<void> _simulateDelay() async {
    final delayMs = _minDelayMs + _random.nextInt(_maxDelayMs - _minDelayMs);
    await Future.delayed(Duration(milliseconds: delayMs));
  }

  /// Check if should simulate error
  bool _shouldSimulateError() {
    return _simulateErrors && _random.nextDouble() < _simulateErrorRate;
  }

  // ============================================================
  // CONNECTION MANAGEMENT
  // ============================================================

  /// Connect to G20 backend
  ///
  /// [host] - Backend host address (default: localhost)
  /// [port] - gRPC port (default: 50051)
  Future<bool> connect({String? host, int? port}) async {
    if (host != null) _host = host;
    if (port != null) _port = port;

    _setConnectionState(G20ConnectionState.connecting);
    debugPrint('[G20API] Connecting to $_host:$_port...');

    // TODO: Replace with real gRPC channel connection
    await _simulateDelay();

    if (_shouldSimulateError()) {
      _setConnectionState(G20ConnectionState.error);
      debugPrint('[G20API] Connection failed (simulated error)');
      return false;
    }

    _setConnectionState(G20ConnectionState.connected);
    debugPrint('[G20API] Connected to $_host:$_port');
    return true;
  }

  /// Disconnect from G20 backend
  void disconnect() {
    // TODO: Replace with real gRPC channel shutdown
    _setConnectionState(G20ConnectionState.disconnected);
    debugPrint('[G20API] Disconnected');
  }

  void _setConnectionState(G20ConnectionState state) {
    _connectionState = state;
    _connectionStateController.add(state);
  }

  // ============================================================
  // TUNING COMMANDS
  // ============================================================

  /// Set SDR tuning parameters
  ///
  /// [centerMHz] - Center frequency in MHz (30-6000 MHz for NV100)
  /// [bwMHz] - Bandwidth in MHz (0.1-50 MHz for NV100)
  Future<TuningResult> setTuning({
    required double centerMHz,
    required double bwMHz,
  }) async {
    debugPrint('[G20API] setTuning(center: $centerMHz MHz, bw: $bwMHz MHz)');

    // TODO: Replace with real gRPC call to DeviceControl.SetFrequency
    await _simulateDelay();

    if (_shouldSimulateError()) {
      return TuningResult(
        success: false,
        centerMHz: centerMHz,
        bandwidthMHz: bwMHz,
        errorMessage: 'Simulated tuning error',
        timestamp: DateTime.now(),
      );
    }

    // Simulated success
    debugPrint('[G20API] Tuning successful: $centerMHz MHz, BW: $bwMHz MHz');
    return TuningResult(
      success: true,
      centerMHz: centerMHz,
      bandwidthMHz: bwMHz,
      timestamp: DateTime.now(),
    );
  }

  /// Set RX gain
  ///
  /// [gainDb] - Gain in dB (0-34 dB for NV100, 0.5 dB steps)
  Future<bool> setGain(double gainDb) async {
    debugPrint('[G20API] setGain($gainDb dB)');

    // TODO: Replace with real gRPC call
    await _simulateDelay();

    if (_shouldSimulateError()) {
      debugPrint('[G20API] setGain failed (simulated error)');
      return false;
    }

    debugPrint('[G20API] Gain set to $gainDb dB');
    return true;
  }

  // ============================================================
  // RX ASSIGNMENT (2-RX System)
  // ============================================================

  /// Assign RX2 to manual mode (viewing)
  ///
  /// In manual mode, RX2 is used for manual viewing while RX1 continues
  /// auto-scanning. On a 2-RX system, this means no active collection.
  Future<RxAssignmentResult> assignRx2ToManual() async {
    debugPrint('[G20API] Assigning RX2 to manual mode...');

    // TODO: Replace with real gRPC call
    await _simulateDelay();

    if (_shouldSimulateError()) {
      return RxAssignmentResult(
        success: false,
        rxIndex: 2,
        assigned: false,
        errorMessage: 'Failed to assign RX2',
        timestamp: DateTime.now(),
      );
    }

    debugPrint('[G20API] RX2 assigned to manual mode');
    return RxAssignmentResult(
      success: true,
      rxIndex: 2,
      assigned: true,
      timestamp: DateTime.now(),
    );
  }

  /// Release RX2 from manual mode (resume auto)
  Future<RxAssignmentResult> releaseRx2() async {
    debugPrint('[G20API] Releasing RX2 from manual mode...');

    // TODO: Replace with real gRPC call
    await _simulateDelay();

    if (_shouldSimulateError()) {
      return RxAssignmentResult(
        success: false,
        rxIndex: 2,
        assigned: true,
        errorMessage: 'Failed to release RX2',
        timestamp: DateTime.now(),
      );
    }

    debugPrint('[G20API] RX2 released - resuming auto scan');
    return RxAssignmentResult(
      success: true,
      rxIndex: 2,
      assigned: false,
      timestamp: DateTime.now(),
    );
  }

  // ============================================================
  // COLLECTION
  // ============================================================

  /// Trigger manual collection
  ///
  /// [signalName] - Name/label for the captured signal
  /// [durationSec] - Capture duration in seconds
  Future<CollectionResult> triggerCapture({
    required String signalName,
    required int durationSec,
  }) async {
    debugPrint('[G20API] triggerCapture(signal: $signalName, duration: ${durationSec}s)');

    // TODO: Replace with real gRPC call to seizerd
    await _simulateDelay();

    if (_shouldSimulateError()) {
      return CollectionResult(
        success: false,
        errorMessage: 'Failed to start capture',
        timestamp: DateTime.now(),
      );
    }

    // Simulated success - generate mock filename
    final dtg = _generateDTG();
    final filename = '${signalName.toUpperCase()}_$dtg.rfcap';

    debugPrint('[G20API] Capture initiated: $filename');
    return CollectionResult(
      success: true,
      filename: filename,
      durationSec: durationSec.toDouble(),
      timestamp: DateTime.now(),
    );
  }

  String _generateDTG([DateTime? time]) {
    final t = time ?? DateTime.now().toUtc();
    const months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
    final hh = t.hour.toString().padLeft(2, '0');
    final mm = t.minute.toString().padLeft(2, '0');
    final ss = t.second.toString().padLeft(2, '0');
    final mon = months[t.month - 1];
    final yy = (t.year % 100).toString().padLeft(2, '0');
    return '$hh$mm${ss}Z$mon$yy';
  }

  // ============================================================
  // STATUS
  // ============================================================

  /// Get current system status
  Future<G20SystemStatus> getStatus() async {
    // TODO: Replace with real gRPC call to streamd telemetry
    await _simulateDelay();

    // Simulated status with slightly varying values
    return G20SystemStatus(
      connectionState: _connectionState,
      temperatureC: 42.0 + _random.nextDouble() * 5,
      cpuUsagePercent: 30.0 + _random.nextDouble() * 20,
      gpuUsagePercent: 50.0 + _random.nextDouble() * 30,
      activeRxCount: 2,
      timestamp: DateTime.now(),
    );
  }

  /// Dispose resources
  void dispose() {
    _connectionStateController.close();
  }
}

// ============================================================
// RIVERPOD PROVIDERS
// ============================================================

/// Global G20 API service instance
final g20ApiServiceProvider = Provider<G20ApiService>((ref) {
  final service = G20ApiService();
  ref.onDispose(() => service.dispose());
  return service;
});

/// Connection state stream provider
final g20ConnectionStateProvider = StreamProvider<G20ConnectionState>((ref) {
  final service = ref.watch(g20ApiServiceProvider);
  return service.connectionState;
});

/// Current connection state (synchronous)
final g20CurrentConnectionStateProvider = Provider<G20ConnectionState>((ref) {
  final asyncState = ref.watch(g20ConnectionStateProvider);
  return asyncState.valueOrNull ?? G20ConnectionState.disconnected;
});
