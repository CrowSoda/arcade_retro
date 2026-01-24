// lib/core/grpc/connection_manager.dart
/// gRPC Connection Manager for Tensorcade Backend
/// Manages connection state, auto-reconnect, and health checks.

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:grpc/grpc.dart';

/// Connection state enum
enum ConnectionState {
  disconnected,
  connecting,
  connected,
  error,
}

/// Connection configuration
class ConnectionConfig {
  final String host;
  final int port;
  final Duration timeout;
  final Duration reconnectInterval;
  final int maxReconnectAttempts;
  final bool useTls;

  const ConnectionConfig({
    this.host = 'localhost',
    this.port = 50051,
    this.timeout = const Duration(seconds: 10),
    this.reconnectInterval = const Duration(seconds: 5),
    this.maxReconnectAttempts = 10,
    this.useTls = false,
  });

  ConnectionConfig copyWith({
    String? host,
    int? port,
    Duration? timeout,
    Duration? reconnectInterval,
    int? maxReconnectAttempts,
    bool? useTls,
  }) {
    return ConnectionConfig(
      host: host ?? this.host,
      port: port ?? this.port,
      timeout: timeout ?? this.timeout,
      reconnectInterval: reconnectInterval ?? this.reconnectInterval,
      maxReconnectAttempts: maxReconnectAttempts ?? this.maxReconnectAttempts,
      useTls: useTls ?? this.useTls,
    );
  }
}

/// Connection state model
class ConnectionStateModel {
  final ConnectionState state;
  final String? errorMessage;
  final int reconnectAttempts;
  final DateTime? connectedAt;
  final DateTime? lastHealthCheck;
  final Duration? latency;
  final ConnectionConfig config;

  const ConnectionStateModel({
    this.state = ConnectionState.disconnected,
    this.errorMessage,
    this.reconnectAttempts = 0,
    this.connectedAt,
    this.lastHealthCheck,
    this.latency,
    this.config = const ConnectionConfig(),
  });

  bool get isConnected => state == ConnectionState.connected;
  bool get isConnecting => state == ConnectionState.connecting;
  bool get hasError => state == ConnectionState.error;

  String get statusText {
    switch (state) {
      case ConnectionState.disconnected:
        return 'Disconnected';
      case ConnectionState.connecting:
        return reconnectAttempts > 0
            ? 'Reconnecting (${reconnectAttempts}/${config.maxReconnectAttempts})...'
            : 'Connecting...';
      case ConnectionState.connected:
        return latency != null ? 'Connected (${latency!.inMilliseconds}ms)' : 'Connected';
      case ConnectionState.error:
        return errorMessage ?? 'Connection Error';
    }
  }

  ConnectionStateModel copyWith({
    ConnectionState? state,
    String? errorMessage,
    int? reconnectAttempts,
    DateTime? connectedAt,
    DateTime? lastHealthCheck,
    Duration? latency,
    ConnectionConfig? config,
  }) {
    return ConnectionStateModel(
      state: state ?? this.state,
      errorMessage: errorMessage,
      reconnectAttempts: reconnectAttempts ?? this.reconnectAttempts,
      connectedAt: connectedAt ?? this.connectedAt,
      lastHealthCheck: lastHealthCheck ?? this.lastHealthCheck,
      latency: latency ?? this.latency,
      config: config ?? this.config,
    );
  }
}

/// Connection manager notifier
class ConnectionManagerNotifier extends StateNotifier<ConnectionStateModel> {
  ConnectionManagerNotifier() : super(const ConnectionStateModel());

  ClientChannel? _channel;
  Timer? _reconnectTimer;
  Timer? _healthCheckTimer;
  bool _disposed = false;

  /// Get the gRPC channel
  ClientChannel? get channel => _channel;

  /// Update connection config
  void setConfig(ConnectionConfig config) {
    state = state.copyWith(config: config);
    // Reconnect if config changed while connected
    if (state.isConnected) {
      disconnect();
      connect();
    }
  }

  /// Connect to the gRPC server
  Future<bool> connect() async {
    if (_disposed) return false;
    if (state.isConnected || state.isConnecting) return state.isConnected;

    state = state.copyWith(
      state: ConnectionState.connecting,
      errorMessage: null,
    );

    try {
      // Create channel
      _channel = ClientChannel(
        state.config.host,
        port: state.config.port,
        options: ChannelOptions(
          credentials: state.config.useTls
              ? const ChannelCredentials.secure()
              : const ChannelCredentials.insecure(),
          connectionTimeout: state.config.timeout,
        ),
      );

      // Test connection with a simple call
      final start = DateTime.now();
      // Note: In real implementation, call GetStatus() here
      await Future.delayed(const Duration(milliseconds: 50)); // Simulated
      final latency = DateTime.now().difference(start);

      state = state.copyWith(
        state: ConnectionState.connected,
        connectedAt: DateTime.now(),
        lastHealthCheck: DateTime.now(),
        latency: latency,
        reconnectAttempts: 0,
      );

      debugPrint('📡 Connected to ${state.config.host}:${state.config.port}');

      // Start health check timer
      _startHealthChecks();

      return true;
    } catch (e) {
      final errorMsg = _parseError(e);
      state = state.copyWith(
        state: ConnectionState.error,
        errorMessage: errorMsg,
      );
      debugPrint('❌ Connection failed: $errorMsg');

      // Schedule reconnect
      _scheduleReconnect();
      return false;
    }
  }

  /// Disconnect from the gRPC server
  Future<void> disconnect() async {
    _cancelTimers();

    if (_channel != null) {
      await _channel!.shutdown();
      _channel = null;
    }

    state = state.copyWith(
      state: ConnectionState.disconnected,
      errorMessage: null,
      reconnectAttempts: 0,
    );

    debugPrint('🔌 Disconnected');
  }

  /// Force reconnection
  Future<bool> reconnect() async {
    await disconnect();
    return connect();
  }

  void _startHealthChecks() {
    _healthCheckTimer?.cancel();
    _healthCheckTimer = Timer.periodic(
      const Duration(seconds: 30),
      (_) => _performHealthCheck(),
    );
  }

  Future<void> _performHealthCheck() async {
    if (!state.isConnected || _channel == null) return;

    try {
      final start = DateTime.now();
      // Note: In real implementation, call GetStatus() here
      await Future.delayed(const Duration(milliseconds: 20)); // Simulated
      final latency = DateTime.now().difference(start);

      state = state.copyWith(
        lastHealthCheck: DateTime.now(),
        latency: latency,
      );
    } catch (e) {
      debugPrint('⚠️ Health check failed: $e');
      state = state.copyWith(
        state: ConnectionState.error,
        errorMessage: 'Health check failed',
      );
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    if (_disposed) return;

    final attempts = state.reconnectAttempts + 1;

    if (attempts > state.config.maxReconnectAttempts) {
      state = state.copyWith(
        state: ConnectionState.error,
        errorMessage: 'Max reconnection attempts reached',
        reconnectAttempts: attempts,
      );
      return;
    }

    state = state.copyWith(reconnectAttempts: attempts);

    // Exponential backoff
    final delay = Duration(
      milliseconds: state.config.reconnectInterval.inMilliseconds *
          (1 << (attempts - 1).clamp(0, 5)),
    );

    debugPrint('🔄 Reconnecting in ${delay.inSeconds}s (attempt $attempts)');

    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(delay, () {
      if (!_disposed) connect();
    });
  }

  void _cancelTimers() {
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
    _healthCheckTimer?.cancel();
    _healthCheckTimer = null;
  }

  String _parseError(dynamic error) {
    if (error is GrpcError) {
      return '${error.codeName}: ${error.message}';
    }
    return error.toString();
  }

  @override
  void dispose() {
    _disposed = true;
    _cancelTimers();
    _channel?.shutdown();
    super.dispose();
  }
}

/// Provider for connection manager
final connectionManagerProvider =
    StateNotifierProvider<ConnectionManagerNotifier, ConnectionStateModel>(
  (ref) => ConnectionManagerNotifier(),
);

/// Provider for connection config
final connectionConfigProvider = StateProvider<ConnectionConfig>(
  (ref) => const ConnectionConfig(),
);

/// Computed provider for connection status
final isConnectedProvider = Provider<bool>(
  (ref) => ref.watch(connectionManagerProvider).isConnected,
);
