import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// RX Channel mode
enum RxMode {
  scanning,   // Auto scanning (inference pipeline)
  manual,     // Manual tune (user commanded)
  idle,       // Not active
  error,      // Error state
}

/// Single RX channel state
class RxChannelState {
  final int rxNumber;       // 1-4
  final RxMode mode;
  final double centerFreqMHz;
  final double bandwidthMHz;
  final bool isConnected;
  final int? countdownSeconds;  // Countdown for manual mode timeout
  final String? errorMessage;

  const RxChannelState({
    required this.rxNumber,
    this.mode = RxMode.idle,
    this.centerFreqMHz = 825.0,
    this.bandwidthMHz = 20.0,
    this.isConnected = false,
    this.countdownSeconds,
    this.errorMessage,
  });

  RxChannelState copyWith({
    int? rxNumber,
    RxMode? mode,
    double? centerFreqMHz,
    double? bandwidthMHz,
    bool? isConnected,
    int? countdownSeconds,
    String? errorMessage,
  }) {
    return RxChannelState(
      rxNumber: rxNumber ?? this.rxNumber,
      mode: mode ?? this.mode,
      centerFreqMHz: centerFreqMHz ?? this.centerFreqMHz,
      bandwidthMHz: bandwidthMHz ?? this.bandwidthMHz,
      isConnected: isConnected ?? this.isConnected,
      countdownSeconds: countdownSeconds,  // Explicitly allow null
      errorMessage: errorMessage,
    );
  }

  /// Display string for mode
  String get modeDisplayString {
    switch (mode) {
      case RxMode.scanning:
        return '🤖 SCAN';
      case RxMode.manual:
        if (countdownSeconds != null) {
          return '✋ ${countdownSeconds}s';
        }
        return '✋ HOLD';
      case RxMode.idle:
        return '⏸️ IDLE';
      case RxMode.error:
        return '❌ ERR';
    }
  }

  /// Color for mode display
  Color get modeColor {
    switch (mode) {
      case RxMode.scanning:
        return const Color(0xFF4CAF50);  // Green
      case RxMode.manual:
        return const Color(0xFFFF9800);  // Orange/Warning
      case RxMode.idle:
        return const Color(0xFF9E9E9E);  // Grey
      case RxMode.error:
        return const Color(0xFFF44336);  // Red
    }
  }
}

/// Multi-RX state - supports up to 4 RX channels
class MultiRxState {
  final List<RxChannelState> channels;

  MultiRxState({List<RxChannelState>? channels})
      : channels = channels ?? const [];

  /// Get RX by number (1-indexed)
  RxChannelState? getRx(int rxNumber) {
    try {
      return channels.firstWhere((ch) => ch.rxNumber == rxNumber);
    } catch (_) {
      return null;
    }
  }

  /// Get all connected RX channels
  List<RxChannelState> get connectedChannels =>
      channels.where((ch) => ch.isConnected).toList();

  /// Get count of connected channels
  int get connectedCount => connectedChannels.length;

  MultiRxState copyWith({List<RxChannelState>? channels}) {
    return MultiRxState(channels: channels ?? this.channels);
  }
}

/// Multi-RX state notifier
class MultiRxNotifier extends StateNotifier<MultiRxState> {
  MultiRxNotifier() : super(MultiRxState()) {
    // Initialize with simulated 2 RX channels (RX1 scanning, RX2 idle)
    _initializeSimulated();
  }

  void _initializeSimulated() {
    state = MultiRxState(channels: [
      // RX1: Always scanning - inference pipeline
      const RxChannelState(
        rxNumber: 1,
        mode: RxMode.scanning,
        centerFreqMHz: 825.0,
        bandwidthMHz: 20.0,
        isConnected: true,
      ),
      // RX2: Initially idle - for manual tuning
      const RxChannelState(
        rxNumber: 2,
        mode: RxMode.idle,
        centerFreqMHz: 825.0,
        bandwidthMHz: 20.0,
        isConnected: true,
      ),
    ]);
  }

  /// Update a specific RX channel
  void updateRx(int rxNumber, RxChannelState Function(RxChannelState) updater) {
    final updatedChannels = state.channels.map((ch) {
      if (ch.rxNumber == rxNumber) {
        return updater(ch);
      }
      return ch;
    }).toList();
    state = state.copyWith(channels: updatedChannels);
  }

  /// Set RX to manual mode
  void setRxManual(int rxNumber, double centerMHz, double bwMHz, int? timeoutSeconds) {
    updateRx(rxNumber, (ch) => ch.copyWith(
      mode: RxMode.manual,
      centerFreqMHz: centerMHz,
      bandwidthMHz: bwMHz,
      countdownSeconds: timeoutSeconds,
    ));
  }

  /// Set RX to scanning mode
  void setRxScanning(int rxNumber, double centerMHz, double bwMHz) {
    updateRx(rxNumber, (ch) => ch.copyWith(
      mode: RxMode.scanning,
      centerFreqMHz: centerMHz,
      bandwidthMHz: bwMHz,
      countdownSeconds: null,
    ));
  }

  /// Set RX to idle
  void setRxIdle(int rxNumber) {
    updateRx(rxNumber, (ch) => ch.copyWith(
      mode: RxMode.idle,
      countdownSeconds: null,
    ));
  }

  /// Update countdown for RX
  void updateCountdown(int rxNumber, int seconds) {
    updateRx(rxNumber, (ch) => ch.copyWith(countdownSeconds: seconds));
  }

  /// Tune RX2 (manual tuning channel)
  void tuneRx2(double centerMHz, double bwMHz, int? timeoutSeconds) {
    setRxManual(2, centerMHz, bwMHz, timeoutSeconds);
  }

  /// Resume RX2 to idle after manual mode
  void rx2ResumeIdle() {
    setRxIdle(2);
  }

  /// Add a new RX channel (for hardware that supports more)
  void addRx(int rxNumber) {
    if (state.getRx(rxNumber) != null) return;  // Already exists
    
    state = state.copyWith(
      channels: [
        ...state.channels,
        RxChannelState(rxNumber: rxNumber, isConnected: true),
      ],
    );
  }

  /// Remove an RX channel
  void removeRx(int rxNumber) {
    state = state.copyWith(
      channels: state.channels.where((ch) => ch.rxNumber != rxNumber).toList(),
    );
  }
}

/// Provider for multi-RX state
final multiRxProvider = StateNotifierProvider<MultiRxNotifier, MultiRxState>((ref) {
  return MultiRxNotifier();
});

/// Provider for specific RX channel (by number)
final rxChannelProvider = Provider.family<RxChannelState?, int>((ref, rxNumber) {
  return ref.watch(multiRxProvider).getRx(rxNumber);
});

/// Provider for connected RX count
final connectedRxCountProvider = Provider<int>((ref) {
  return ref.watch(multiRxProvider).connectedCount;
});
